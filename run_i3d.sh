#!/bin/bash
# Train and evaluate scene-aware dialog models
# Copyright 2018 Mitsubishi Electric Research Labs

. path.sh

stage=1
use_slurm=false
slurm_queue=clusterNew
workdir=`pwd`

# train|val|test sets including dialog text for each video
train_set=data/train_set4DSTC7-AVSD.json
valid_set=data/valid_set4DSTC7-AVSD.json
#test_set=data/test_set4DSTC7-AVSD.json
test_set=data/test_set.json

# directory to read feature files
fea_dir=data/charades_features
# feature file pattern
fea_file="<FeaType>/<ImageID>.npy"
# input feature types
fea_type="i3d_rgb i3d_flow"

# network architecture
# multimodal encoder
enc_psize="512 512"  # dims of projection layers for input features
enc_hsize="0 0"    # dims of cell states (0: no LSTM layer)
att_size=128       # dim to decide temporal attention
mout_size=256      # dim of final projection layer
# input (question) encoder
embed_size=100
in_enc_layers=2
in_enc_hsize=128
# hierarchical history encoder
hist_enc_layers="2 1"  # numbers of word-level layers & QA-pair layers
hist_enc_hsize=128     # dim of hidden layer
hist_out_size=128      # dim of final projection layer
# response (answer) decoder
dec_layers=2    # number of layers
dec_psize=128   # dim of word-embedding layer
dec_hsize=128   # dim of cell states

# training params
num_epochs=15   # number of maximum epochs
batch_size=64   # batch size
max_length=256  # batch size is reduced if len(input_feature) >= max_length
optimizer=Adam  # SGD|AdaDelta|RMSprop
seed=1          # random seed

# generator params
beam=5          # beam width
penalty=1.0     # penalty added to the score of each hypothesis
nbest=5         # number of hypotheses to be output
model_epoch=best  # model epoch number to be used

. utils/parse_options.sh || exit 1;


# directory and feature file setting
enc_psize_=`echo $enc_psize|sed "s/ /-/g"`
enc_hsize_=`echo $enc_hsize|sed "s/ /-/g"`
fea_type_=`echo $fea_type|sed "s/ /-/g"`

expdir=exp/avsd_${fea_type_}_${optimizer}_ep${enc_psize_}_eh${enc_hsize_}_dp${dec_psize}_dh${dec_hsize}_att${att_size}_bs${batch_size}_seed${seed}

# command settings
if [ $use_slurm = true ]; then
  train_cmd="srun --job-name train -X --chdir=$workdir --gres=gpu:1 -p $slurm_queue"
  test_cmd="srun --job-name test -X --chdir=$workdir --gres=gpu:1 -p $slurm_queue"
  gpu_id=0
else
  train_cmd=""
  test_cmd=""
  gpu_id=`utils/get_available_gpu_id.sh`
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail
#set -x

# preparation
if [ $stage -le 1 ]; then
    echo -------------------------
    echo stage 1: preparation 
    echo -------------------------
    echo setup ms-coco evaluation tool
    if [ ! -d utils/coco-caption ]; then
        git clone https://github.com/tylin/coco-caption utils/coco-caption
        patch -p0 -u < utils/coco-caption.patch
    else
        echo Already exists.
    fi
    echo -------------------------
    echo checking feature files in $fea_dir
    for ftype in $fea_type; do
        if [ ! -d $fea_dir/$ftype ]; then
            echo cannot access: $fea_dir/$ftype
            echo download and extract feature files into the directory
            exit
        fi
        echo ${ftype}: `ls $fea_dir/$ftype | wc -l`
    done
fi

# training phase
mkdir -p $expdir
if [ $stage -le 2 ]; then
    echo -------------------------
    echo stage 2: model training
    echo -------------------------
    $train_cmd code/avsd_train.py \
      --gpu $gpu_id \
      --optimizer $optimizer \
      --fea-type $fea_type \
      --train-path "$fea_dir/$fea_file" \
      --train-set $train_set \
      --valid-path "$fea_dir/$fea_file" \
      --valid-set $valid_set \
      --num-epochs $num_epochs \
      --batch-size $batch_size \
      --max-length $max_length \
      --model $expdir/avsd_model \
      --enc-psize $enc_psize \
      --enc-hsize $enc_hsize \
      --att-size $att_size \
      --mout-size $mout_size \
      --embed-size $embed_size \
      --in-enc-layers $in_enc_layers \
      --in-enc-hsize $in_enc_hsize \
      --hist-enc-layers $hist_enc_layers \
      --hist-enc-hsize $hist_enc_hsize \
      --hist-out-size $hist_out_size \
      --dec-layers $dec_layers \
      --dec-psize $dec_psize \
      --dec-hsize $dec_hsize \
      --rand-seed $seed \
      |& tee $expdir/train.log
fi

# testing phase
if [ $stage -le 3 ]; then
    echo -----------------------------
    echo stage 3: generate responses
    echo -----------------------------
    for data_set in $test_set; do
        echo start response generation for $data_set
        target=$(basename ${data_set%.*})
        result=${expdir}/result_${target}_b${beam}_p${penalty}.json
        test_log=${result%.*}.log
        $test_cmd code/avsd_generate.py \
          --gpu $gpu_id \
          --test-path "$fea_dir/$fea_file" \
          --test-set $data_set \
          --model-conf $expdir/avsd_model.conf \
          --model $expdir/avsd_model_${model_epoch} \
          --beam $beam \
          --penalty $penalty \
          --nbest $nbest \
          --output $result \
         |& tee $test_log
    done
fi

# scoring
if [ $stage -le 4 ]; then
    echo --------------------------
    echo stage 4: score results
    echo --------------------------
    for data_set in $test_set; do
        echo start evaluation for $data_set
        target=$(basename ${data_set%.*})
        result=${expdir}/result_${target}_b${beam}_p${penalty}.json
        reference=${result%.*}_ref.json
        hypothesis=${result%.*}_hyp.json
        result_eval=${result%.*}.eval
        echo Evaluating: $result
        python utils/get_annotation.py -s data/stopwords.txt $data_set $reference
        python utils/get_hypotheses.py -s data/stopwords.txt $result $hypothesis
        python utils/evaluate.py $reference $hypothesis >& $result_eval
        echo Wrote details in $result_eval
        echo "--- summary ---"
        awk '/^(Bleu_[1-4]|METEOR|ROUGE_L|CIDEr):/{print $0; if($1=="CIDEr:"){exit}}'\
            $result_eval
        echo "---------------"
    done
fi
