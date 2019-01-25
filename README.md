# Audio Visual Scene-aware Dialog (AVSD) example for DSTC7

This is an implementation of the naive fusion of audio and video features targeted at the AVSD Challenge at DSTC7. Details of our scheme are in the following paper. While, the results in paper used a Chainer implementation of the system, the provided code is in PyTorch (translated from Chainer).

Please cite the paper for the baseline system:

        @inproceedings{alamri@DSTC7,
                title={Audio Visual Scene-aware dialog (AVSD) Track for Natural Language Generation in DSTC7},
                author={Huda Alamri and Chiori Hori and Tim K. Marks and Dhruv Batra and Devi Parikh},
                booktitle={DSTC7 at AAAI2019 Workshop},
                year={2018}
        }

Please cite the paper for attentional multimoda fusion.
https://arxiv.org/abs/1806.08409

      @article{hori2018end,
        title={End-to-End Audio Visual Scene-Aware Dialog using Multimodal Attention-Based Video Features},
        author={Hori, Chiori and Alamri, Huda and Wang, Jue and Winchern, Gordon and Hori, Takaaki and Cherian, Anoop and Marks, Tim K and Cartillier, Vincent and Lopes, Raphael Gontijo and Das, Abhishek and others},
        journal={arXiv preprint arXiv:1806.08409},
        year={2018}
      }

## Required packages

- python 2.7
- pytorch 0.4.1
- numpy
- six
- java 1.8.0   (for coco-evaluation tools)

## DSTC7 AVSD track data links

   https://github.com/hudaAlamri/DSTC7-Audio-Visual-Scene-Aware-Dialog-AVSD-Challenge

## How to run the code:

   1. Obtain the dialog data files (.json) and store them in `data/`.
      - `train_set4DSTC7-AVSD.json` (official training set)
      - `valid_set4DSTC7-AVSD.json` (official validation set)
      - `test_set4DSTC7-AVSD.json` (official test set. this file does not include groundtruth at this moment)
      - `test_set.json`  (prototype test set used for tentative evaluation)
 
   2. Make directory `data/charades_features` and extract files under the directory from downloaded feature packages:
      - `i3d_rgb.tgz`, `i3d_flow.tgz`, `vggish.tgz`  (train and validation sets)
      - `i3d_rgb_testset.tgz`, `i3d_flow_testset.tgz`, `vggish_testset.tgz` (official test sets)

   3. Run `run_i3d.sh` to train and test the network, that uses `i3d_rgb` and `i3d_flow` features. (`run_i3d+vggish.sh` is another example that further uses `vggish` features)

   4. Model files and generated sentences will be stored in your experiment related folder under `exp/`, where `result_test_set_b5_p1.0.json` includes the generated sentences for the test set.

   5. You can see the evaluation result `result_test_set_b5_p1.0.eval` in the folder. Note that the test set is not the official one.
      - if you generate sentences for the official test set, you can run the script as
      
            run_i3d.sh --stage 3 --test-set data/test_set4DSTC7-AVSD.json --fea-file "<FeaType>_testset/<ImageID>.npy"

## Result:

The following results were obtained for the "prototype" test set by using `run_i3d.sh`.


| METRIC | RESULT |
| ------ | -------|
| Bleu_1 | 0.273  |
| Bleu_2 | 0.173  |
| Bleu_3 | 0.118  |
| Bleu_4 | 0.084  |
| METEOR | 0.117  |
| ROUGE_L| 0.291  |
| CIDEr  | 0.766  |

