# Audio Visual Scene-aware Dialog (AVSD) example for DSTC7

## Required packages

- python 2.7
- pytorch 0.4.1
- numpy
- six
- java 1.8.0   (for coco-evaluation tools)

## How to run the code:

   1. Obtain the dialog data files (.json) and store them in `data/`.
      - `train_set4DSTC7-AVSD.json` (official training set)
      - `valid_set4DSTC7-AVSD.json` (official validation set)
      - `test_set4DSTC7-AVSD.json` (official test set. this file does not include groundtruth at this moment)
      - `test_set.json`  (prototype test set used for tentative evaluation)
 
   2. Obtain the feature data and extract files under `data/charades_features/`.
      - `i3d_rgb`, `i3d_flow`, `vggish`  (train and validation sets)
      - `i3d_rgb_testset`, `i3d_flow_testset`, `vggish_testset` (official test sets)

   3. Run `run_i3d.sh` to train and test the network, that uses `i3d_rgb` and `i3d_flow` features. (`run_i3d+vggish.sh` is another example that further uses `vggish` features)

   4. Model files and generated sentences will be stored in your experiment related folder under `exp/`, where `result_test_set_b5_p1.0.json` includes the generated sentences for the test set.

   5. You can see the evaluation result `result_test_set_b5_p1.0.eval` in the folder. Note that the test set is not the official one.
      - if you generate sentences for the official test set, you can run the script as
      
            run_i3d.sh --stage 3 --test-set data/test_set4DSTC7-AVSD.json --fea-file "<FeaType>_testset/<ImageID>.npy"

## Result:

The following results were obtained for the official validation set by using `run_i3d.sh`.

| METRIC | RESULT |
| ------ | -------|
| Bleu_1 | 0.273  |
| Bleu_2 | 0.173  |
| Bleu_3 | 0.118  |
| Bleu_4 | 0.084  |
| METEOR | 0.117  |
| ROUGE_L| 0.291  |
| CIDEr  | 0.766  |

