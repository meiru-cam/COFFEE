#!/bin/bash

## train cross ranker
# python ./xgear/cl_eval_cross.py


## evaluation
# export CUDA_VISIBLE_DEVICES=0
lr=5e-5
upd=30
data=./trigger_candidates/ere/t5base-nofilter10-beam_seqscore_trig_bleu
model='./cache_eval/ere/20230614_155827/'

for model in $(ls ./cache_eval/ere/20230614*/ -d .)
  do python ./src/train_selector.py --model_name bert-base-uncased \
      --batch_size 128 --total_steps 50000 --update_steps $upd \
      --warmup_steps 1000 --lr $lr --candidates_path $data \
      --train_candi_pool_size 10 --train_negative_num 10 --test_candi_span 5 \
      --ckpt_path $model \
      --save_prefix $model/best_weight/ --evaluate --weight --use_rank --save_pred
  done
# Evaluation on test, using the optimal hyperparameter
# best params  {'weight': 0.3, 'threshold': 0.15000000000000002, 'tau_gen': 0, 'tau_rank': 1}
# current model  20230614_155827
# Trigger I ---- tp/total_g/total_p  201/ 309/ 293, f1  66.78
# Trigger C ---- tp/total_g/total_p  176/ 309/ 293, f1  58.47
