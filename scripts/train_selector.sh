#!/bin/bash


# finetune margin, span, and test_candi_num
lr=5e-4
upd=30
bs=4
lm=0.2
span=5
test_candi_num=3
tw=0.2
thres=0.4
data=./trigger_candidates/ere/t5base-nofilter10-beam_seqscore_trig_bleu2
seed=11
for lm in 0.2 0.3 0.4
  do for span in 5 7
    do for test_candi_num in 3 5
      do python ./src/train_selector.py --model_name bert-base-uncased \
          --batch_size $bs --total_steps 5000 --update_steps $upd \
          --warmup_steps 1000 --lr $lr --candidates_path $data \
          --train_candi_pool_size $span --train_negative_num $span --test_candi_span $test_candi_num --loss_margin $lm \
          --tune_weight $tw --threshold $thres --weight --seed $seed
      done
    done
  done


data=./trigger_candidates/ere/t5base-nofilter10-beam_seqscore_trig_bleu3
for lm in 0.2 0.3 0.4
  do for span in 5 7
    do for test_candi_num in 3 5
      do python ./src/train_selector.py --model_name bert-base-uncased \
          --batch_size $bs --total_steps 5000 --update_steps $upd \
          --warmup_steps 1000 --lr $lr --candidates_path $data \
          --train_candi_pool_size $span --train_negative_num $span --test_candi_span $test_candi_num --loss_margin $lm \
          --tune_weight $tw --threshold $thres --weight --seed $seed
      done
    done
  done

data=./trigger_candidates/ere/t5base-nofilter10-beam_seqscore_trig_bleu4
for lm in 0.2 0.3 0.4
  do for span in 5 7
    do for test_candi_num in 3 5
      do python ./src/train_selector.py --model_name bert-base-uncased \
          --batch_size $bs --total_steps 5000 --update_steps $upd \
          --warmup_steps 1000 --lr $lr --candidates_path $data \
          --train_candi_pool_size $span --train_negative_num $span --test_candi_span $test_candi_num --loss_margin $lm \
          --tune_weight $tw --threshold $thres --weight --seed $seed
      done
    done
  done
