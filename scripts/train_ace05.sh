#!/bin/bash

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

CONFIG="./config/config_ace05p_T5-base_en.json"

# CONFIG="./config/config_ace05_bart-large_en.json"
# CONFIG="./config/config_ace05_T5-large_en.json"

for seed in 0 42 528 622
  do python ./coffee/train.py -c $CONFIG -s $seed
done
