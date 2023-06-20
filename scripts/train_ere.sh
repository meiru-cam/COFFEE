#!/bin/bash

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

CONFIG="./config/config_ere_T5-base_en.json"

for seed in 0 42 528 622 616
  do python ./src/train_generator.py -c $CONFIG -s $seed
done

# python ./src/train.py -c $CONFIG -s 0
