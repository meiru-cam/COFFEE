#!/bin/bash

export ACE_PATH="./Dataset/ace_2005_td_v7/data/"
export OUTPUT_PATH="../processed_data"

export TOKENIZER_NAME='T5'
export PRETRAINED_TOKENIZER_NAME='t5-base'

rm -r $OUTPUT_PATH/ace05p_en_$TOKENIZER_NAME
mkdir $OUTPUT_PATH/ace05p_en_$TOKENIZER_NAME

python src/process_ace.py -i $ACE_PATH -o $OUTPUT_PATH/ace05p_en_$TOKENIZER_NAME -s src/splits/ACE-EN -b $PRETRAINED_TOKENIZER_NAME -l english
