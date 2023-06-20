#!/bin/bash

export ACE_XU_PATH="./Dataset/ace_2005_Xuetal/en/json"
export OUTPUT_PATH="../processed_data"

export TOKENIZER_NAME='T5'
export PRETRAINED_TOKENIZER_NAME='t5-base'

rm -r $OUTPUT_PATH/ace05_en_$TOKENIZER_NAME
mkdir $OUTPUT_PATH/ace05_en_$TOKENIZER_NAME
for SET in train dev test
do
    python src/process_ace_xuetal.py -i $ACE_XU_PATH/${SET}.json -o $OUTPUT_PATH/ace05_en_$TOKENIZER_NAME/${SET}.json -b $PRETRAINED_TOKENIZER_NAME -w 1 -l english
done
