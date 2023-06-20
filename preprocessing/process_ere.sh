#!/bin/bash
export PYTHONWARNINGS=ignore

#==============================================================#
export ERE_PATH="./Dataset/DEFT_English_Light_and_Rich_ERE_Annotation/data/rich_ere"
export OUTPUT_PATH="../processed_data"

export TOKENIZER_NAME='T5'
export PRETRAINED_TOKENIZER_NAME='t5-base'

rm -r $OUTPUT_PATH/ere_en_$TOKENIZER_NAME
mkdir $OUTPUT_PATH/ere_en_$TOKENIZER_NAME

python src/process_ere.py -i $ERE_PATH -o $OUTPUT_PATH/ere_en_$TOKENIZER_NAME -s src/splits/ERE-EN -b $PRETRAINED_TOKENIZER_NAME -l english
