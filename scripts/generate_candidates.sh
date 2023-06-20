#!/bin/bash

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=1

# MODEL="./output/ere_T5base_trig_arg_notemp_notype/20230612_232653/best_model.mdl"
MODEL="./output/ere_T5base_trig_arg_notemp_notype_new/20230615_231117/best_model.mdl"
CONFIG_EN="./config/config_ere_T5-base_en_old.json"
OUTPUT_DIR="./trigger_candidates/ere/t5base-nofilter10-beam_seqscore_trig_bleu_new/"
echo "======================"
echo "Generating Trigger Candidates"
echo "======================"
python ./src/inference_candidates.py -c $CONFIG_EN -m $MODEL -o $OUTPUT_DIR --beam 10 --beam_group 1 --num_return 10 --type ranktrig


MODEL="./output/ere_T5base_trig_arg_notemp_notype_new/20230616_005535/best_model.mdl"
CONFIG_EN="./config/config_ere_T5-base_en_old.json"
OUTPUT_DIR="./trigger_candidates/ere/t5base-nofilter10-beam_seqscore_trig_bleu_new1/"
echo "======================"
echo "Generating Trigger Candidates"
echo "======================"
python ./src/inference_candidates.py -c $CONFIG_EN -m $MODEL -o $OUTPUT_DIR --beam 10 --beam_group 1 --num_return 10 --type ranktrig


MODEL="./output/ere_T5base_trig_arg_notemp_notype_new/20230616_023524/best_model.mdl"
CONFIG_EN="./config/config_ere_T5-base_en_old.json"
OUTPUT_DIR="./trigger_candidates/ere/t5base-nofilter10-beam_seqscore_trig_bleu_new2/"
echo "======================"
echo "Generating Trigger Candidates"
echo "======================"
python ./src/inference_candidates.py -c $CONFIG_EN -m $MODEL -o $OUTPUT_DIR --beam 10 --beam_group 1 --num_return 10 --type ranktrig


MODEL="./output/ere_T5base_trig_arg_notemp_notype_new/20230616_041126/best_model.mdl"
CONFIG_EN="./config/config_ere_T5-base_en_old.json"
OUTPUT_DIR="./trigger_candidates/ere/t5base-nofilter10-beam_seqscore_trig_bleu_new3/"
echo "======================"
echo "Generating Trigger Candidates"
echo "======================"
python ./src/inference_candidates.py -c $CONFIG_EN -m $MODEL -o $OUTPUT_DIR --beam 10 --beam_group 1 --num_return 10 --type ranktrig


MODEL="./output/ere_T5base_trig_arg_notemp_notype_new/20230616_054733/best_model.mdl"
CONFIG_EN="./config/config_ere_T5-base_en_old.json"
OUTPUT_DIR="./trigger_candidates/ere/t5base-nofilter10-beam_seqscore_trig_bleu_new4/"
echo "======================"
echo "Generating Trigger Candidates"
echo "======================"
python ./src/inference_candidates.py -c $CONFIG_EN -m $MODEL -o $OUTPUT_DIR --beam 10 --beam_group 1 --num_return 10 --type ranktrig

