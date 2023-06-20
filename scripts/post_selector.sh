#!/bin/bash

vocab="./finetuned_data/ere_T5_trig_arg_notemp_notype/vocab.json"
refevent="./finetuned_data/ere_test_all_event.txt"
argoutput="./finetuned_data/ere_test_all_argoutput.txt"

model_path="./output/ere_T5base_trig_arg_notemp_notype/20230612_232653/best_model.mdl"
config="./config/config_ere_T5-base_en.json"
arginput="./cache_eval/ere/20230614_164858/best_weight/test_ranked_True_weight_0.150_threshold_0.150_arginput.txt"


echo $arginput && python ./src/post_selector.py --arginput_path $arginput --argoutput_path $argoutput \
     --ckpt_path $model_path --config_path $config --vocab_path $vocab --beam_size 4 --refevent_path $refevent


# for argin in $(ls ./cache_eval/20220826_005717/plots_weight/test_ranked* -d)
#   do echo $argin && python ./xgear/post_ranker.py --arginput_path $argin --argoutput_path $argoutput \
#     --ckpt_path $model_path --config_path $config --vocab_path $vocab --beam_size 1
#   done

