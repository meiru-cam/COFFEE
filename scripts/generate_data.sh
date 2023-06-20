#!/bin/bash

# CONFIG_EN="./config/config_ace05_T5-base_en.json"
# python ./src/generate_data.py -c $CONFIG_EN

# CONFIG_EN="./config/config_ace05p_T5-base_en.json"
# python ./src/generate_data.py -c $CONFIG_EN

CONFIG_EN="./config/config_ere_T5-base_en.json"
python ./src/generate_data.py -c $CONFIG_EN
 