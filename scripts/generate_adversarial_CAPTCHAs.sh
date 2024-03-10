#!/bin/bash

# Modify ./ACG_Framework/config_stage1.py and ./ACG_Framework/config_stage2.py (default to Geetest)

python ./ACG_Framework/stage1_generate_finegrained_noise_on_character.py

python ./ACG_Framework/stage2_generate_global_noise_on_captcha.py