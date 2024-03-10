#!/bin/bash

python ./Adv_Eval/get_DSR.py \
    --scheme_list geetest \
    --attack_list SR,YV,FA \
    --CAP_dirname_list global_ens_svre_25.5

python ./Adv_Eval/get_RSR_ASR.py \
    --scheme_list geetest \
    --attack_list SR,YV,FA \
    --CAP_dirname_list global_ens_svre_25.5 \
    --metric_list RSR,ASR

python ./Adv_Eval/get_ASR_EM.py \
    --scheme_list geetest \
    --CAP_dirname_list global_ens_svre_25.5

