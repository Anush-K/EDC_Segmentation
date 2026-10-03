#!/bin/bash
run() { PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python3 runners_edc/edc_$1.py -sn $2 --use_rqasw $3 --gpu 1 > log_$2.txt 2>&1; }
run isic2018 isic_edc_last False
run aptos    aptos_rq_last  True
run aptos    aptos_edc_last False
run br35h    br35h_rq_last  True
run br35h    br35h_edc_last False
