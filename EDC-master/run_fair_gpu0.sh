#!/bin/bash
run() {
  name=$2
  logfile="log_${2}_v2.txt"
  for attempt in 1 2 3 4 5 6; do
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
    if [ "$free_mib" -lt 6000 ]; then
      echo "[$(date)] attempt $attempt: only ${free_mib}MiB free, waiting..." >> "$logfile"
      sleep 300
      continue
    fi
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
      python3 runners_edc/edc_$1.py -sn $2 --use_rqasw $3 --gpu 0 > "$logfile" 2>&1
    if grep -q "RuntimeError" "$logfile"; then
      echo "[$(date)] attempt $attempt failed, retrying..." >> "$logfile"
      sleep 300
      continue
    fi
    break
  done
}
run isic2018 isic_edc_last False
run aptos    aptos_rq_last  True
run aptos    aptos_edc_last False
run br35h    br35h_rq_last  True
run br35h    br35h_edc_last False
