# 3-day GPU plan (about 55-65 h of runs). Each run waits for 7 GB of free GPU memory and is
# skipped if it is already finished, so the script can be restarted at any time.
#   0. wait until the COVID-19 chain (chain_covid.sh) has finished
#   1. COVID-19 seeds 4-5 : R2, R5, R3        (100 epochs, as decided)      ~15 h
#   2. BUSI  ablation rows: R1 (3-ch, no heatmap), R4 (heatmap factor only), seeds 1-3   ~6 h
#   3. LGG   ablation rows: R1, R4, seeds 1-3                                ~18 h
#   4. Kvasir ablation rows: R1, R4, seeds 1-3 (if time is left)             ~10 h
B=/home/cs24d0008/EDC_Segmentation
EDC=$B/EDC-master
HERE=$EDC/runners_edc/BUSI_HGBL
U=$EDC/segmentation/unet.py
R=$B/HGBL_results
cd $HERE
wait_mem () { while [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0) -lt 7000 ]; do sleep 120; done; }
run () {   # dataset run seed seg_root [extra args]
  ds=$1; r=$2; s=$3; root=$4; shift 4
  if grep -q test_dice $R/$ds/yours/$r/seed$s/summary.json 2>/dev/null; then echo "$ds $r seed$s already done"; return; fi
  wait_mem
  echo "$(date) $ds $r seed$s"
  CUDA_VISIBLE_DEVICES=0 python train_seg.py --dataset $ds --run $r --seed $s --workers 8 --splits $root/splits --heat_dir $root/heatmaps --arch yours --unet_path $U --out $R "$@" 2>&1 | grep -E "TEST|Error"
}
echo "$(date) start (waiting for the COVID-19 chain)"
while pgrep -f "chain_covid.sh" > /dev/null; do sleep 300; done
echo "$(date) COVID-19 chain finished"
for s in 4 5; do for r in R2 R5 R3; do run covid19 $r $s $B/COVID19_SEG_HGBL --epochs 100; done; done
for r in R1 R4; do for s in 1 2 3; do run busi $r $s $B/BUSI_SEG_HGBL; done; done
for r in R1 R4; do for s in 1 2 3; do run lgg $r $s $B/LGG_SEG_HGBL; done; done
for r in R1 R4; do for s in 1 2 3; do run kvasir $r $s $B/KVASIR_SEG_HGBL; done; done
echo "$(date) ALL DONE"