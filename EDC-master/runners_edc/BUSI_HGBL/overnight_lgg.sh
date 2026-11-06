#!/usr/bin/env bash
# Overnight LGG chain (about 10-12 GPU hours + waiting). Every step waits for free GPU memory,
# skips work that is already finished, and writes its own log.
#   1. LGG R3 seeds 4-5            (skipped if done / waits if another R3 queue is running)
#   2. skull-strip Stage 1 images  (CPU)
#   3. Stage 1 LGG on skull-stripped images, 5 seeds  -> heatmaps in LGG_SEG_HGBL_SS
#   4. heatmap quality of the new heatmaps on VALIDATION (CPU) -> scan_lgg_ss.log
B=/home/cs24d0008/EDC_Segmentation
EDC=$B/EDC-master
HERE=$EDC/runners_edc/BUSI_HGBL
L=$B/LGG_SEG_HGBL
U=$EDC/segmentation/unet.py
R=$B/HGBL_results
wait_mem () { while [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0) -lt $1 ]; do sleep 120; done; }
cd $HERE
echo "$(date) start"

# 1. LGG R3 seeds 4-5
while pgrep -f "train_seg.py --dataset lgg --run R3" > /dev/null; do sleep 300; done
for s in 4 5; do
  if grep -q test_dice $R/lgg/yours/R3/seed$s/summary.json 2>/dev/null; then echo "R3 seed$s already done"; continue; fi
  wait_mem 7000
  echo "$(date) LGG R3 seed$s"
  CUDA_VISIBLE_DEVICES=0 python train_seg.py --dataset lgg --run R3 --seed $s --workers 8 \
     --splits $L/splits --heat_dir $L/heatmaps --arch yours --unet_path $U --out $R 2>&1 | grep -E "TEST|Error"
done

# 2. skull strip (CPU)
if [ ! -d $EDC/datasets/LGG_HGBL_EXPORT_SS ]; then
  echo "$(date) skull strip"; python lgg_skullstrip.py --data_root $EDC/datasets --erode 10
fi

# 3. Stage 1 on skull-stripped images
if ! grep -q "COMPLETED" $EDC/stage1_lgg_ss.log 2>/dev/null; then
  wait_mem 12000
  echo "$(date) Stage 1 LGG skull-stripped"
  cd $EDC
  python runners_edc/edc_lgg_hgbl.py --data_dir datasets/LGG_HGBL_SS --val_dir datasets/LGG_HGBL_VAL_SS \
     --export_dir datasets/LGG_HGBL_EXPORT_SS --seg_root $B/LGG_SEG_HGBL_SS --save_name edc_lgg_hgbl_ss > stage1_lgg_ss.log 2>&1
  cd $HERE
fi

# 4. heatmap quality on validation
mkdir -p $B/LGG_SEG_HGBL_SS/splits && cp -r $L/splits/lgg $B/LGG_SEG_HGBL_SS/splits/
echo "$(date) heatmap scan"
python heatmap_postprocess.py --dataset lgg --seg_root $B/LGG_SEG_HGBL_SS --scan > scan_lgg_ss.log 2>&1
echo "$(date) ALL DONE"