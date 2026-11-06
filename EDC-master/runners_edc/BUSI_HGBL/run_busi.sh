#!/usr/bin/env bash
# Full BUSI pipeline for the HGBL paper. Run step by step:
#   bash run_busi.sh 1   data                bash run_busi.sh 2   Stage 1 (5 seeds)
#   bash run_busi.sh 3   quick Stage 2 test  bash run_busi.sh 4   lambda grid (validation)
#   bash run_busi.sh 5   main grid R1-R7 x 5 seeds                bash run_busi.sh 6   tables
#
# Layout assumed:
#   EDC-master/runners_edc/BUSI_HGBL/  busi_1_prepare_data.py, edc_busi_hgbl.py, run_busi.sh
#   EDC-master/runners_edc/            train_seg.py, seg_data.py, losses.py, metrics.py,
#                                      hgbl_models.py, common.py, aggregate.py, ... (HGBL_v2 files)
#   EDC-master/segmentation/unet.py    your UNet, used as it is (not copied, not renamed)
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # .../runners_edc/BUSI_HGBL
BASE=/home/cs24d0008/EDC_Segmentation
EDC="$BASE/EDC-master"
BUSI_ORIG="$EDC/Breast BUSI/Dataset_BUSI_with_GT"            # original BUSI: normal/ benign/ malignant/
OLD_SPLIT="$EDC/BUSI"                                         # old split: its 27 test normals are kept
SEG="$BASE/BUSI_SEG_HGBL"                                     # Stage 2 data (new folder)
HGBL="$EDC/runners_edc"                                       # where the HGBL_v2 .py files are
UNET="$EDC/segmentation/unet.py"                              # your UNet file
RES="$BASE/HGBL_results"                                      # all Stage 2 results
LB=${LB:-5}; LH=${LH:-1}
SEG_ARGS=(--splits "$SEG/splits" --heat_dir "$SEG/heatmaps" --arch yours --unet_path "$UNET" --out "$RES")

case "$1" in
1)  echo "normal train: $(ls "$OLD_SPLIT/train/NORMAL" | wc -l) (expect 106)   normal test: $(ls "$OLD_SPLIT/test/NORMAL" | wc -l) (expect 27)"
    python "$HERE/busi_1_prepare_data.py" --busi_root "$BUSI_ORIG" --data_root "$EDC/datasets" \
        --seg_root "$SEG" --normal_split_from "$OLD_SPLIT" \
        --exclude_stage2 "benign (433)" "malignant (145)"   # identical image with two labels
    cat "$SEG/busi_report.txt" ;;
2)  cd "$EDC"
    python "$HERE/edc_busi_hgbl.py" --data_dir datasets/BUSI_HGBL --val_dir datasets/BUSI_HGBL_VAL \
        --seg_root "$SEG" --save_name edc_busi_hgbl
    python "$HGBL/check_heatmaps.py" --seg_root "$SEG" ;;
3)  cd "$HGBL"
    python train_seg.py --dataset busi --run R5 --seed 1 --epochs 5 "${SEG_ARGS[@]/$RES/$BASE/HGBL_quicktest}" ;;
4)  cd "$HGBL"
    for lb in 1 3 5 10; do for lh in 0 0.5 1 2; do for s in 1 2; do
      python train_seg.py --dataset busi --run R5 --seed $s "${SEG_ARGS[@]}" \
          --lam_b $lb --lam_h $lh --tag lb${lb}_lh${lh} --no_test
    done; done; done
    python aggregate.py --arch yours --results "$RES" --splits "$SEG/splits" | sed -n '/Lambda grid/,$p' ;;
5)  cd "$HGBL"
    python heatmap_quality.py --dataset busi --splits "$SEG/splits" --heat_dir "$SEG/heatmaps" --out "$RES"
    for run in R2 R5 R1 R3 R4 R6 R7; do for s in 1 2 3 4 5; do
      python train_seg.py --dataset busi --run $run --seed $s "${SEG_ARGS[@]}" --lam_b $LB --lam_h $LH
    done; done ;;
6)  cd "$HGBL"
    python aggregate.py --arch yours --results "$RES" --splits "$SEG/splits"
    cat "$EDC/saved_models/edc_busi_hgbl/stage1_results_busi.json" ;;
*)  echo "usage: bash run_busi.sh 1|2|3|4|5|6" ;;
esac