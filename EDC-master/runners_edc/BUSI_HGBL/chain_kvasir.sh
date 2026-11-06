B=/home/cs24d0008/EDC_Segmentation
EDC=$B/EDC-master
HERE=$EDC/runners_edc/BUSI_HGBL
K=$B/KVASIR_SEG_HGBL
U=$EDC/segmentation/unet.py
R=$B/HGBL_results
wait_mem () { while [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0) -lt $1 ]; do sleep 120; done; }
echo "$(date) start"
if ! grep -q "COMPLETED" $EDC/stage1_kvasir.log 2>/dev/null; then
  wait_mem 10000
  echo "$(date) Kvasir Stage 1"
  cd $EDC
  python runners_edc/edc_kvasir_hgbl.py --data_dir datasets/KVASIR_HGBL --val_dir datasets/KVASIR_HGBL_VAL --export_dir datasets/KVASIR_HGBL_EXPORT --seg_root $K > stage1_kvasir.log 2>&1
fi
n=$(ls $K/heatmaps 2>/dev/null | wc -l)
if [ $n -lt 1000 ]; then echo "$(date) STOP: only $n heatmaps, check stage1_kvasir.log"; exit 1; fi
cd $HERE
python heatmap_quality.py --dataset kvasir --splits $K/splits --heat_dir $K/heatmaps --out $R
for r in R2 R5 R3; do for s in 1 2 3; do
  if grep -q test_dice $R/kvasir/yours/$r/seed$s/summary.json 2>/dev/null; then echo "$r seed$s already done"; continue; fi
  wait_mem 7000
  echo "$(date) Kvasir $r seed$s"
  CUDA_VISIBLE_DEVICES=0 python train_seg.py --dataset kvasir --run $r --seed $s --workers 8 --splits $K/splits --heat_dir $K/heatmaps --arch yours --unet_path $U --out $R 2>&1 | grep -E "TEST|Error"
done; done
echo "$(date) ALL DONE"
