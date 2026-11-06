B=/home/cs24d0008/EDC_Segmentation
EDC=$B/EDC-master
HERE=$EDC/runners_edc/BUSI_HGBL
C=$B/COVID19_SEG_HGBL
U=$EDC/segmentation/unet.py
R=$B/HGBL_results
EP=100
wait_mem () { while [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0) -lt $1 ]; do sleep 120; done; }
echo "$(date) start"
if ! grep -q "COMPLETED" $EDC/stage1_covid19.log 2>/dev/null; then
  wait_mem 12000
  echo "$(date) COVID-19 Stage 1"
  cd $EDC
  python runners_edc/BUSI_HGBL/edc_covid19_hgbl.py --data_dir datasets/COVID19_HGBL --val_dir datasets/COVID19_HGBL_VAL --export_dir datasets/COVID19_HGBL_EXPORT --seg_root $C > stage1_covid19.log 2>&1
fi
n=$(ls $C/heatmaps 2>/dev/null | wc -l)
if [ $n -lt 3600 ]; then echo "$(date) STOP: only $n heatmaps, check stage1_covid19.log"; exit 1; fi
cd $HERE
python heatmap_quality.py --dataset covid19 --splits $C/splits --heat_dir $C/heatmaps --out $R
for r in R2 R5 R3; do for s in 1 2 3; do
  if grep -q test_dice $R/covid19/yours/$r/seed$s/summary.json 2>/dev/null; then echo "$r seed$s already done"; continue; fi
  wait_mem 7000
  echo "$(date) COVID-19 $r seed$s"
  CUDA_VISIBLE_DEVICES=0 python train_seg.py --dataset covid19 --run $r --seed $s --epochs $EP --workers 8 --splits $C/splits --heat_dir $C/heatmaps --arch yours --unet_path $U --out $R 2>&1 | grep -E "TEST|Error"
done; done
echo "$(date) ALL DONE"