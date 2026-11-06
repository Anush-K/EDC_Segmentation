# HGBL v2 – corrected Stage 2 pipeline (run this before re-submitting)

Put this folder next to your `segmentation/` folder on the server and copy your
`unet.py` into it (so `--arch yours` uses the same network as before).
Python 3.8 / PyTorch 1.12 compatible. Needs: torch, numpy, opencv, scipy, scikit-learn, pandas.

## What was wrong in the old scripts (found in the cross-check)

| # | Old code (`train_segmentation_*.py`, `dataset_*_seg.py`) | Effect | Fixed in |
|---|---|---|---|
| 1 | Heatmap read *after* augmentation, never flipped/rotated | 4th channel and HGBL weight do not line up with image/mask in training | `seg_data.py` |
| 2 | One dataset object (`train=True`) for train and test | test images were randomly augmented | `train_seg.py` |
| 3 | Best epoch chosen by Dice on the **test** split | optimistic, test-set selection | `train_seg.py` (val split) |
| 4 | `random_split` by slice | LGG: same patient in train and test | `make_splits.py` |
| 5 | `OneCycleLR(steps_per_epoch=len(loader))` but `sched.step()` once per epoch | LR never leaves warm-up (peak 3e-4 is never reached) | `train_seg.py` |
| 6 | Dice/IoU per batch of 8, then averaged | not per-image; last batch weighted wrongly | `metrics.py` |
| 7 | Heatmap read with `cv2.imread(path, 0)` | if `*_map.png` is a JET colour map, high scores become dark (wrong values) | `check_heatmaps.py` |
| 8 | One seed, no p-value, no HD95 | weak evidence | `run_all.sh`, `aggregate.py` |

## Order of work (per dataset: busi, lgg, covid19, kvasir)

**Step 1 – heatmaps (5 min).**
`python check_heatmaps.py --seg_root ../LGG_SEG` → read the RESULT line.
* "JET-coloured": run again with `--convert`; use `--heat_dir ../LGG_SEG/heatmaps_npy`.
* "grayscale": also run `--convert` (just copies to .npy).
* "NOT pure JET (overlay)": re-export the raw `p_all` maps from Stage 1 as .npy (stem.npy).

**Step 2 – splits (5 min).**
BUSI uses YOUR fixed normal split (85 / 21 / 27) and all 647 abnormal images (437 benign + 210 malignant):
```
python make_splits.py --dataset busi --seg_root ../BUSI_SEG \
   --normal_train ../EDC-master/datasets/BUSI/train/NORMAL \
   --normal_val   ../EDC-master/datasets/BUSI/val/NORMAL \
   --normal_test  ../EDC-master/datasets/BUSI/test/NORMAL \
   --stage1_test_abnormal all --merge_masks
```
Expected `counts.txt`: Stage 1 test 27 + 647 = 674; Stage 2 453 / 65 / 129
(benign 306 / 44 / 87, malignant 147 / 21 / 42). `report.txt` lists images with two mask
files (merged with `--merge_masks`) and duplicate images that fall in different splits.
`../BUSI_SEG/images` must hold all 647 abnormal images and `masks/` their masks; the heatmaps of
all 647 come from your Stage 1 test run (all 647 are in the Stage 1 test set).

LGG (patient-level) and the other datasets:
```
python make_splits.py --dataset lgg --seg_root ../LGG_SEG \
   --normal_dirs ../EDC-master/datasets/LGG/train/NORMAL ../EDC-master/datasets/LGG/test/NORMAL \
   --materialize ../EDC-master/datasets/LGG_v2
```
Writes `splits/lgg/*.csv` and `counts.txt` (numbers for Table "Datasets" of the paper).
For busi / covid19 / kvasir use the same command with their folders (split by image).

**Step 3 – Stage 1 again (only needed where the split changed: LGG must; the others recommended).**
Point your `edc_<ds>.py --data_dir` to the materialised folder (`train/NORMAL` for training).
Do NOT select the checkpoint by test AUC: use the final epoch.
Run its evaluation once on `val/` and once on `test/`, save score CSVs, then
`python stage1_threshold.py --val val_scores.csv --test test_scores.csv` → Table 2 numbers.
Export new heatmaps for all Stage 2 images, then repeat Step 1 `--convert`.

**Step 4a – λ grid on validation (test split not used).**
`STAGE=lambda SEEDS="1 2" bash run_all.sh` → `python aggregate.py` prints the λ table.
Choose (λb, λh) with the best validation Dice.

**Step 4b – main grid (7 runs × 5 seeds × 4 datasets = 140 runs).**
`LB=5 LH=1 bash run_all.sh`  (put the chosen λ values). Start with R2 and R5 (Table 9);
R1, R3, R4, R6, R7 are the ablation/controls.
Quick check first: `SEEDS=1 DSETS=busi bash run_all.sh` (BUSI is the smallest).

**Step 5 – heatmap quality** is run automatically inside `run_all.sh` (`heatmap_quality.py`).

**Step 6 – tables.** `python aggregate.py --arch yours` prints LaTeX rows for
Table 9 (Dice/IoU/HD95/BF1 + p), the ablation table, heatmap quality, λ grid.
Edit the numbers in `figsrc/make_result_figures.py` of the paper package and rerun it for Figs 5–6.

Optional second backbone: `ARCH=smp_unet SEEDS="1 2 3" bash run_all.sh` (R2 and R5 are enough).

## How to read the controls (decides the main claim of the paper)
* R5 > R6 and R5 > R7 (significant): the heatmap content helps → keep the claim.
* R5 ≈ R6 ≈ R7: the gain comes from the boundary weight → reframe as a boundary-weighted loss.
* R5 ≈ R2 on a dataset: write "on par", not "improves".

## Tested here
All scripts were run end to end on a small synthetic LGG-like dataset (CPU, 2 epochs,
`--arch attunet`): split building with patient check, JET detection and inversion
(mean error 0.002), joint augmentation alignment check, all 7 runs, aggregation.
NOT tested: your `unet.py`, real data, GPU, convergence.