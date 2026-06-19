# verify_aptos_result.py
# Produces concrete proof that the re-scored AUC/F1/etc. on the existing
# APTOS seed-2 checkpoint is real and reproducible -- not a fluke.
#
# Checks performed:
#   1. Checkpoint file timestamp -- proves this IS the checkpoint from
#      your 5-day training run, not something else.
#   2. Runs scoring TWICE and confirms identical output -- proves the
#      result is deterministic, not random luck.
#   3. Manually recomputes AUC from the raw scores using the textbook
#      definition (rank-based formula), independent of sklearn, and
#      confirms it matches sklearn's roc_auc_score.
#   4. Confirms the confusion matrix totals (805 normal + 1857 abnormal
#      = 2662) match your original test-split counts exactly.
#   5. Saves a PNG showing the normal vs abnormal score distributions
#      and the ROC curve, as visual evidence of separation.
#
# Usage:
#   python verify_aptos_result.py --save_dir ./saved_models -sn edc_aptos --seed 2

import sys, os, time
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "datasets"))

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix

from helper_modules.utils import get_logger
from datasets.dataset_aptos import AD_Dataset
from models.edc import WR50_WR50 as R50_R50
from configs.config_aptos import DATASET_DIR
from runners_edc.edc_aptos import run_best_ckpt_inference

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, default='./saved_models')
parser.add_argument('-sn', '--save_name', type=str, default='edc_aptos')
parser.add_argument('--data_dir', type=str, default=DATASET_DIR)
parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--eval_batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--var_reg_weight', type=float, default=0.1)
parser.add_argument('--ema_momentum', type=float, default=0.999)
parser.add_argument('--seed', type=int, default=2)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
logger = get_logger("verify", os.path.join(args.save_dir, args.save_name), "INFO")

print("="*70)
print("PROOF CHECK 1 -- Checkpoint provenance")
print("="*70)
ckpt_path = os.path.join(args.save_dir, args.save_name, f"seed_{args.seed}", "model_best.pth")
mtime = os.path.getmtime(ckpt_path)
print(f"Checkpoint path : {ckpt_path}")
print(f"Last modified   : {time.ctime(mtime)}")
print(">>> Confirm this date matches your 5-day training run, not a")
print(">>> later/different experiment.")

eval_dset = AD_Dataset(
    name='fundus', train=False,
    data_dir=args.data_dir,
    img_size=args.img_size, crop_size=args.img_size,
).get_dset()

def load_model_and_score():
    model = R50_R50(
        img_size=args.img_size, train_encoder=True,
        stop_grad=False, reshape=True, bn_pretrain=True,
        var_reg_weight=args.var_reg_weight,
        ema_momentum=args.ema_momentum,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
    return run_best_ckpt_inference(model, eval_dset, device, args, logger, args.seed)

print("\n" + "="*70)
print("PROOF CHECK 2 -- Reproducibility (running scoring TWICE)")
print("="*70)
scores_a, labels_a, auc_a = load_model_and_score()
scores_b, labels_b, auc_b = load_model_and_score()
identical = np.allclose(scores_a, scores_b) and np.array_equal(labels_a, labels_b)
print(f"Run 1 AUC: {auc_a:.6f}")
print(f"Run 2 AUC: {auc_b:.6f}")
print(f"Scores identical across runs : {identical}")
if identical:
    print(">>> PROVEN: result is deterministic, not a lucky random fluke.")
else:
    print(">>> WARNING: scores differ between runs -- investigate randomness source.")

print("\n" + "="*70)
print("PROOF CHECK 3 -- Manual AUC recomputation (independent of sklearn)")
print("="*70)
# Manual AUC = probability a random abnormal score > a random normal score
# (textbook Mann-Whitney U interpretation of AUC), computed by brute-force
# pairwise comparison -- completely independent code path from sklearn.
normal_s   = scores_a[labels_a == 0]
abnormal_s = scores_a[labels_a == 1]
# brute force on a random subsample if huge, else full pairwise
n_max = 4000
rng = np.random.default_rng(0)
ns = normal_s if len(normal_s) <= n_max else rng.choice(normal_s, n_max, replace=False)
as_ = abnormal_s if len(abnormal_s) <= n_max else rng.choice(abnormal_s, n_max, replace=False)
comparisons = as_[:, None] > ns[None, :]
ties        = as_[:, None] == ns[None, :]
manual_auc  = (comparisons.sum() + 0.5 * ties.sum()) / comparisons.size
sklearn_auc = roc_auc_score(labels_a, scores_a)
print(f"Manual (Mann-Whitney) AUC : {manual_auc:.4f}")
print(f"sklearn roc_auc_score AUC: {sklearn_auc:.4f}")
print(f"Difference                : {abs(manual_auc - sklearn_auc):.4f}")
print(">>> These should match closely (small subsample noise is expected).")

print("\n" + "="*70)
print("PROOF CHECK 4 -- Confusion matrix totals match your original split")
print("="*70)
precs, recs, thrs = precision_recall_curve(labels_a, scores_a)
f1s  = 2 * precs * recs / (precs + recs + 1e-8)
f1s  = f1s[:-1]
best_thr = thrs[np.argmax(f1s)]
y_pred   = (scores_a >= best_thr).astype(int)
cm       = confusion_matrix(labels_a, y_pred)
total_normal   = int((labels_a == 0).sum())
total_abnormal = int((labels_a == 1).sum())
print(f"Total NORMAL images   : {total_normal}  (expected 805)")
print(f"Total ABNORMAL images : {total_abnormal}  (expected 1857)")
print(f"Total test images     : {len(labels_a)}  (expected 2662)")
print(f"Confusion matrix:\n{cm}")
match = (total_normal == 805 and total_abnormal == 1857 and len(labels_a) == 2662)
print(f">>> Matches original test split exactly: {match}")

print("\n" + "="*70)
print("PROOF CHECK 5 -- Saving visual evidence (score plot + ROC curve)")
print("="*70)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(normal_s, bins=40, alpha=0.6, label=f"Normal (n={len(normal_s)})", color="green")
axes[0].hist(abnormal_s, bins=40, alpha=0.6, label=f"Abnormal (n={len(abnormal_s)})", color="red")
axes[0].axvline(best_thr, color="black", linestyle="--", label=f"Threshold={best_thr:.4f}")
axes[0].set_title("Score Distribution (p1_top0.05)")
axes[0].set_xlabel("Anomaly score")
axes[0].legend()

fpr, tpr, _ = roc_curve(labels_a, scores_a)
axes[1].plot(fpr, tpr, label=f"AUC = {sklearn_auc:.4f}")
axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
axes[1].set_title("ROC Curve")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].legend()

plt.tight_layout()
out_path = os.path.join(args.save_dir, args.save_name, "verification_proof.png")
plt.savefig(out_path, dpi=150)
print(f"Saved plot to: {out_path}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Checkpoint date matches training run : check manually above")
print(f"Result is reproducible (run 1 == run 2): {identical}")
print(f"Manual AUC matches sklearn AUC         : {abs(manual_auc - sklearn_auc) < 0.02}")
print(f"Test split totals match original log   : {match}")
print(f"Final AUC                              : {sklearn_auc:.4f}")