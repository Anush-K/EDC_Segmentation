import sys, os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "datasets"))

import argparse
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix

from helper_modules.utils import get_logger
from datasets.dataset_aptos import AD_Dataset
from datasets.data_utils import get_data_loader
from models.edc import WR50_WR50 as R50_R50
from configs.config_aptos import DATASET_DIR
from runners_edc.edc_aptos import topk_score

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
logger = get_logger("quick_rescore", os.path.join(args.save_dir, args.save_name), "INFO")

eval_dset = AD_Dataset(
    name='fundus', train=False,
    data_dir=args.data_dir,
    img_size=args.img_size, crop_size=args.img_size,
).get_dset()

model = R50_R50(
    img_size=args.img_size, train_encoder=True,
    stop_grad=False, reshape=True, bn_pretrain=True,
    var_reg_weight=args.var_reg_weight,
    ema_momentum=args.ema_momentum,
).to(device)

ckpt_path = os.path.join(args.save_dir, args.save_name, f"seed_{args.seed}", "model_best.pth")
print(f"Loading checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)

loader = get_data_loader(eval_dset, args.eval_batch_size,
                          num_workers=args.num_workers, drop_last=False)

raw_pall, raw_p1, labels = [], [], []
model.eval()
with torch.no_grad():
    for batch in loader:
        idx, x, mask, y, fname = batch
        x = x.to(device)
        result = model(x)
        raw_pall.append(result['p_all'].detach().cpu())
        raw_p1.append(result['p1'].detach().cpu())
        labels.extend(y.numpy())

raw_pall = torch.cat(raw_pall, dim=0)
raw_p1   = torch.cat(raw_p1, dim=0)
labels   = np.array(labels)

k_ratios = (0.005, 0.01, 0.02, 0.05)
candidates = {}
candidates['p_all_max'] = raw_pall.flatten(1).amax(dim=1).numpy()
candidates['p1_max']    = raw_p1.flatten(1).amax(dim=1).numpy()
for kr in k_ratios:
    candidates[f'p_all_top{kr}'] = topk_score(raw_pall, kr).numpy()
    candidates[f'p1_top{kr}']    = topk_score(raw_p1, kr).numpy()

def full_row(name, scores, labels):
    precs, recs, thrs = precision_recall_curve(labels, scores)
    f1s = 2 * precs * recs / (precs + recs + 1e-8)
    f1s = f1s[:-1]
    thr = thrs[np.argmax(f1s)]
    f1  = f1s[np.argmax(f1s)]
    y_pred = (scores >= thr).astype(int)
    cm = confusion_matrix(labels, y_pred)
    tn, fp_, fn_, tp_ = cm.ravel()
    spec = tn / (tn + fp_ + 1e-8)
    rec  = tp_ / (tp_ + fn_ + 1e-8)
    acc  = (tp_ + tn) / len(labels)
    auc  = roc_auc_score(labels, scores)
    print(f"{name:<16}{auc:>8.4f}{f1:>8.4f}{acc:>8.4f}{rec:>8.4f}{spec:>8.4f}{thr:>10.4f}")
    return auc, f1, acc, rec, spec

print("\n" + "="*90)
print(f"{'Method':<16}{'AUC':>8}{'F1':>8}{'Acc':>8}{'Recall':>8}{'Spec':>8}{'Thr':>10}")
print("="*90)
for name, scores in candidates.items():
    full_row(name, scores, labels)
print("="*90)
print(f"{'EDC Paper':<16}{0.9541:>8.4f}{0.9306:>8.4f}{0.9008:>8.4f}{0.9596:>8.4f}{0.8112:>8.4f}")
print(f"{'EA2D Target':<16}{0.9753:>8.4f}{0.9395:>8.4f}{0.9340:>8.4f}{0.9334:>8.4f}{0.9347:>8.4f}")
