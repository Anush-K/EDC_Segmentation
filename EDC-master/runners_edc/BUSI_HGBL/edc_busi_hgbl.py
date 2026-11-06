# runners_edc/edc_busi_hgbl.py
# Stage 1 (RQASW) for BUSI in the HGBL paper - leakage-free version of edc_busi.py.
#
# Put this file in runners_edc/ or runners_edc/BUSI_HGBL/ (it finds EDC-master itself). It uses YOUR modules unchanged
# (methods/edc1.py, models/edc.py, datasets/dataset_busi.py, datasets/data_utils.py).
#
# What is the SAME as edc_busi.py
#   model R50_R50 (bn_pretrain=True, stop_grad=False, use_rqasw=True), AdamW,
#   lr 5e-5 / lr_encoder 1e-5, var_reg_weight 1.0, ema_momentum 0.999, batch 8,
#   3000 iterations, eval every 500 iterations, image score = mean of p_all,
#   5 seeds (0-4), heatmaps saved as <stem>_map.png (grayscale, min-max to 0-255).
#
# What is CHANGED (each change removes a use of TEST labels)
#   1. The evaluation loader used DURING training is the VALIDATION set
#      (BUSI_HGBL_VAL: 21 normal + 65 abnormal). model_best.pth is therefore chosen
#      by validation AUC, not test AUC.
#   2. Always the 5-seed ENSEMBLE (fixed rule). The old code chose "ensemble or best
#      seed" by test AUC.
#   3. Score normalisation of each seed is fitted on its validation scores and then
#      applied unchanged to its test scores.
#   4. The F1 threshold is chosen on the validation ensemble scores and applied to test.
#   5. Heatmaps for Stage 2 = mean of the 5 seeds' p_all maps (each seed: its
#      model_best.pth). The old code copied the heatmaps of the seed with the best
#      test AUC, and those maps came from the LAST iteration, not model_best.
#      Heatmaps are written for ALL test images (27 normal + 647 abnormal).
#
# Usage (from EDC-master/):
#   python runners_edc/BUSI_HGBL/edc_busi_hgbl.py \
#       --data_dir datasets/BUSI_HGBL --val_dir datasets/BUSI_HGBL_VAL \
#       --seg_root ../BUSI_SEG_HGBL --save_name edc_busi_hgbl

import sys
import os

def _find_edc_root():
    """EDC-master = first parent folder that contains methods/, models/ and helper_modules/.
    Works whether this file is in runners_edc/ or in runners_edc/BUSI_HGBL/."""
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        if all(os.path.isdir(os.path.join(d, x)) for x in ("methods", "models", "helper_modules")):
            return d
        d = os.path.dirname(d)
    raise SystemExit("EDC-master not found above " + os.path.abspath(__file__))


ROOT_DIR = _find_edc_root()
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "datasets"))

import argparse
import json
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import pandas as pd
import cv2
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve
from collections import Counter

from helper_modules.utils import get_logger, count_parameters, over_write_args_from_file
from helper_modules.train_utils import TBLog, get_optimizer_v2, get_multistep_schedule_with_warmup
from methods.edc1 import EDC
from datasets.dataset_busi import AD_Dataset
from datasets.data_utils import get_data_loader
from models.edc import R50_R50

import warnings
warnings.filterwarnings("ignore")


def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


@torch.no_grad()
def score_and_maps(model, loader, device, img_size):
    """Image scores (mean of p_all) and p_all maps upsampled to img_size, for one loader."""
    model.eval()
    scores, labels, names, maps = [], [], [], []
    for _, x, _, y, fname in loader:
        p = model(x.to(device))["p_all"]
        scores.extend(p.mean(dim=(1, 2, 3)).cpu().numpy().tolist())
        labels.extend([int(v) for v in y])
        names.extend(list(fname))
        up = F.interpolate(p, size=(img_size, img_size), mode="bilinear", align_corners=False)
        maps.append(up[:, 0].cpu().numpy().astype(np.float32))
    return np.array(scores), np.array(labels), names, np.concatenate(maps, 0)


def min_max(a):
    lo, hi = a.min(), a.max()
    return np.zeros_like(a) if hi == lo else (a - lo) / (hi - lo)


def run_single_seed(args, seed):
    random.seed(seed); torch.manual_seed(seed)
    np.random.seed(seed); cudnn.deterministic = True
    torch.cuda.empty_cache()
    logger = args._logger
    save_path = os.path.join(args.save_dir, args.save_name)
    seed_save_path = os.path.join(save_path, f"seed_{seed}")
    os.makedirs(seed_save_path, exist_ok=True)

    mk = lambda d, train: AD_Dataset(name=args.dataset, train=train, data_dir=d,
                                     img_size=args.img_size, crop_size=args.img_size).get_dset()
    train_dset = mk(args.data_dir, True)
    val_dset = mk(args.val_dir, False)      # CHANGED 1: validation set for model selection
    test_dset = mk(args.data_dir, False)    # used only after training, once

    if seed == args.seeds[0]:
        for nm, d in (("Train", train_dset), ("Val", val_dset), ("Test", test_dset)):
            c = Counter(np.array(d.targets))
            logger.info(f"=== {nm} === total {len(d.targets)}  normal {c.get(0, 0)}  abnormal {c.get(1, 0)}")
        n_train = len(train_dset)
        rec = max(args.num_train_iter, max(1, n_train // args.batch_size) * 50)
        if rec > args.num_train_iter:
            args.num_train_iter = rec
            args.num_eval_iter = max(args.num_eval_iter, rec // 10)
        logger.info(f"num_train_iter = {args.num_train_iter}, num_eval_iter = {args.num_eval_iter}")

    gen = torch.Generator().manual_seed(seed)
    loader_dict = {
        'train': get_data_loader(train_dset, args.batch_size, data_sampler=args.train_sampler,
                                 num_iters=args.num_train_iter, num_workers=args.num_workers,
                                 distributed=False, generator=gen),
        'eval': get_data_loader(val_dset, args.eval_batch_size,
                                num_workers=args.num_workers, drop_last=False),
    }
    test_loader = get_data_loader(test_dset, args.eval_batch_size,
                                  num_workers=args.num_workers, drop_last=False)

    model = R50_R50(img_size=args.img_size, train_encoder=True, stop_grad=args.stop_grad,
                    reshape=True, bn_pretrain=True, var_reg_weight=args.var_reg_weight,
                    ema_momentum=args.ema_momentum, use_rqasw=args.use_rqasw)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 0.01
    device = args.device
    model = model.to(device)

    runner = EDC(model=model, num_eval_iter=args.num_eval_iter, amap_reduction='mean',
                 tb_log=None, logger=logger)
    logger.info(f"[Seed {seed}] Trainable Params: {count_parameters(runner.model)}")
    optimizer = get_optimizer_v2(runner.model, args.optim, args.lr, args.momentum,
                                 lr_encoder=args.lr_encoder, weight_decay=args.weight_decay)
    scheduler = get_multistep_schedule_with_warmup(optimizer, milestones=[1e10], gamma=0.2,
                                                   num_warmup_steps=0)
    runner.set_optimizer(optimizer, scheduler)
    runner.set_data_loader(loader_dict)
    runner.tb_log = TBLog(seed_save_path, "tb", use_tensorboard=False)
    args.seed_save_path = seed_save_path
    runner.train(args, device=device, logger=logger)

    # edc1.py writes validation visuals to <save>/heatmap during training: not used, removed
    top_hm = os.path.join(save_path, 'heatmap')
    if os.path.exists(top_hm):
        shutil.rmtree(top_hm)

    best_ckpt = os.path.join(seed_save_path, "model_best.pth")
    if os.path.exists(best_ckpt):
        ck = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ck['model'] if 'model' in ck else ck)
        logger.info(f"[Seed {seed}] loaded model_best.pth (chosen on VALIDATION AUC)")
    else:
        logger.warning(f"[Seed {seed}] no model_best.pth, using the last iteration")
    w = torch.stack([model.ema_l1, model.ema_l2, model.ema_l3]).cpu().numpy()
    w = (w / w.sum()).tolist()
    logger.info(f"[Seed {seed}] RQASW weights: {w}")

    vs, vy, vn, _ = score_and_maps(model, loader_dict['eval'], device, args.img_size)
    ts, ty, tn, tmaps = score_and_maps(model, test_loader, device, args.img_size)
    out = dict(seed=seed, val_auc=float(roc_auc_score(vy, vs)), test_auc=float(roc_auc_score(ty, ts)),
               rqasw_w=w)
    logger.info(f"[Seed {seed}] val AUC {out['val_auc']:.4f} | test AUC {out['test_auc']:.4f}")
    # per-seed heatmaps: each map min-max normalised per image, stored compactly
    np.save(os.path.join(seed_save_path, "test_maps.npy"),
            np.stack([min_max(m) for m in tmaps]).astype(np.float16))
    return out, (vs, vy, vn), (ts, ty, tn)


def main(args):
    save_path = os.path.join(args.save_dir, args.save_name)
    os.makedirs(save_path, exist_ok=True)
    logger = get_logger(args.save_name, save_path, "INFO")
    args._logger = logger
    args.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"device {args.device}")

    per_seed, V, T = [], [], []
    for seed in args.seeds:
        logger.info(f"\n{'=' * 50}\nRunning seed {seed}\n{'=' * 50}")
        o, v, t = run_single_seed(args, seed)
        per_seed.append(o); V.append(v); T.append(t)
        torch.cuda.empty_cache()

    vy, vnames = V[0][1], V[0][2]; ty, tnames = T[0][1], T[0][2]
    for v, t in zip(V, T):                       # same image order in every seed
        assert list(v[2]) == list(vnames) and list(t[2]) == list(tnames)

    # CHANGED 2+3: fixed 5-seed ensemble; normalisation fitted on validation scores
    def norm_fit(s):
        mu, sd = s.mean(), s.std() + 1e-8
        z = (s - mu) / sd
        return mu, sd, z.min(), z.max()

    def norm_apply(s, p):
        mu, sd, lo, hi = p
        return ((s - mu) / sd - lo) / (hi - lo + 1e-8)

    vnorm, tnorm = [], []
    for v, t in zip(V, T):
        p = norm_fit(v[0]); vnorm.append(norm_apply(v[0], p)); tnorm.append(norm_apply(t[0], p))
    v_ens, t_ens = np.mean(vnorm, 0), np.mean(tnorm, 0)

    # CHANGED 4: threshold from validation only
    P, R, thr = precision_recall_curve(vy, v_ens)
    f1 = 2 * P * R / (P + R + 1e-8)
    tau = float(thr[int(np.argmax(f1[:-1]))])
    yp = (t_ens >= tau).astype(int)
    tn_, fp_, fn_, tp_ = confusion_matrix(ty, yp, labels=[0, 1]).ravel()
    rec = tp_ / max(tp_ + fn_, 1); spec = tn_ / max(tn_ + fp_, 1); prec = tp_ / max(tp_ + fp_, 1)
    res = {
        "AUC": float(roc_auc_score(ty, t_ens)),
        "F1": float(2 * prec * rec / max(prec + rec, 1e-8)),
        "Accuracy": float((tp_ + tn_) / len(ty)),
        "Recall": float(rec), "Specificity": float(spec), "Precision": float(prec),
        "TP": int(tp_), "TN": int(tn_), "FP": int(fp_), "FN": int(fn_),
        "threshold_from_val": tau,
        "val_ensemble_AUC": float(roc_auc_score(vy, v_ens)),
        "test_AUC_per_seed": [o["test_auc"] for o in per_seed],
        "test_AUC_seed_mean": float(np.mean([o["test_auc"] for o in per_seed])),
        "test_AUC_seed_sd": float(np.std([o["test_auc"] for o in per_seed], ddof=1)) if len(per_seed) > 1 else 0.0,
        "rqasw_weights_per_seed": [o["rqasw_w"] for o in per_seed],
        "n_test": int(len(ty)), "n_test_normal": int((ty == 0).sum()), "n_test_abnormal": int((ty == 1).sum()),
        "n_val": int(len(vy)),
    }
    with open(os.path.join(save_path, "stage1_results_busi.json"), "w") as f:
        json.dump(res, f, indent=2)
    pd.DataFrame({"Filename": tnames, "GT": ty, "Score": t_ens, "Pred": yp}).to_csv(
        os.path.join(save_path, "results_test_edc_busi_hgbl.csv"), index=False)
    pd.DataFrame({"Filename": vnames, "GT": vy, "Score": v_ens}).to_csv(
        os.path.join(save_path, "results_val_edc_busi_hgbl.csv"), index=False)

    # CHANGED 5: Stage 2 heatmaps = mean of the 5 seeds (model_best of each)
    maps = np.mean([np.load(os.path.join(save_path, f"seed_{s}", "test_maps.npy")).astype(np.float32)
                    for s in args.seeds], axis=0)
    hm_dir = os.path.join(save_path, "heatmap"); os.makedirs(hm_dir, exist_ok=True)
    seg_hm = os.path.join(args.seg_root, "heatmaps") if args.seg_root else None
    if seg_hm:
        os.makedirs(seg_hm, exist_ok=True)
    n = 0
    for name, lab, m in zip(tnames, ty, maps):
        stem = os.path.splitext(name)[0]
        g = (min_max(m) * 255).astype(np.uint8)       # same format as edc1.py: <stem>_map.png
        cv2.imwrite(os.path.join(hm_dir, stem + "_map.png"), g)
        if seg_hm and lab == 1:
            cv2.imwrite(os.path.join(seg_hm, stem + ".png"), g); n += 1

    print("\n======== STAGE 1 - BUSI (5-seed ensemble, threshold from validation) ========")
    for k in ("AUC", "F1", "Accuracy", "Recall", "Specificity", "Precision"):
        print(f"  {k:<12} {100 * res[k]:6.2f}")
    print(f"  TP {res['TP']}  TN {res['TN']}  FP {res['FP']}  FN {res['FN']}   (test: "
          f"{res['n_test_normal']} normal + {res['n_test_abnormal']} abnormal)")
    print(f"  per-seed test AUC: {' '.join('%.4f' % a for a in res['test_AUC_per_seed'])}"
          f"  (mean {res['test_AUC_seed_mean']:.4f} +/- {res['test_AUC_seed_sd']:.4f})")
    print(f"  threshold (validation): {tau:.4f}")
    if seg_hm:
        print(f"  {n} abnormal heatmaps written to {seg_hm}")
    logger.warning("COMPLETED.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('-sn', '--save_name', type=str, default='edc_busi_hgbl')
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('-o', '--overwrite', action='store_true', default=True)
    parser.add_argument('--use_tensorboard', action='store_true', default=True)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--num_train_iter', type=int, default=3000)
    parser.add_argument('--num_eval_iter', type=int, default=500)
    parser.add_argument('-bsz', '--batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--optim', type=str, default='AdamW')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--lr_encoder', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--amp', type=str2bool, default=False)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--var_reg_weight', type=float, default=1.0)
    parser.add_argument('--ema_momentum', type=float, default=0.999)
    parser.add_argument('--use_rqasw', type=str2bool, default=True)
    parser.add_argument('--stop_grad', type=str2bool, default=False)
    parser.add_argument('--data_dir', type=str, required=True, help="BUSI_HGBL: train/NORMAL + test/{NORMAL,ABNORMAL}")
    parser.add_argument('--val_dir', type=str, required=True, help="BUSI_HGBL_VAL: test/{NORMAL,ABNORMAL} = validation")
    parser.add_argument('--seg_root', type=str, default=None, help="writes <seg_root>/heatmaps/<stem>.png for Stage 2")
    parser.add_argument('-ds', '--dataset', type=str, default='busi')
    parser.add_argument('--train_sampler', type=str, default='RandomSampler')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--c', type=str, default='')
    args = parser.parse_args()
    over_write_args_from_file(args, args.c)
    if os.environ.get("RQASW_MODE"):
        raise SystemExit("RQASW_MODE is set (B1 control experiment). Unset it for the paper runs.")
    sp = os.path.join(args.save_dir, args.save_name)
    if os.path.exists(sp) and args.overwrite and not args.resume:
        shutil.rmtree(sp)
    main(args)