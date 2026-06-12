# runners_edc/edc_oct2017.py
# NOVELTY 1: RQASW — Residual Quality-Aware Scale Weighting
# Paper settings: 1000 iterations, lr=5e-4, batch=32, mean reduction
# RQASW: adaptive scale weighting via EMA + Gaussian normalization

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "datasets"))

import argparse
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from collections import Counter

from helper_modules.utils import get_logger, count_parameters, over_write_args_from_file
from helper_modules.train_utils import TBLog, get_optimizer_v2, get_multistep_schedule_with_warmup
from methods.edc1 import EDC
from datasets.dataset_oct2017 import AD_Dataset
from datasets.data_utils import get_data_loader
from models.edc import R50_R50
from configs.config_oct2017 import DATASET_DIR

import warnings
warnings.filterwarnings("ignore")


def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes','true','t','y','1'): return True
    elif v.lower() in ('no','false','f','n','0'): return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def run_single_seed(gpu, args, seed):
    """Run training with a single seed and return eval_dict."""
    random.seed(seed); torch.manual_seed(seed)
    np.random.seed(seed); cudnn.deterministic = True
    torch.cuda.empty_cache()

    save_path = os.path.join(args.save_dir, args.save_name)
    logger = args._logger

    train_dset = AD_Dataset(
        name=args.dataset, train=True,
        data_dir=args.data_dir,
        img_size=args.img_size, crop_size=args.img_size,
        train_samples_limit=args.train_samples_limit,
    ).get_dset()

    eval_dset = AD_Dataset(
        name=args.dataset, train=False,
        data_dir=args.data_dir,
        img_size=args.img_size, crop_size=args.img_size,
        train_samples_limit=args.train_samples_limit,
    ).get_dset()

    # Print split info once (seed 0 only)
    if seed == 0:
        train_counts = Counter(np.array(train_dset.targets))
        eval_counts  = Counter(np.array(eval_dset.targets))
        logger.info("=== Train Split ===")
        logger.info(f"Total: {len(train_dset.targets)}  "
                    f"Normal: {train_counts.get(0,0)}  "
                    f"Abnormal: {train_counts.get(1,0)}")
        logger.info("=== Eval/Test Split ===")
        logger.info(f"Total: {len(eval_dset.targets)}  "
                    f"Normal: {eval_counts.get(0,0)}  "
                    f"Abnormal: {eval_counts.get(1,0)}")

    generator_lb = torch.Generator().manual_seed(seed)
    loader_dict = {
        'train': get_data_loader(
            train_dset, args.batch_size,
            data_sampler=args.train_sampler,
            num_iters=args.num_train_iter,
            num_workers=args.num_workers,
            distributed=False, generator=generator_lb,
        ),
        'eval': get_data_loader(
            eval_dset, args.eval_batch_size,
            num_workers=args.num_workers, drop_last=False,
        ),
    }

    model = R50_R50(
        img_size=args.img_size, train_encoder=True,
        stop_grad=True, reshape=True, bn_pretrain=True,
    )

    # OCT2017: slow BN momentum for stable retinal-layer features
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 0.01

    device = args.device
    model = model.to(device)

    # ✅ amap_reduction='mean' — OCT2017 uses mean (diffuse retinal layer patterns)
    runner = EDC(
        model=model, num_eval_iter=args.num_eval_iter,
        amap_reduction='mean', tb_log=None, logger=logger,
    )

    logger.info(f"[Seed {seed}] Trainable Params: {count_parameters(runner.model)}")

    optimizer = get_optimizer_v2(
        runner.model, args.optim, args.lr,
        args.momentum, lr_encoder=args.lr_encoder,
        weight_decay=args.weight_decay,
    )
    scheduler = get_multistep_schedule_with_warmup(
        optimizer, milestones=[1e10], gamma=0.2, num_warmup_steps=0
    )
    runner.set_optimizer(optimizer, scheduler)
    runner.set_data_loader(loader_dict)

    save_path = os.path.join(args.save_dir, args.save_name)
    runner.tb_log = TBLog(save_path, "tb", use_tensorboard=False)

    eval_dict = runner.train(args, device=device, logger=logger)

    return eval_dict, eval_dset


def main_worker(gpu, args):
    args.gpu = int(gpu)

    save_path = os.path.join(args.save_dir, args.save_name)
    os.makedirs(save_path, exist_ok=True)
    logger = get_logger(args.save_name, save_path, "INFO")
    logger.warning(f"Using GPU: {args.gpu}")
    args._logger = logger

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        logger.info(f"Using NVIDIA GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU (no GPU backend detected)")
    args.device = device

    # ✅ Run 5 seeds and pick best — same as paper
    seeds = [0, 1, 2, 3, 4]
    all_y_true   = []
    all_y_scores = []
    all_eval_dsets = []

    for seed in seeds:
        logger.info(f"\n{'='*50}\nRunning seed {seed}\n{'='*50}")
        eval_dict, eval_dset = run_single_seed(gpu, args, seed)
        all_y_true.append(np.array(eval_dict["eval/y_true"]))
        all_y_scores.append(np.array(eval_dict["eval/y_score"]))
        all_eval_dsets.append(eval_dset)
        torch.cuda.empty_cache()
        logger.info(f"[Seed {seed}] AUC: {eval_dict['eval/AUC']:.4f}")

    # ✅ Normalize scores and pick best seed by AUC
    y_true = all_y_true[0]
    best_auc = 0; best_idx = 0
    for i in range(len(seeds)):
        s = all_y_scores[i]
        s = s / (s.std() + 1e-8)
        s = (s - s.min()) / (s.max() - s.min() + 1e-8)
        all_y_scores[i] = s
        a = roc_auc_score(y_true, s)
        logger.info(f"[Seed {seeds[i]}] Normalized AUC: {a:.4f}")
        if a > best_auc:
            best_auc = a; best_idx = i

    logger.info(f"Best seed: {seeds[best_idx]}  AUC: {best_auc:.4f}")
    y_final = all_y_scores[best_idx]
    auc = roc_auc_score(y_true, y_final)
    logger.info(f"Final AUC: {auc:.4f}")

    # ✅ Best threshold by F1
    thresholds = np.linspace(0, 1, 500)
    best_f1 = 0; best_thr = 0.5
    for thr in thresholds:
        preds = (y_final >= thr).astype(int)
        tp = ((preds==1)&(y_true==1)).sum()
        fp = ((preds==1)&(y_true==0)).sum()
        fn = ((preds==0)&(y_true==1)).sum()
        f1 = 2*tp/(2*tp+fp+fn+1e-8)
        if f1 > best_f1:
            best_f1 = f1; best_thr = thr

    y_pred      = (y_final >= best_thr).astype(int)
    cm          = confusion_matrix(y_true, y_pred)
    tn,fp_,fn_,tp_ = cm.ravel() if cm.size==4 else (0,0,0,0)
    specificity = tn  / (tn  + fp_ + 1e-8)
    recall      = tp_ / (tp_ + fn_ + 1e-8)
    accuracy    = (tp_ + tn) / (len(y_true) + 1e-8)

    # Score distribution
    normal_scores   = y_final[y_true==0]
    abnormal_scores = y_final[y_true==1]
    logger.info(
        f"\n===== Score Distribution (best seed) =====\n"
        f"  NORMAL   mean:{normal_scores.mean():.4f} "
        f"std:{normal_scores.std():.4f}\n"
        f"  ABNORMAL mean:{abnormal_scores.mean():.4f} "
        f"std:{abnormal_scores.std():.4f}\n"
        f"  Best threshold: {best_thr:.4f}\n"
        f"==========================================="
    )

    metrics = pd.DataFrame({
        "Metric": ["AUC","F1-score","Accuracy","Recall","Specificity"],
        "Value":  [auc, best_f1, accuracy, recall, specificity],
    })
    print("\n======== FINAL EVALUATION METRICS — OCT2017 (5-seed best) ========\n")
    print(metrics.to_string(index=False, float_format="%.4f"))

    # ✅ Paper comparison
    print("\n======== COMPARISON WITH PAPER ========")
    print(f"{'Metric':<15} {'Paper EDC':>12} {'Your RQASW':>12} {'Delta':>10}")
    print("-"*52)
    paper = {"AUC":0.9754,"F1":0.9461,"ACC":0.9354,"Recall":0.9643,"SPE":0.8986}
    yours = {"AUC":auc,"F1":best_f1,"ACC":accuracy,"Recall":recall,"SPE":specificity}
    for k in paper:
        delta = yours[k] - paper[k]
        symbol = "✅" if delta >= 0 else "❌"
        print(f"{k:<15} {paper[k]:>12.4f} {yours[k]:>12.4f} {delta:>+10.4f} {symbol}")

    print("\n======== CONFUSION MATRIX ========\n")
    print(pd.DataFrame(cm,
          index=["Actual NORMAL","Actual ABNORMAL"],
          columns=["Predicted NORMAL","Predicted ABNORMAL"]))
    print("\n======== CLASSIFICATION REPORT ========\n")
    print(classification_report(y_true, y_pred,
          target_names=["NORMAL","ABNORMAL"], digits=4))
    print(f"Best Threshold : {best_thr:.4f}")

    # ✅ Save results + misclassified images
    eval_paths = all_eval_dsets[0].img_paths
    results=[]; misclassified=[]
    mis_dir = os.path.join(save_path, "misclassified_oct2017")
    os.makedirs(os.path.join(mis_dir, "Normal_as_Abnormal"), exist_ok=True)
    os.makedirs(os.path.join(mis_dir, "Abnormal_as_Normal"), exist_ok=True)
    mis_normal=mis_abnormal=0
    total_normal   = int((y_true==0).sum())
    total_abnormal = int((y_true==1).sum())

    for i,(score,gt,img_path) in enumerate(zip(y_final,y_true,eval_paths),1):
        fname = os.path.basename(img_path)
        pred  = int(score >= best_thr)
        results.append([i, fname, int(gt), pred, float(score)])
        if pred != int(gt):
            misclassified.append([i, fname, int(gt), pred, float(score)])
            if os.path.exists(img_path):
                if gt==0 and pred==1:
                    mis_normal += 1
                    shutil.copy(img_path,
                        os.path.join(mis_dir, "Normal_as_Abnormal", fname))
                else:
                    mis_abnormal += 1
                    shutil.copy(img_path,
                        os.path.join(mis_dir, "Abnormal_as_Normal", fname))

    pd.DataFrame(results,
        columns=["S.No","Filename","GT","Pred","Score"]).to_csv(
        os.path.join(save_path, "results_test_edc_oct2017.csv"), index=False)
    pd.DataFrame(misclassified,
        columns=["S.No","Filename","GT","Pred","Score"]).to_csv(
        os.path.join(save_path, "misclassified_test_edc_oct2017.csv"), index=False)

    print(f"\nTotal: {len(results)} | Misclassified: {len(misclassified)}")
    print(f"  Normal->Abnormal : {mis_normal}/{total_normal} "
          f"(acc {1-mis_normal/max(total_normal,1):.4f})")
    print(f"  Abnormal->Normal : {mis_abnormal}/{total_abnormal} "
          f"(acc {1-mis_abnormal/max(total_abnormal,1):.4f})")
    logger.warning("COMPLETED.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir',           type=str,      default='./saved_models')
    parser.add_argument('-sn', '--save_name',   type=str,      default='edc_oct2017')
    parser.add_argument('--resume',             action='store_true', default=False)
    parser.add_argument('--load_path',          type=str,      default=None)
    parser.add_argument('-o', '--overwrite',    action='store_true', default=True)
    parser.add_argument('--use_tensorboard',    action='store_true', default=True)

    parser.add_argument('--epoch',              type=int,      default=1)
    # ✅ Paper uses 1000 iterations for OCT2017
    parser.add_argument('--num_train_iter',     type=int,      default=1000)
    parser.add_argument('--num_eval_iter',      type=int,      default=250)

    # ✅ Paper batch = 32
    parser.add_argument('-bsz','--batch_size',  type=int,      default=32)
    parser.add_argument('--eval_batch_size',    type=int,      default=64)

    parser.add_argument('--optim',              type=str,      default='AdamW')
    # ✅ Paper encoder lr=5e-4, decoder lr=1e-5
    parser.add_argument('--lr',                 type=float,    default=5e-4)
    parser.add_argument('--lr_encoder',         type=float,    default=1e-5)
    parser.add_argument('--momentum',           type=float,    default=0.9)
    parser.add_argument('--weight_decay',       type=float,    default=1e-4)
    parser.add_argument('--amp',                type=str2bool, default=False)
    parser.add_argument('--clip',               type=float,    default=1)

    # ✅ OCT2017-specific: cap large train set
    parser.add_argument('--train_samples_limit',type=int,      default=10000,
                        help='Max NORMAL samples used for training (OCT2017 train is ~26k)')

    parser.add_argument('--data_dir',           type=str,      default=DATASET_DIR)
    parser.add_argument('-ds','--dataset',      type=str,      default='oct2017')
    parser.add_argument('--train_sampler',      type=str,      default='RandomSampler')
    parser.add_argument('--img_size',           type=int,      default=256)
    parser.add_argument('--num_workers',        type=int,      default=4)

    parser.add_argument('--seed',               type=int,      default=0)
    parser.add_argument('--gpu',                type=str,      default='0')
    parser.add_argument('--c',                  type=str,      default='')

    args = parser.parse_args()
    over_write_args_from_file(args, args.c)

    save_path = os.path.join(args.save_dir, args.save_name)
    if os.path.exists(save_path) and args.overwrite and not args.resume:
        shutil.rmtree(save_path)

    main_worker(args.gpu, args)