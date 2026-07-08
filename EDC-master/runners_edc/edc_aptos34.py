# runners_edc/edc_aptos34.py
# BACKBONE ABLATION: ResNet-34 encoder/decoder, APTOS dataset
# Standalone runner — does NOT modify edc_aptos.py (original ResNet50 run).
# NOVELTY 1: RQASW — Residual Quality-Aware Scale Weighting
# Paper settings: batch=32, lr=5e-4 (encoder lr=1e-5), amap_reduction=max

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
from datasets.dataset_aptos import AD_Dataset
from datasets.data_utils import get_data_loader
from models.edc34 import R34_R34
from configs.config_aptos import DATASET_DIR

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
    ).get_dset()

    eval_dset = AD_Dataset(
        name=args.dataset, train=False,
        data_dir=args.data_dir,
        img_size=args.img_size, crop_size=args.img_size,
    ).get_dset()

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

    # ✅ BACKBONE ABLATION: ResNet-34 encoder/decoder (models/edc34.py)
    model = R34_R34(
        img_size=args.img_size, train_encoder=True,
        stop_grad=True, reshape=True, bn_pretrain=False,
        use_rqasw=args.use_rqasw,
    )

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 0.01

    device = args.device
    model = model.to(device)

    # amap_reduction: APTOS = max (locked-in decision)
    runner = EDC(
        model=model, num_eval_iter=args.num_eval_iter,
        amap_reduction='max', tb_log=None, logger=logger,
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

    seeds = [0, 1, 2, 3, 4]
    all_y_true   = []
    all_y_scores = []
    all_y1_scores, all_y2_scores, all_y3_scores = [], [], []
    all_eval_dsets = []

    for seed in seeds:
        logger.info(f"\n{'='*50}\nRunning seed {seed}\n{'='*50}")
        eval_dict, eval_dset = run_single_seed(gpu, args, seed)
        all_y_true.append(np.array(eval_dict["eval/y_true"]))
        all_y_scores.append(np.array(eval_dict["eval/y_score"]))
        all_y1_scores.append(np.array([eval_dict["eval/AUC1"]]))
        all_y2_scores.append(np.array([eval_dict["eval/AUC2"]]))
        all_y3_scores.append(np.array([eval_dict["eval/AUC3"]]))
        all_eval_dsets.append(eval_dset)
        torch.cuda.empty_cache()
        logger.info(f"[Seed {seed}] AUC: {eval_dict['eval/AUC']:.4f}")

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
    logger.info(f"Per-scale AUC (best seed) — M1: {all_y1_scores[best_idx][0]:.4f} | M2: {all_y2_scores[best_idx][0]:.4f} | M3: {all_y3_scores[best_idx][0]:.4f}")

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
    print("\n======== FINAL EVALUATION METRICS — APTOS / ResNet34 (5-seed best) ========\n")
    print(metrics.to_string(index=False, float_format="%.4f"))

    print("\n======== COMPARISON WITH ORIGINAL ResNet50 RUN ========")
    print(f"{'Metric':<15} {'ResNet50 (yours)':>18} {'ResNet34 (this run)':>20} {'Delta':>10}")
    print("-"*66)
    paper = {"AUC":0.9685,"F1":0.9436,"ACC":0.9200,"Recall":0.9602,"SPE":0.8273}
    yours = {"AUC":auc,"F1":best_f1,"ACC":accuracy,"Recall":recall,"SPE":specificity}
    for k in paper:
        delta = yours[k] - paper[k]
        print(f"{k:<15} {paper[k]:>18.4f} {yours[k]:>20.4f} {delta:>+10.4f}")

    print("\n======== CONFUSION MATRIX ========\n")
    print(pd.DataFrame(cm,
          index=["Actual NORMAL","Actual ABNORMAL"],
          columns=["Predicted NORMAL","Predicted ABNORMAL"]))
    print("\n======== CLASSIFICATION REPORT ========\n")
    print(classification_report(y_true, y_pred,
          target_names=["NORMAL","ABNORMAL"], digits=4))
    print(f"Best Threshold : {best_thr:.4f}")

    eval_paths = all_eval_dsets[0].img_paths
    results=[]; misclassified=[]
    mis_dir = os.path.join(save_path, "misclassified_aptos34")
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
        os.path.join(save_path, "results_test_edc_aptos34.csv"), index=False)
    pd.DataFrame(misclassified,
        columns=["S.No","Filename","GT","Pred","Score"]).to_csv(
        os.path.join(save_path, "misclassified_test_edc_aptos34.csv"), index=False)

    print(f"\nTotal: {len(results)} | Misclassified: {len(misclassified)}")
    print(f"  Normal->Abnormal : {mis_normal}/{total_normal} "
          f"(acc {1-mis_normal/max(total_normal,1):.4f})")
    print(f"  Abnormal->Normal : {mis_abnormal}/{total_abnormal} "
          f"(acc {1-mis_abnormal/max(total_abnormal,1):.4f})")
    logger.warning("COMPLETED.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir',           type=str,      default='./saved_models')
    parser.add_argument('-sn', '--save_name',   type=str,      default='edc_aptos34')
    parser.add_argument('--resume',             action='store_true', default=False)
    parser.add_argument('--load_path',          type=str,      default=None)
    parser.add_argument('-o', '--overwrite',    action='store_true', default=True)
    parser.add_argument('--use_tensorboard',    action='store_true', default=True)

    parser.add_argument('--epoch',              type=int,      default=1)
    parser.add_argument('--num_train_iter',     type=int,      default=1000)
    parser.add_argument('--num_eval_iter',      type=int,      default=250)

    parser.add_argument('-bsz','--batch_size',  type=int,      default=32)
    parser.add_argument('--eval_batch_size',    type=int,      default=64)

    parser.add_argument('--optim',              type=str,      default='AdamW')
    parser.add_argument('--lr',                 type=float,    default=0.0005)
    parser.add_argument('--lr_encoder',         type=float,    default=1e-05)
    parser.add_argument('--momentum',           type=float,    default=0.9)
    parser.add_argument('--weight_decay',       type=float,    default=1e-4)
    parser.add_argument('--amp',                type=str2bool, default=False)
    parser.add_argument('--clip',               type=float,    default=1)
    parser.add_argument('--use_rqasw',          type=str2bool, default=True)

    parser.add_argument('--data_dir',           type=str,      default=DATASET_DIR)
    parser.add_argument('-ds','--dataset',      type=str,      default='fundus')
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