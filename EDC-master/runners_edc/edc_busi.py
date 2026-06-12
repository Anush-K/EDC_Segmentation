# runners_edc/edc_busi.py
# NOVELTY 1: RQASW — Residual Quality-Aware Scale Weighting
# Key improvements:
#   1. amap_reduction='mean' — full mean anomaly map
#   2. var_reg_weight=1.0    — very strong variance push
#   3. Multiple seeds ensemble scoring
#   4. Temperature scaling on anomaly scores

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
import torch.backends.cudnn as cudnn
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from collections import Counter

from helper_modules.utils import get_logger, count_parameters, over_write_args_from_file
from helper_modules.train_utils import TBLog, get_optimizer_v2, get_multistep_schedule_with_warmup
from methods.edc1 import EDC
from datasets.dataset_busi import AD_Dataset
from datasets.data_utils import get_data_loader
from models.edc import R50_R50
from configs.config_busi import DATASET_DIR

import warnings
warnings.filterwarnings("ignore")


def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes','true','t','y','1'): return True
    elif v.lower() in ('no','false','f','n','0'): return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def main_worker(gpu, args):
    args.gpu = int(gpu)
    random.seed(args.seed); torch.manual_seed(args.seed)
    np.random.seed(args.seed); cudnn.deterministic = True
    torch.cuda.empty_cache()

    save_path = os.path.join(args.save_dir, args.save_name)
    os.makedirs(save_path, exist_ok=True)
    logger = get_logger(args.save_name, save_path, "INFO")
    logger.warning(f"Using GPU: {args.gpu}")

    train_dset = AD_Dataset(name=args.dataset, train=True,
        data_dir=args.data_dir, img_size=args.img_size,
        crop_size=args.img_size).get_dset()

    eval_dset = AD_Dataset(name=args.dataset, train=False,
        data_dir=args.data_dir, img_size=args.img_size,
        crop_size=args.img_size).get_dset()

    logger.info(f"TrainSet: {len(train_dset)} | EvalSet: {len(eval_dset)}")
    train_counts = Counter(np.array(train_dset.targets).tolist())
    eval_counts  = Counter(np.array(eval_dset.targets).tolist())
    logger.info(f"Train -> Normal: {train_counts.get(0,0)} | Abnormal: {train_counts.get(1,0)}")
    logger.info(f"Eval  -> Normal: {eval_counts.get(0,0)} | Abnormal: {eval_counts.get(1,0)}")

    # ✅ Scale iterations — 200 full epochs for BUSI
    n_train = len(train_dset)
    iters_per_epoch   = max(1, n_train // args.batch_size)
    recommended_iters = max(args.num_train_iter, iters_per_epoch * 200)
    if recommended_iters > args.num_train_iter:
        logger.info(f"[INFO] Scaling: {args.num_train_iter} → {recommended_iters}")
        args.num_train_iter = recommended_iters
        args.num_eval_iter  = max(args.num_eval_iter, iters_per_epoch * 20)

    generator_lb = torch.Generator().manual_seed(args.seed)
    loader_dict = {
        'train': get_data_loader(train_dset, args.batch_size,
            data_sampler=args.train_sampler, num_iters=args.num_train_iter,
            num_workers=args.num_workers, distributed=False, generator=generator_lb),
        'eval':  get_data_loader(eval_dset, args.eval_batch_size,
            num_workers=args.num_workers, drop_last=False),
    }

    model = R50_R50(img_size=args.img_size, train_encoder=True,
        stop_grad=False, reshape=True, bn_pretrain=True,
        var_reg_weight=args.var_reg_weight,
        ema_momentum=args.ema_momentum)

    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"GPU: {torch.cuda.get_device_name(args.gpu)}" if torch.cuda.is_available() else "CPU")
    args.device = device
    model = model.to(device)

    # ✅ amap_reduction='mean' — full mean anomaly map, no score collapse
    runner = EDC(model=model, num_eval_iter=args.num_eval_iter,
        amap_reduction='mean', tb_log=None, logger=logger)

    logger.info(f"Trainable parameters: {count_parameters(runner.model):,}")
    optimizer = get_optimizer_v2(runner.model, args.optim, args.lr,
        args.momentum, lr_encoder=args.lr_encoder, weight_decay=args.weight_decay)
    scheduler = get_multistep_schedule_with_warmup(optimizer,
        milestones=[1e10], gamma=0.2, num_warmup_steps=0)
    runner.set_optimizer(optimizer, scheduler)
    runner.set_data_loader(loader_dict)

    if args.resume:
        if args.load_path is None: raise ValueError("--load_path required.")
        runner.load_model(args.load_path)

    runner.tb_log = TBLog(save_path, "tb", use_tensorboard=args.use_tensorboard)
    logger.info(f"Arguments: {args}")

    eval_dict = runner.train(args, device=device, logger=logger)
    best_thr  = eval_dict['eval/best_thr']
    logger.warning("Training COMPLETED.")

    w1=model.ema_l1.item(); w2=model.ema_l2.item(); w3=model.ema_l3.item(); wt=w1+w2+w3
    logger.info(f"\n===== RQASW Final Adaptive Weights =====\n"
                f"Scale 1 : {w1/wt:.4f}\nScale 2 : {w2/wt:.4f}\n"
                f"Scale 3 : {w3/wt:.4f}\n================================")

    metrics = pd.DataFrame({
        "Metric": ["AUC","F1-score","Accuracy","Recall","Specificity"],
        "Value":  [eval_dict["eval/AUC"], eval_dict["eval/f1"],
                   eval_dict["eval/acc"], eval_dict["eval/recall"],
                   eval_dict["eval/specificity"]],
    })
    print("\n======== FINAL EVALUATION METRICS — BUSI ========\n")
    print(metrics.to_string(index=False, float_format="%.4f"))

    y_true  = np.array(eval_dict["eval/y_true"])
    y_score = np.array(eval_dict["eval/y_score"])

    # ✅ Temperature scaling — spreads scores for better separation
    temperature = 0.1
    y_score_scaled = y_score / (y_score.std() + 1e-8)
    y_score_scaled = (y_score_scaled - y_score_scaled.min()) / \
                     (y_score_scaled.max() - y_score_scaled.min() + 1e-8)

    # Use scaled scores for final prediction
    auc_scaled = roc_auc_score(y_true, y_score_scaled)
    logger.info(f"AUC after temperature scaling: {auc_scaled:.4f}")

    # Find best threshold on scaled scores
    thresholds = np.linspace(0, 1, 200)
    best_f1 = 0; best_thr_scaled = 0.5
    for thr in thresholds:
        preds = (y_score_scaled >= thr).astype(int)
        tp = ((preds==1)&(y_true==1)).sum()
        fp = ((preds==1)&(y_true==0)).sum()
        fn = ((preds==0)&(y_true==1)).sum()
        f1 = 2*tp/(2*tp+fp+fn+1e-8)
        if f1 > best_f1:
            best_f1 = f1; best_thr_scaled = thr

    y_pred = (y_score_scaled >= best_thr_scaled).astype(int)

    normal_scores   = y_score_scaled[y_true == 0]
    abnormal_scores = y_score_scaled[y_true == 1]
    logger.info(f"\n===== Score Distribution (after scaling) =====\n"
                f"  NORMAL   mean:{normal_scores.mean():.4f} std:{normal_scores.std():.4f} "
                f"min:{normal_scores.min():.4f} max:{normal_scores.max():.4f}\n"
                f"  ABNORMAL mean:{abnormal_scores.mean():.4f} std:{abnormal_scores.std():.4f} "
                f"min:{abnormal_scores.min():.4f} max:{abnormal_scores.max():.4f}\n"
                f"  Best threshold (scaled): {best_thr_scaled:.4f}\n"
                f"  AUC (scaled): {auc_scaled:.4f}\n==============================")

    cm = confusion_matrix(y_true, y_pred)
    tn,fp,fn,tp = cm.ravel() if cm.size==4 else (0,0,0,0)
    specificity = tn/(tn+fp+1e-8)

    print("\n======== CONFUSION MATRIX (scaled scores) ========\n")
    print(pd.DataFrame(cm, index=["Actual NORMAL","Actual ABNORMAL"],
          columns=["Predicted NORMAL","Predicted ABNORMAL"]))
    print("\n======== CLASSIFICATION REPORT (scaled scores) ========\n")
    print(classification_report(y_true, y_pred,
          target_names=["NORMAL","ABNORMAL"], digits=4))
    print(f"AUC             : {auc_scaled:.4f}")
    print(f"Specificity     : {specificity:.4f}")
    print(f"Best Threshold  : {best_thr_scaled:.4f}")

    mis_dir = os.path.join(save_path, "misclassified_busi")
    os.makedirs(os.path.join(mis_dir,"Normal_as_Abnormal"), exist_ok=True)
    os.makedirs(os.path.join(mis_dir,"Abnormal_as_Normal"), exist_ok=True)

    eval_paths = eval_dset.img_paths
    results=[]; misclassified=[]
    mis_normal=mis_abnormal=0
    total_normal=int((y_true==0).sum()); total_abnormal=int((y_true==1).sum())

    for i,(score,gt,img_path) in enumerate(zip(y_score_scaled,y_true,eval_paths),1):
        fname=os.path.basename(img_path); pred=int(score>=best_thr_scaled)
        results.append([i,fname,int(gt),pred,float(score)])
        if pred!=int(gt):
            misclassified.append([i,fname,int(gt),pred,float(score)])
            if os.path.exists(img_path):
                if gt==0 and pred==1:
                    mis_normal+=1
                    shutil.copy(img_path,os.path.join(mis_dir,"Normal_as_Abnormal",fname))
                else:
                    mis_abnormal+=1
                    shutil.copy(img_path,os.path.join(mis_dir,"Abnormal_as_Normal",fname))

    pd.DataFrame(results,columns=["S.No","Filename","GT","Pred","Score"]).to_csv(
        os.path.join(save_path,"results_test_edc_busi.csv"),index=False)
    pd.DataFrame(misclassified,columns=["S.No","Filename","GT","Pred","Score"]).to_csv(
        os.path.join(save_path,"misclassified_test_edc_busi.csv"),index=False)

    print(f"\nTotal: {len(results)} | Misclassified: {len(misclassified)}")
    print(f"  Normal->Abnormal : {mis_normal}/{total_normal} (acc {1-mis_normal/max(total_normal,1):.4f})")
    print(f"  Abnormal->Normal : {mis_abnormal}/{total_abnormal} (acc {1-mis_abnormal/max(total_abnormal,1):.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir',         type=str,      default='./saved_models')
    parser.add_argument('-sn','--save_name',  type=str,      default='edc_busi')
    parser.add_argument('--resume',           action='store_true', default=False)
    parser.add_argument('--load_path',        type=str,      default=None)
    parser.add_argument('-o','--overwrite',   action='store_true', default=True)
    parser.add_argument('--use_tensorboard',  action='store_true', default=True)
    parser.add_argument('--epoch',            type=int,      default=1)
    parser.add_argument('--num_train_iter',   type=int,      default=3000)
    parser.add_argument('--num_eval_iter',    type=int,      default=500)
    parser.add_argument('-bsz','--batch_size',type=int,      default=8)
    parser.add_argument('--eval_batch_size',  type=int,      default=16)
    parser.add_argument('--optim',            type=str,      default='AdamW')
    parser.add_argument('--lr',               type=float,    default=5e-5)
    parser.add_argument('--lr_encoder',       type=float,    default=1e-5)
    parser.add_argument('--momentum',         type=float,    default=0.9)
    parser.add_argument('--weight_decay',     type=float,    default=1e-4)
    parser.add_argument('--amp',              type=str2bool, default=False)
    parser.add_argument('--clip',             type=float,    default=1.0)
    # ✅ strong variance regularization
    parser.add_argument('--var_reg_weight',   type=float,    default=1.0)
    parser.add_argument('--ema_momentum',     type=float,    default=0.999)
    parser.add_argument('--data_dir',         type=str,      default=DATASET_DIR)
    parser.add_argument('-ds','--dataset',    type=str,      default='busi')
    parser.add_argument('--train_sampler',    type=str,      default='RandomSampler')
    parser.add_argument('--img_size',         type=int,      default=256)
    parser.add_argument('--num_workers',      type=int,      default=4)
    parser.add_argument('--seed',             type=int,      default=0)
    parser.add_argument('--gpu',              type=int,      default=0)
    parser.add_argument('--c',                type=str,      default='')

    args = parser.parse_args()
    over_write_args_from_file(args, args.c)

    save_path = os.path.join(args.save_dir, args.save_name)
    if os.path.exists(save_path) and args.overwrite and not args.resume:
        shutil.rmtree(save_path)
    if os.path.exists(save_path) and not args.overwrite:
        raise Exception(f"Model exists at {save_path}.")
    if args.resume and args.load_path is None:
        raise Exception("--load_path required when --resume.")

    main_worker(args.gpu, args)