import sys
import os

# --------------------------------------------------
# Add project root to PYTHONPATH
# --------------------------------------------------
ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "datasets"))

import argparse
import random
import shutil

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

from collections import Counter

from helper_modules.utils import (
    get_logger,
    count_parameters,
    over_write_args_from_file
)

from helper_modules.train_utils import (
    TBLog,
    get_optimizer_v2,
    get_multistep_schedule_with_warmup
)

from methods.edc1 import EDC
from datasets.dataset_busi import AD_Dataset
from datasets.data_utils import get_data_loader
from models.edc import R50_R50
from configs.config_busi import DATASET_DIR

import warnings
warnings.filterwarnings("ignore")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def main_worker(gpu, args):

    args.gpu = int(gpu)
    assert args.seed is not None

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True

    # Free GPU memory
    torch.cuda.empty_cache()

    save_path = os.path.join(args.save_dir, args.save_name)
    os.makedirs(save_path, exist_ok=True)

    logger = get_logger(args.save_name, save_path, "INFO")
    logger.warning(f"Using GPU: {args.gpu}")

    # --------------------------------------------------
    # DATASETS
    # --------------------------------------------------
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

    logger.info(f"TrainSet: {len(train_dset)} | EvalSet: {len(eval_dset)}")

    train_counts = Counter(np.array(train_dset.targets).tolist())
    eval_counts  = Counter(np.array(eval_dset.targets).tolist())

    logger.info(f"Train -> Normal: {train_counts.get(0,0)} | "
                f"Abnormal: {train_counts.get(1,0)}")
    logger.info(f"Eval  -> Normal: {eval_counts.get(0,0)} | "
                f"Abnormal: {eval_counts.get(1,0)}")

    # ✅ FIX: BUSI is small (106 train images) — scale iterations
    # to ensure model sees enough data (10 full epochs minimum)
    n_train = len(train_dset)
    iters_per_epoch = max(1, n_train // args.batch_size)
    recommended_iters = max(args.num_train_iter, iters_per_epoch * 10)

    if recommended_iters > args.num_train_iter:
        logger.info(
            f"[INFO] Scaling num_train_iter: "
            f"{args.num_train_iter} → {recommended_iters} "
            f"(10 epochs × {n_train} samples ÷ batch {args.batch_size})"
        )
        args.num_train_iter = recommended_iters
        args.num_eval_iter  = max(args.num_eval_iter, iters_per_epoch * 2)

    generator_lb = torch.Generator().manual_seed(args.seed)

    loader_dict = {
        'train': get_data_loader(
            train_dset, args.batch_size,
            data_sampler=args.train_sampler,
            num_iters=args.num_train_iter,
            num_workers=args.num_workers,
            distributed=False,
            generator=generator_lb,
        ),
        'eval': get_data_loader(
            eval_dset, args.eval_batch_size,
            num_workers=args.num_workers,
            drop_last=False,
        ),
    }

    # --------------------------------------------------
    # MODEL
    # --------------------------------------------------
    model = R50_R50(
        img_size=args.img_size,
        train_encoder=True,
        stop_grad=False,
        reshape=True,
        bn_pretrain=True,
        var_reg_weight=args.var_reg_weight,
        ema_momentum=args.ema_momentum,
    )

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        logger.info(f"GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device("cpu")
        logger.info("CPU mode")

    args.device = device
    model = model.to(device)

    # --------------------------------------------------
    # RUNNER
    # --------------------------------------------------
    runner = EDC(
        model=model,
        num_eval_iter=args.num_eval_iter,
        amap_reduction='mean',
        tb_log=None,
        logger=logger,
    )

    logger.info(f"Trainable parameters: {count_parameters(runner.model):,}")

    optimizer = get_optimizer_v2(
        runner.model, args.optim, args.lr, args.momentum,
        lr_encoder=args.lr_encoder, weight_decay=args.weight_decay,
    )

    scheduler = get_multistep_schedule_with_warmup(
        optimizer, milestones=[1e10], gamma=0.2, num_warmup_steps=0
    )

    runner.set_optimizer(optimizer, scheduler)
    runner.set_data_loader(loader_dict)

    if args.resume:
        if args.load_path is None:
            raise ValueError("--load_path required when --resume.")
        runner.load_model(args.load_path)

    runner.tb_log = TBLog(save_path, "tb", use_tensorboard=args.use_tensorboard)
    logger.info(f"Arguments: {args}")

    # --------------------------------------------------
    # TRAIN
    # --------------------------------------------------
    eval_dict = runner.train(args, device=device, logger=logger)
    best_thr  = eval_dict['eval/best_thr']
    logger.warning("Training COMPLETED.")

    # --------------------------------------------------
    # FINAL WEIGHTS
    # --------------------------------------------------
    w1 = model.ema_l1.item()
    w2 = model.ema_l2.item()
    w3 = model.ema_l3.item()
    wt = w1 + w2 + w3

    logger.info(
        f"\n===== RQASW Final Adaptive Weights =====\n"
        f"Scale 1 : {w1/wt:.4f}\n"
        f"Scale 2 : {w2/wt:.4f}\n"
        f"Scale 3 : {w3/wt:.4f}\n"
        f"================================"
    )

    # --------------------------------------------------
    # METRICS
    # --------------------------------------------------
    metrics = pd.DataFrame({
        "Metric": ["AUC", "F1-score", "Accuracy", "Recall", "Specificity"],
        "Value":  [eval_dict["eval/AUC"], eval_dict["eval/f1"],
                   eval_dict["eval/acc"], eval_dict["eval/recall"],
                   eval_dict["eval/specificity"]],
    })

    print("\n======== FINAL EVALUATION METRICS — BUSI ========\n")
    print(metrics.to_string(index=False, float_format="%.4f"))

    # ✅ Score distribution logger
    y_true  = np.array(eval_dict["eval/y_true"])
    y_score = np.array(eval_dict["eval/y_score"])
    y_pred  = (y_score >= best_thr).astype(int)

    normal_scores   = y_score[y_true == 0]
    abnormal_scores = y_score[y_true == 1]
    logger.info(
        f"\n===== Score Distribution =====\n"
        f"  NORMAL   mean:{normal_scores.mean():.4f} "
        f"std:{normal_scores.std():.4f} "
        f"min:{normal_scores.min():.4f} "
        f"max:{normal_scores.max():.4f}\n"
        f"  ABNORMAL mean:{abnormal_scores.mean():.4f} "
        f"std:{abnormal_scores.std():.4f} "
        f"min:{abnormal_scores.min():.4f} "
        f"max:{abnormal_scores.max():.4f}\n"
        f"  Best threshold: {best_thr:.4f}\n"
        f"=============================="
    )

    cm = confusion_matrix(y_true, y_pred)
    print("\n======== CONFUSION MATRIX ========\n")
    print(pd.DataFrame(cm,
          index=["Actual NORMAL", "Actual ABNORMAL"],
          columns=["Predicted NORMAL", "Predicted ABNORMAL"]))

    print("\n======== CLASSIFICATION REPORT ========\n")
    print(classification_report(y_true, y_pred,
          target_names=["NORMAL", "ABNORMAL"], digits=4))
    print(f"Best Threshold (F1-optimised): {best_thr:.4f}")

    # --------------------------------------------------
    # MISCLASSIFICATION ANALYSIS
    # --------------------------------------------------
    mis_dir         = os.path.join(save_path, "misclassified_busi")
    norm_as_abn_dir = os.path.join(mis_dir, "Normal_as_Abnormal")
    abn_as_norm_dir = os.path.join(mis_dir, "Abnormal_as_Normal")
    os.makedirs(norm_as_abn_dir, exist_ok=True)
    os.makedirs(abn_as_norm_dir, exist_ok=True)

    eval_paths    = eval_dset.img_paths
    results       = []
    misclassified = []
    mis_normal = mis_abnormal = 0
    total_normal   = int((y_true == 0).sum())
    total_abnormal = int((y_true == 1).sum())

    for i, (score, gt, img_path) in enumerate(
            zip(y_score, y_true, eval_paths), 1):
        fname = os.path.basename(img_path)
        pred  = int(score >= best_thr)
        results.append([i, fname, int(gt), pred, float(score)])
        if pred != int(gt):
            misclassified.append([i, fname, int(gt), pred, float(score)])
            if os.path.exists(img_path):
                if gt == 0 and pred == 1:
                    mis_normal += 1
                    shutil.copy(img_path, os.path.join(norm_as_abn_dir, fname))
                else:
                    mis_abnormal += 1
                    shutil.copy(img_path, os.path.join(abn_as_norm_dir, fname))

    pd.DataFrame(results, columns=["S.No","Filename","GT","Pred","Score"]).to_csv(
        os.path.join(save_path, "results_test_edc_busi.csv"), index=False)
    pd.DataFrame(misclassified, columns=["S.No","Filename","GT","Pred","Score"]).to_csv(
        os.path.join(save_path, "misclassified_test_edc_busi.csv"), index=False)

    print(f"\nTotal: {len(results)} | Misclassified: {len(misclassified)}")
    print(f"  Normal->Abnormal : {mis_normal}/{total_normal} "
          f"(acc {1-mis_normal/max(total_normal,1):.4f})")
    print(f"  Abnormal->Normal : {mis_abnormal}/{total_abnormal} "
          f"(acc {1-mis_abnormal/max(total_abnormal,1):.4f})")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir',         type=str,      default='./saved_models')
    parser.add_argument('-sn', '--save_name', type=str,      default='edc_busi')
    parser.add_argument('--resume',           action='store_true', default=False)
    parser.add_argument('--load_path',        type=str,      default=None)
    parser.add_argument('-o', '--overwrite',  action='store_true', default=True)
    parser.add_argument('--use_tensorboard',  action='store_true', default=True)

    parser.add_argument('--epoch',            type=int,      default=1)
    # ✅ base iters — auto-scaled up inside main_worker
    parser.add_argument('--num_train_iter',   type=int,      default=3000)
    parser.add_argument('--num_eval_iter',    type=int,      default=500)

    parser.add_argument('-bsz','--batch_size',type=int,      default=32)
    parser.add_argument('--eval_batch_size',  type=int,      default=64)

    parser.add_argument('--optim',            type=str,      default='AdamW')
    parser.add_argument('--lr',               type=float,    default=5e-4)
    parser.add_argument('--lr_encoder',       type=float,    default=1e-4)
    parser.add_argument('--momentum',         type=float,    default=0.9)
    parser.add_argument('--weight_decay',     type=float,    default=1e-4)
    parser.add_argument('--amp',              type=str2bool, default=False)
    parser.add_argument('--clip',             type=float,    default=1.0)
    parser.add_argument('--var_reg_weight',   type=float,    default=0.1)
    parser.add_argument('--ema_momentum',     type=float,    default=0.99)

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