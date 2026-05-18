# runners_edc/edc_lgg.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import random
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter

from helper_modules.utils import get_logger, count_parameters, over_write_args_from_file
from helper_modules.train_utils import TBLog, get_optimizer_v2, get_multistep_schedule_with_warmup
from methods.edc1 import EDC
from datasets.dataset import AD_Dataset
from datasets.data_utils import get_data_loader
from models.edc import R50_R50
from configs.config_lgg import DATASET_DIR

import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


# ---------------------------------------------------------------------------
# Main worker
# ---------------------------------------------------------------------------
def main_worker(gpu, args):
    # FIX: ensure gpu is always an int
    args.gpu = int(gpu)

    assert args.seed is not None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True

    # Paths & logger
    save_path = os.path.join(args.save_dir, args.save_name)
    os.makedirs(save_path, exist_ok=True)
    logger = get_logger(args.save_name, save_path, "INFO")
    logger.warning(f"Using GPU: {args.gpu}")

    # -----------------------------------------------------------------------
    # Datasets
    # -----------------------------------------------------------------------
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

    print(f"TrainSet: {len(train_dset)} images")
    print(f"EvalSet : {len(eval_dset)} images")

    train_labels = np.array(train_dset.targets)
    eval_labels  = np.array(eval_dset.targets)
    train_counts = Counter(train_labels.tolist())
    eval_counts  = Counter(eval_labels.tolist())

    print("=== Train Split ===")
    print(f"  Normal  : {train_counts.get(0, 0)}")
    print(f"  Abnormal: {train_counts.get(1, 0)}")
    print("=== Eval/Test Split ===")
    print(f"  Normal  : {eval_counts.get(0, 0)}")
    print(f"  Abnormal: {eval_counts.get(1, 0)}")

    # -----------------------------------------------------------------------
    # Data loaders
    # -----------------------------------------------------------------------
    generator_lb = torch.Generator()
    generator_lb.manual_seed(args.seed)

    loader_dict = {
        'train': get_data_loader(
            train_dset,
            args.batch_size,
            data_sampler=args.train_sampler,
            num_iters=args.num_train_iter,
            num_workers=args.num_workers,
            distributed=False,
            generator=generator_lb,
        ),
        'eval': get_data_loader(
            eval_dset,
            args.eval_batch_size,
            num_workers=args.num_workers,
            drop_last=False,
        ),
    }

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    model = R50_R50(
        img_size=args.img_size,
        train_encoder=True,
        stop_grad=True,
        reshape=True,
        bn_pretrain=False,
    )

    # Stable BN momentum for small medical datasets
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 0.01

    # -----------------------------------------------------------------------
    # Device
    # -----------------------------------------------------------------------
    if torch.cuda.is_available():
        # FIX: args.gpu is now guaranteed int
        device = torch.device(f"cuda:{args.gpu}")
        logger.info(f"Using NVIDIA GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU (no GPU detected)")

    args.device = device
    model = model.to(device)

    # -----------------------------------------------------------------------
    # Runner
    # -----------------------------------------------------------------------
    runner = EDC(
        model=model,
        num_eval_iter=args.num_eval_iter,
        amap_reduction=0.1,       # top-10% mean — more robust than max
        tb_log=None,
        logger=logger,
    )
    logger.info(f"Trainable parameters: {count_parameters(runner.model):,}")

    # -----------------------------------------------------------------------
    # Optimiser + scheduler
    # -----------------------------------------------------------------------
    optimizer = get_optimizer_v2(
        runner.model,
        args.optim,
        args.lr,
        args.momentum,
        lr_encoder=args.lr_encoder,
        weight_decay=args.weight_decay,
    )
    scheduler = get_multistep_schedule_with_warmup(
        optimizer, milestones=[1e10], gamma=0.2, num_warmup_steps=0
    )
    runner.set_optimizer(optimizer, scheduler)
    runner.set_data_loader(loader_dict)

    # Optional resume
    if args.resume:
        if args.load_path is None:
            raise ValueError("--load_path required when --resume is set.")
        runner.load_model(args.load_path)

    # TensorBoard
    runner.tb_log = TBLog(save_path, "tb", use_tensorboard=args.use_tensorboard)

    logger.info(f"Arguments: {args}")

    # -----------------------------------------------------------------------
    # Train  (FIX: do NOT call generate_heatmaps again — train() does it)
    # -----------------------------------------------------------------------
    eval_dict = runner.train(args, device=device, logger=logger)
    best_thr  = eval_dict['eval/best_thr']
    logger.warning("Training and Evaluation COMPLETED.")

    # -----------------------------------------------------------------------
    # Final metrics summary
    # -----------------------------------------------------------------------
    metrics_table = pd.DataFrame({
        "Metric": ["AUC", "F1-score", "Accuracy", "Recall (Sensitivity)", "Specificity"],
        "Value":  [
            eval_dict["eval/AUC"],
            eval_dict["eval/f1"],
            eval_dict["eval/acc"],
            eval_dict["eval/recall"],
            eval_dict["eval/specificity"],
        ],
    })
    print("\n================ FINAL EVALUATION METRICS — LGG ================\n")
    print(metrics_table.to_string(index=False, float_format="%.4f"))

    # -----------------------------------------------------------------------
    # Confusion matrix + classification report
    # -----------------------------------------------------------------------
    y_true  = np.array(eval_dict["eval/y_true"])
    y_score = np.array(eval_dict["eval/y_score"])
    y_pred  = (y_score >= best_thr).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual NORMAL", "Actual ABNORMAL"],
        columns=["Predicted NORMAL", "Predicted ABNORMAL"],
    )
    print("\n================ CONFUSION MATRIX ================\n")
    print(cm_df)

    print("\n================ CLASSIFICATION REPORT ================\n")
    print(classification_report(y_true, y_pred,
                                target_names=["NORMAL", "ABNORMAL"], digits=4))
    print(f"Best Threshold (F1-optimised): {best_thr:.4f}")

    # -----------------------------------------------------------------------
    # Misclassification analysis — restored from dead code block
    # -----------------------------------------------------------------------
    mis_dir            = os.path.join(save_path, "misclassified_lgg")
    norm_as_abn_dir    = os.path.join(mis_dir, "Normal_as_Abnormal")
    abn_as_norm_dir    = os.path.join(mis_dir, "Abnormal_as_Normal")
    os.makedirs(norm_as_abn_dir, exist_ok=True)
    os.makedirs(abn_as_norm_dir, exist_ok=True)

    # Rebuild path list from eval dataset
    eval_paths = eval_dset.img_paths

    results       = []
    misclassified = []
    mis_normal    = 0
    mis_abnormal  = 0
    total_normal   = int((y_true == 0).sum())
    total_abnormal = int((y_true == 1).sum())

    for i, (score, gt, img_path) in enumerate(zip(y_score, y_true, eval_paths), start=1):
        fname = os.path.basename(img_path)
        pred  = int(score >= best_thr)
        results.append([i, fname, int(gt), pred, float(score)])

        if pred != int(gt):
            misclassified.append([i, fname, int(gt), pred, float(score)])
            if not os.path.exists(img_path):
                print(f"  Warning: missing file {img_path}")
                continue
            if gt == 0 and pred == 1:
                mis_normal += 1
                shutil.copy(img_path, os.path.join(norm_as_abn_dir, fname))
            else:
                mis_abnormal += 1
                shutil.copy(img_path, os.path.join(abn_as_norm_dir, fname))

    # Save CSVs
    pd.DataFrame(results, columns=["S.No", "Filename", "GT", "Pred", "Score"]).to_csv(
        os.path.join(save_path, "results_test_edc_lgg.csv"), index=False
    )
    pd.DataFrame(misclassified, columns=["S.No", "Filename", "GT", "Pred", "Score"]).to_csv(
        os.path.join(save_path, "misclassified_test_edc_lgg.csv"), index=False
    )

    total_mis = len(misclassified)
    print(f"\nTotal test samples : {len(results)}")
    print(f"Total misclassified: {total_mis}")
    print(f"  Normal→Abnormal  : {mis_normal}  / {total_normal}  "
          f"(acc {1 - mis_normal / max(total_normal, 1):.4f})")
    print(f"  Abnormal→Normal  : {mis_abnormal} / {total_abnormal} "
          f"(acc {1 - mis_abnormal / max(total_abnormal, 1):.4f})")
    print(f"\nCSVs and misclassified images saved to: {mis_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EDC training on LGG dataset')

    # Saving / loading
    parser.add_argument('--save_dir',         type=str,      default='./saved_models')
    parser.add_argument('-sn', '--save_name', type=str,      default='edc_lgg')
    parser.add_argument('--resume',           action='store_true', default=False)
    parser.add_argument('--load_path',        type=str,      default=None)
    parser.add_argument('-o', '--overwrite',  action='store_true', default=True)
    parser.add_argument('--use_tensorboard',  action='store_true', default=True)

    # Training
    parser.add_argument('--epoch',            type=int,      default=1)
    parser.add_argument('--num_train_iter',   type=int,      default=20000,   # FIX: was 1000
                        help='Total training iterations (≥10k recommended)')
    parser.add_argument('--num_eval_iter',    type=int,      default=1000,    # FIX: evaluate every 1k iters
                        help='Evaluation frequency in iterations')
    parser.add_argument('-bsz', '--batch_size',      type=int, default=32)
    parser.add_argument('--eval_batch_size',         type=int, default=64)

    # Optimiser
    parser.add_argument('--optim',            type=str,      default='AdamW')
    parser.add_argument('--lr',               type=float,    default=5e-4)
    parser.add_argument('--lr_encoder',       type=float,    default=5e-5)
    parser.add_argument('--momentum',         type=float,    default=0.9)
    parser.add_argument('--weight_decay',     type=float,    default=1e-4)
    parser.add_argument('--amp',              type=str2bool, default=False)
    parser.add_argument('--clip',             type=float,    default=1.0)

    # Data
    parser.add_argument('--data_dir',         type=str,      default=DATASET_DIR)
    parser.add_argument('-ds', '--dataset',   type=str,      default='lgg_mri')
    parser.add_argument('--train_sampler',    type=str,      default='RandomSampler')
    parser.add_argument('--img_size',         type=int,      default=256)
    parser.add_argument('--num_workers',      type=int,      default=4)

    # GPU / seed
    parser.add_argument('--seed',             type=int,      default=0)
    parser.add_argument('--gpu',              type=int,      default=0,        # FIX: int not str
                        help='GPU index to use (0-indexed)')

    # Config file override
    parser.add_argument('--c',                type=str,      default='')

    args = parser.parse_args()
    over_write_args_from_file(args, args.c)

    # Handle save directory
    save_path = os.path.join(args.save_dir, args.save_name)
    if os.path.exists(save_path) and args.overwrite and not args.resume:
        shutil.rmtree(save_path)
    if os.path.exists(save_path) and not args.overwrite:
        raise Exception(f"Model already exists at {save_path}. Use --overwrite to replace.")
    if args.resume and args.load_path is None:
        raise Exception("--load_path required when --resume is set.")

    main_worker(args.gpu, args)