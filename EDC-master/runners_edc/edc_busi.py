import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from helper_modules.utils import get_logger, count_parameters
from helper_modules.train_utils import TBLog, get_optimizer_v2, get_multistep_schedule_with_warmup
from methods.edc1 import EDC

import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from datasets.dataset import AD_Dataset
from datasets.data_utils import get_data_loader
from models.edc import R50_R50

from configs.config_busi import DATASET_DIR
from collections import Counter
import warnings
warnings.filterwarnings("ignore")


def main_worker(gpu, args):

    args.gpu = gpu
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True

    save_path = os.path.join(args.save_dir, args.save_name)
    logger = get_logger(args.save_name, save_path, "INFO")

    # ---------------- DATASET ----------------
    train_dset = AD_Dataset(name=args.dataset, train=True, data_dir=args.data_dir).get_dset()
    eval_dset = AD_Dataset(name=args.dataset, train=False, data_dir=args.data_dir).get_dset()

    print("Train:", len(train_dset))
    print("Test :", len(eval_dset))
    print("Train distribution:", Counter(train_dset.targets))
    print("Test distribution :", Counter(eval_dset.targets))

    loader_dict = {}

    loader_dict["train"] = get_data_loader(
        train_dset,
        args.batch_size,
        data_sampler=args.train_sampler,
        num_iters=args.num_train_iter,
        num_workers=args.num_workers,
    )

    loader_dict["eval"] = get_data_loader(
        eval_dset,
        args.eval_batch_size,
        num_workers=args.num_workers,
        drop_last=False,
    )

    # ---------------- MODEL ----------------
    model = R50_R50(
        img_size=args.img_size,
        train_encoder=True,
        stop_grad=True,
        reshape=True,
        bn_pretrain=False,
    )

    runner = EDC(model=model, num_eval_iter=args.num_eval_iter, logger=logger)

    logger.info(f"Params: {count_parameters(runner.model)}")

    # ---------------- OPTIMIZER ----------------
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

    # ---------------- DEVICE ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runner.model = runner.model.to(device)

    # ---------------- TRAIN ----------------
    runner.set_data_loader(loader_dict)
    runner.tb_log = TBLog(save_path, "tb", use_tensorboard=args.use_tensorboard)

    eval_dict = runner.train(args, device=device, logger=logger)

    print("\nGenerating heatmaps...\n")
    runner.generate_heatmaps(device, args)

    # ---------------- METRICS ----------------
    print("\n===== FINAL RESULTS (BUSI) =====\n")

    metrics = pd.DataFrame({
        "Metric": ["AUC", "F1", "Accuracy", "Recall", "Specificity"],
        "Value": [
            eval_dict["eval/AUC"],
            eval_dict["eval/f1"],
            eval_dict["eval/acc"],
            eval_dict["eval/recall"],
            eval_dict["eval/specificity"],
        ]
    })

    print(metrics.to_string(index=False, float_format="%.4f"))

    # ---------------- CONFUSION MATRIX ----------------
    y_true = np.array(eval_dict["eval/y_true"])
    y_score = np.array(eval_dict["eval/y_score"])
    thr = eval_dict["eval/best_thr"]

    y_pred = (y_score >= thr).astype(int)

    print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))

    print("\nClassification Report:\n")
    print(classification_report(
        y_true, y_pred,
        target_names=["NORMAL", "ABNORMAL"],
        digits=4
    ))


# ---------------- MAIN ----------------
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    # SAVE
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('--save_name', type=str, default='edc_busi')

    # REQUIRED (resume support)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--overwrite', action='store_true', default=True)

    # TRAINING
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--num_train_iter', type=int, default=1000)
    parser.add_argument('--num_eval_iter', type=int, default=250)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=64)

    # OPTIMIZER
    parser.add_argument('--optim', type=str, default='AdamW')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr_encoder', type=float, default=5e-5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # 🔥 ALL REQUIRED FIXES
    parser.add_argument('--train_sampler', type=str, default='RandomSampler')
    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--clip', type=float, default=1.0)

    # DATA
    parser.add_argument('--data_dir', type=str, default=DATASET_DIR)
    parser.add_argument('--dataset', type=str, default='busi')

    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)

    # SYSTEM
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--use_tensorboard', action='store_true', default=True)

    args = parser.parse_args()

    main_worker(args.gpu, args)