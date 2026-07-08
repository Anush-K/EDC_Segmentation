# runners_edc/edc_br35h_sampeff.py
#
# SECTION F: Sample-Efficiency Analysis (Br35H)
# Standalone file — does NOT modify edc_br35h.py or methods/edc1.py.
# Reuses the confirmed ResNet50 + RQASW model (models/edc.py, untouched)
# with Br35H's confirmed settings (stop_grad=True, lr_encoder=5e-5,
# clip=1.0, num_eval_iter=400, var_reg_weight/ema_momentum).
#
# Subsamples the TRAINING set only to --num_train_samples images
# (test set always stays full-size). Single seed per run — the
# "best AUC during training" (peak across eval checkpoints) is already
# tracked internally by methods/edc1.py's train() loop via best_eval_auc,
# so no changes to the training/eval logic are needed — we just read it.
#
# NOTE: unlike edc_br35h.py's 5-seed ensemble, this script runs ONE seed
# per sample size (matches the paper's Section F methodology — the point
# is the train-time AUC trajectory, not seed-ensembling). This keeps the
# APTOS and Br35H sample-efficiency numbers apples-to-apples.
#
# Usage (run once per sample size):
#   python runners_edc/edc_br35h_sampeff.py --num_train_samples 100  --save_name edc_br35h_n100
#   python runners_edc/edc_br35h_sampeff.py --num_train_samples 200  --save_name edc_br35h_n200
#   python runners_edc/edc_br35h_sampeff.py --num_train_samples 500  --save_name edc_br35h_n500
#   python runners_edc/edc_br35h_sampeff.py --num_train_samples 1000 --save_name edc_br35h_n1000
#   python runners_edc/edc_br35h_sampeff.py --num_train_samples 2000 --save_name edc_br35h_n2000
#   # full-dataset point: reuse your existing confirmed Table 1 result (99.96% AUC) — no re-run needed

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

from helper_modules.utils import get_logger, count_parameters, over_write_args_from_file
from helper_modules.train_utils import TBLog, get_optimizer_v2, get_multistep_schedule_with_warmup
from methods.edc1 import EDC
from datasets.dataset_br35h import AD_Dataset
from datasets.data_utils import get_data_loader
from models.edc import R50_R50          # confirmed original backbone — untouched file
from configs.config_br35h import DATASET_DIR

import warnings
warnings.filterwarnings("ignore")


def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes','true','t','y','1'): return True
    elif v.lower() in ('no','false','f','n','0'): return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def main_worker(args):
    args.gpu = int(args.gpu)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    args.device = device

    save_path = os.path.join(args.save_dir, args.save_name)
    os.makedirs(save_path, exist_ok=True)
    logger = get_logger(args.save_name, save_path, "INFO")
    logger.warning(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(args.gpu)}")

    random.seed(args.seed); torch.manual_seed(args.seed)
    np.random.seed(args.seed); cudnn.deterministic = True

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

    full_n = len(train_dset.img_paths)
    logger.info(f"Full training set size: {full_n}")

    # ---- SAMPLE-EFFICIENCY: subsample TRAINING set only ------------------
    if args.num_train_samples is not None and args.num_train_samples < full_n:
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(full_n, args.num_train_samples, replace=False)
        train_dset.img_paths = [train_dset.img_paths[i] for i in idx]
        train_dset.targets   = [train_dset.targets[i]   for i in idx]
        logger.info(f"Subsampled training set to {len(train_dset.img_paths)} "
                    f"images ({100*len(train_dset.img_paths)/full_n:.2f}% of full set)")
    else:
        logger.info("Using FULL training set (no subsampling)")
    # -----------------------------------------------------------------------

    logger.info(f"Test set size: {len(eval_dset.img_paths)} "
                f"(Normal: {eval_dset.targets.count(0)}, Abnormal: {eval_dset.targets.count(1)})")

    generator_lb = torch.Generator().manual_seed(args.seed)
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

    # ✅ Br35H confirmed settings: stop_grad=True, var_reg_weight/ema_momentum passed through
    model = R50_R50(
        img_size=args.img_size, train_encoder=True,
        stop_grad=True, reshape=True, bn_pretrain=False,
        var_reg_weight=args.var_reg_weight,
        ema_momentum=args.ema_momentum,
        use_rqasw=args.use_rqasw,
    )
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 0.01
    model = model.to(device)

    runner = EDC(
        model=model, num_eval_iter=args.num_eval_iter,
        amap_reduction='max', tb_log=None, logger=logger,
    )
    logger.info(f"Trainable Params: {count_parameters(runner.model)}")

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
    runner.tb_log = TBLog(save_path, "tb", use_tensorboard=False)

    eval_dict = runner.train(args, device=device, logger=logger)

    final_auc = eval_dict['eval/AUC']          # AUC at end of training
    best_auc  = eval_dict['eval/best_auc']     # peak AUC seen during training
    best_it   = eval_dict['eval/best_it']

    print("\n======== SAMPLE-EFFICIENCY RESULT — BR35H ========")
    print(f"Training samples used : {len(train_dset.img_paths)} "
          f"({100*len(train_dset.img_paths)/full_n:.2f}% of full set)")
    print(f"Final AUC (end of training)      : {final_auc*100:.2f}%")
    print(f"Best AUC (peak during training)  : {best_auc*100:.2f}%  @ iter {best_it}")
    print("====================================================\n")

    # Append to a running results CSV so all sample sizes accumulate in one place
    results_csv = os.path.join(args.save_dir, "sample_efficiency_br35h_results.csv")
    write_header = not os.path.exists(results_csv)
    with open(results_csv, "a") as f:
        if write_header:
            f.write("num_train_samples,final_auc,best_auc,best_it,save_name\n")
        f.write(f"{len(train_dset.img_paths)},{final_auc:.4f},{best_auc:.4f},"
                f"{best_it},{args.save_name}\n")
    logger.warning(f"Result appended to {results_csv}")
    logger.warning("COMPLETED.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir',           type=str,      default='./saved_models')
    parser.add_argument('-sn', '--save_name',   type=str,      default='edc_br35h_sampeff')
    parser.add_argument('-o', '--overwrite',    action='store_true', default=True)
    parser.add_argument('--resume',             action='store_true', default=False)

    parser.add_argument('--num_train_iter',     type=int,      default=4000)
    parser.add_argument('--num_eval_iter',      type=int,      default=400)

    parser.add_argument('-bsz','--batch_size',  type=int,      default=32)
    parser.add_argument('--eval_batch_size',    type=int,      default=64)

    parser.add_argument('--optim',              type=str,      default='AdamW')
    # ✅ Br35H confirmed lr/lr_encoder (matches edc_br35h.py fix)
    parser.add_argument('--lr',                 type=float,    default=0.0005)
    parser.add_argument('--lr_encoder',         type=float,    default=5e-05)
    parser.add_argument('--momentum',           type=float,    default=0.9)
    parser.add_argument('--weight_decay',       type=float,    default=1e-4)
    parser.add_argument('--amp',                type=str2bool, default=False)
    parser.add_argument('--clip',               type=float,    default=1.0)
    parser.add_argument('--var_reg_weight',     type=float,    default=0.1)
    parser.add_argument('--ema_momentum',       type=float,    default=0.999)
    parser.add_argument('--use_rqasw',          type=str2bool, default=True)

    # ---- Sample-efficiency specific ----
    parser.add_argument('--num_train_samples',  type=int,      default=None,
                         help='Subsample training set to this many images. '
                              'Omit / leave None to use the full training set.')

    parser.add_argument('--data_dir',           type=str,      default=DATASET_DIR)
    parser.add_argument('-ds','--dataset',      type=str,      default='brain')
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

    main_worker(args)