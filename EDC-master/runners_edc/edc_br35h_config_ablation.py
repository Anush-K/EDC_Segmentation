# runners_edc/edc_br35h_config_ablation.py
#
# SECTION E: Progressive Configuration Ablation (Br35H, Configs 1-3)
# Standalone file — does NOT modify edc_br35h.py or methods/edc1.py.
# Fills in the missing Br35H Config 1/2/3 rows to match the EDC-style
# progressive ablation you already have for APTOS (Table 7).
#
# Config 4 (EDC baseline, 99.85% AUC) and Config 5 (RQASW, 99.96% AUC)
# are already covered by your existing edc_br35h.py / edc_br35h_perscale
# runs — no need to rerun those here.
#
# Preset -> (train_encoder, stop_grad, reshape[=L_global], use_rqasw):
#   1 : (False, True,  False, False)   frozen encoder baseline
#   2 : (True,  False, False, False)   + optimize encoder (no stop-grad)
#   3 : (True,  True,  False, False)   + stop-gradient
#
# Single seed per run (seed=0), matching the same "AUC / AUC-best-during-
# training" semantics already used for Table 7's APTOS AUC† column and
# for your sample-efficiency scripts — no 5-seed ensembling needed here,
# since this table reports a single progressive run per config, exactly
# like EDC's own published Configs 1-3.
#
# Usage:
#   python runners_edc/edc_br35h_config_ablation.py --preset 1 --save_name edc_br35h_config1
#   python runners_edc/edc_br35h_config_ablation.py --preset 2 --save_name edc_br35h_config2
#   python runners_edc/edc_br35h_config_ablation.py --preset 3 --save_name edc_br35h_config3

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


PRESETS = {
    # preset: (train_encoder, stop_grad, reshape, use_rqasw)
    1: (False, True,  False, False),   # frozen encoder baseline
    2: (True,  False, False, False),   # + optimize encoder, no stop-grad
    3: (True,  True,  False, False),   # + stop-gradient
}


def main_worker(args):
    args.gpu = int(args.gpu)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    args.device = device

    train_encoder, stop_grad, reshape, use_rqasw = PRESETS[args.preset]

    save_path = os.path.join(args.save_dir, args.save_name)
    os.makedirs(save_path, exist_ok=True)
    logger = get_logger(args.save_name, save_path, "INFO")
    logger.warning(f"Using device: {device}")
    logger.info(f"Config preset {args.preset}: train_encoder={train_encoder}, "
                f"stop_grad={stop_grad}, reshape(L_global)={reshape}, use_rqasw={use_rqasw}")
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

    logger.info(f"Train set: {len(train_dset.img_paths)} images (full, no subsampling)")
    logger.info(f"Test set: {len(eval_dset.img_paths)} images "
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

    model = R50_R50(
        img_size=args.img_size,
        train_encoder=train_encoder,
        stop_grad=stop_grad,
        reshape=reshape,
        bn_pretrain=False,
        var_reg_weight=args.var_reg_weight,
        ema_momentum=args.ema_momentum,
        use_rqasw=use_rqasw,
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
    best_auc  = eval_dict['eval/best_auc']     # peak AUC during training (= AUC†)
    best_it   = eval_dict['eval/best_it']

    print(f"\n======== CONFIG {args.preset} RESULT — Br35H ========")
    print(f"train_encoder={train_encoder}  stop_grad={stop_grad}  "
          f"reshape(L_global)={reshape}  use_rqasw={use_rqasw}")
    print(f"AUC  (end of training)  : {final_auc*100:.2f}%")
    print(f"AUC† (best during training) : {best_auc*100:.2f}%  @ iter {best_it}")
    print("====================================================\n")

    results_csv = os.path.join(args.save_dir, "br35h_config_ablation_results.csv")
    write_header = not os.path.exists(results_csv)
    with open(results_csv, "a") as f:
        if write_header:
            f.write("preset,train_encoder,stop_grad,reshape,use_rqasw,auc,auc_best,best_it,save_name\n")
        f.write(f"{args.preset},{train_encoder},{stop_grad},{reshape},{use_rqasw},"
                f"{final_auc:.4f},{best_auc:.4f},{best_it},{args.save_name}\n")
    logger.warning(f"Result appended to {results_csv}")
    logger.warning("COMPLETED.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--preset',              type=int,      required=True, choices=[1, 2, 3],
                         help='Which config preset to run (1, 2, or 3). '
                              'Config 4/5 already covered by existing edc_br35h.py runs.')

    parser.add_argument('--save_dir',           type=str,      default='./saved_models')
    parser.add_argument('-sn', '--save_name',   type=str,      default='edc_br35h_config')
    parser.add_argument('-o', '--overwrite',    action='store_true', default=True)
    parser.add_argument('--resume',             action='store_true', default=False)

    parser.add_argument('--num_train_iter',     type=int,      default=4000)
    parser.add_argument('--num_eval_iter',      type=int,      default=400)

    parser.add_argument('-bsz','--batch_size',  type=int,      default=32)
    parser.add_argument('--eval_batch_size',    type=int,      default=64)

    parser.add_argument('--optim',              type=str,      default='AdamW')
    parser.add_argument('--lr',                 type=float,    default=0.0005)
    parser.add_argument('--lr_encoder',         type=float,    default=5e-05)
    parser.add_argument('--momentum',           type=float,    default=0.9)
    parser.add_argument('--weight_decay',       type=float,    default=1e-4)
    parser.add_argument('--amp',                type=str2bool, default=False)
    parser.add_argument('--clip',               type=float,    default=1.0)
    parser.add_argument('--var_reg_weight',     type=float,    default=0.1)
    parser.add_argument('--ema_momentum',       type=float,    default=0.999)

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