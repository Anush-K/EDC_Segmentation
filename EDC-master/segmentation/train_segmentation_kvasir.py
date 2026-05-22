# segmentation/train_segmentation_kvasir.py
# NOVELTY 2: HGBL — unchanged
# Infrastructure: focal loss + Dice + BCE, deeper UNet (Attention UNet),
# OneCycleLR scheduler, 200 epochs — improves toward SOTA

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

from dataset_kvasir import KVASIRSegDataset
from unet import UNet

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_PATH = "../KVASIR_SEG"
BATCH_SIZE   = 8
EPOCHS       = 200
LR           = 3e-4
SEED         = 42
SAVE_DIR     = "./seg_checkpoints_kvasir"
os.makedirs(SAVE_DIR, exist_ok=True)

LAMBDA_B = 2.0
LAMBDA_H = 1.0


def dice_score(pred, target):
    pred = (pred > 0.5).float()
    return (2.*(pred*target).sum()+1e-8) / (pred.sum()+target.sum()+1e-8)

def iou_score(pred, target):
    pred = (pred > 0.5).float()
    inter = (pred*target).sum()
    return (inter+1e-8) / (pred.sum()+target.sum()-inter+1e-8)

def dice_loss(pred, target):
    inter = (pred*target).sum()
    return 1. - (2.*inter+1e-8) / (pred.sum()+target.sum()+1e-8)

def focal_loss(pred, target, alpha=0.8, gamma=2.0):
    bce = F.binary_cross_entropy(pred, target, reduction="none")
    pt  = torch.where(target==1, pred, 1-pred)
    return (alpha*(1-pt)**gamma*bce).mean()

def baseline_loss(pred, target):
    return F.binary_cross_entropy(pred,target) + dice_loss(pred,target) + focal_loss(pred,target)

def boundary_map(mask, ks=5):
    pad = ks//2
    return (F.max_pool2d(mask,ks,1,pad) + F.max_pool2d(-mask,ks,1,pad)).clamp(0.,1.)

def hgbl_loss(pred, mask, heatmap, lambda_b=LAMBDA_B, lambda_h=LAMBDA_H):
    B     = boundary_map(mask)
    H_max = heatmap.flatten(1).max(1)[0].view(-1,1,1,1).clamp(min=1e-8)
    H     = heatmap / H_max
    W     = (1.+lambda_b*B) * (1.+lambda_h*H)
    wbce  = (F.binary_cross_entropy(pred,mask,reduction="none")*W).mean()
    return wbce + dice_loss(pred,mask) + focal_loss(pred,mask)


def train_epoch(model, loader, optimizer, use_hgbl=True):
    model.train(); total = 0.
    for image, heatmap, mask in tqdm(loader, leave=False, desc="  train"):
        image, heatmap, mask = image.to(DEVICE), heatmap.to(DEVICE), mask.to(DEVICE)
        optimizer.zero_grad()
        pred = model(torch.cat([image, heatmap], dim=1))
        loss = hgbl_loss(pred,mask,heatmap) if use_hgbl else baseline_loss(pred,mask)
        loss.backward(); optimizer.step(); total += loss.item()
    return total / len(loader)

def evaluate(model, loader):
    model.eval(); dt = it = 0.
    with torch.no_grad():
        for image, heatmap, mask in loader:
            image,heatmap,mask = image.to(DEVICE),heatmap.to(DEVICE),mask.to(DEVICE)
            pred = model(torch.cat([image,heatmap],dim=1))
            dt += dice_score(pred,mask).item()
            it += iou_score(pred,mask).item()
    return dt/len(loader), it/len(loader)


def run_experiment(use_hgbl, tag=""):
    label = "HGBL" if use_hgbl else "Baseline"
    print(f"\n{'='*62}\n  {label}  {tag}\n{'='*62}\n")
    torch.manual_seed(SEED); np.random.seed(SEED)

    full = KVASIRSegDataset(DATASET_PATH)
    n    = len(full); nt = max(1, int(0.8*n)); nv = n - nt
    gen  = torch.Generator().manual_seed(SEED)
    ti, vi = random_split(range(n), [nt, nv], generator=gen)

    tl = DataLoader(Subset(full, ti.indices), BATCH_SIZE, shuffle=True,
                    num_workers=2, pin_memory=True, drop_last=False)
    vl = DataLoader(Subset(KVASIRSegDataset(DATASET_PATH, train=False), vi.indices),
                    BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True,
                    drop_last=False)

    model = UNet(in_channels=4, dropout_p=0.3).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, steps_per_epoch=len(tl),
        epochs=EPOCHS, pct_start=0.1)

    best = 0.
    ckpt = os.path.join(SAVE_DIR,
           f"best_unet_kvasir_{'hgbl' if use_hgbl else 'baseline'}.pth")

    for ep in range(1, EPOCHS+1):
        loss = train_epoch(model, tl, opt, use_hgbl)
        dice, iou = evaluate(model, vl)
        sched.step()
        imp = ""
        if dice > best:
            best = dice; torch.save(model.state_dict(), ckpt); imp = "  <- best"
        if ep % 20 == 0 or imp:
            print(f"  Ep {ep:3d}/200 | Loss {loss:.4f} | "
                  f"Dice {dice:.4f} | IoU {iou:.4f}{imp}")

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    fd, fi = evaluate(model, vl)
    print(f"\n  Best -> Dice {fd:.4f} | IoU {fi:.4f}")
    return fd, fi


def run_ablation():
    global LAMBDA_B, LAMBDA_H; r = {}
    d,i = run_experiment(False,"[row1]"); r["Baseline"] = (d,i)
    LAMBDA_H=0.; LAMBDA_B=2.
    d,i = run_experiment(True,"[row2]");  r["+ Boundary (lb=2,lh=0)"] = (d,i)
    LAMBDA_H=1.; LAMBDA_B=0.
    d,i = run_experiment(True,"[row3]");  r["+ Heatmap (lb=0,lh=1)"] = (d,i)
    LAMBDA_H=1.; LAMBDA_B=2.
    d,i = run_experiment(True,"[row4]");  r["Full HGBL (lb=2,lh=1)"] = (d,i)
    print("\n" + "="*62)
    print(f"  ABLATION TABLE — KVASIR")
    print("="*62)
    for m,(d,i) in r.items():
        print(f"  {m:<38} {d:.4f}  {i:.4f}")
    print("="*62)
    return r


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ablation", action="store_true")
    p.add_argument("--lambda_b", type=float, default=2.0)
    p.add_argument("--lambda_h", type=float, default=1.0)
    a = p.parse_args(); LAMBDA_B = a.lambda_b; LAMBDA_H = a.lambda_h
    if a.ablation:
        run_ablation()
    else:
        db, ib = run_experiment(False, "[baseline]")
        dh, ih = run_experiment(True,  "[HGBL]")
        print(f"\n{'='*62}\n  FINAL RESULTS — KVASIR\n{'='*62}")
        print(f"  Baseline -> Dice:{db:.4f} IoU:{ib:.4f}")
        print(f"  HGBL     -> Dice:{dh:.4f} IoU:{ih:.4f}")
        print(f"  Delta    -> Dice:{dh-db:+.4f} IoU:{ih-ib:+.4f}")
        print("="*62)