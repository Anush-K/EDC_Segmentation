import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import random
import cv2
import numpy as np

from dataset import LGGSegDataset
from unet import UNet


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_PATH = "../LGG_SEG"
BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-4


# ---------------------------
# Dice metric
# ---------------------------

def dice_score(pred, target):

    pred = (pred > 0.5).float()

    intersection = (pred * target).sum()

    return (2. * intersection + 1e-8) / (
        pred.sum() + target.sum() + 1e-8
    )


# ---------------------------
# IoU metric
# ---------------------------

def iou_score(pred, target):

    pred = (pred > 0.5).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    return (intersection + 1e-8) / (union + 1e-8)

# ---------------------------
# Dice loss
# ---------------------------
def dice_loss(pred, target):

    smooth = 1e-8

    intersection = (pred * target).sum()

    dice = (2. * intersection + smooth) / (
        pred.sum() + target.sum() + smooth
    )

    return 1 - dice

# ---------------------------
# Training loop
# ---------------------------
def train_epoch(model, loader, optimizer, criterion):


    model.train()

    total_loss = 0

    for x, y in tqdm(loader):

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        pred = model(x)

        loss = criterion(pred, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ---------------------------
# Evaluation loop
# ---------------------------

def evaluate(model, loader):

    model.eval()

    dice_total = 0
    iou_total = 0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(x)

            dice_total += dice_score(pred, y).item()
            iou_total += iou_score(pred, y).item()

    dice = dice_total / len(loader)
    iou = iou_total / len(loader)

    return dice, iou


# ---------------------------
# Train function
# ---------------------------

def run_experiment(use_heatmap, use_pseudo, save_dir=None):

    dataset = LGGSegDataset(DATASET_PATH, use_heatmap=use_heatmap, use_pseudo=use_pseudo)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=4)

    in_channels = 4 if use_heatmap else 3

    model = UNet(in_channels).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    #criterion = nn.BCELoss()

    bce = nn.BCELoss()

    def criterion(pred, target):
        return bce(pred, target) + dice_loss(pred, target)

    best_dice = 0
    best_iou = 0

    for epoch in range(EPOCHS):

        loss = train_epoch(model, train_loader, optimizer, criterion)

        dice, iou = evaluate(model, test_loader)

        if dice > best_dice:
            best_dice = dice
            best_iou = iou

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Loss {loss:.4f} | Dice {dice:.4f} | IoU {iou:.4f}"
        )

    save_grid_predictions(model, test_loader, save_dir, use_heatmap, use_pseudo)
    return best_dice, best_iou


def save_grid_predictions(model, loader, save_dir, use_heatmap, use_pseudo):

    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    results = []

    with torch.no_grad():
        for x, y in loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(x)
            pred_bin = (pred > 0.5).float()

            inter = (pred_bin * y).sum(dim=(1,2,3))
            union = pred_bin.sum(dim=(1,2,3)) + y.sum(dim=(1,2,3)) - inter

            dice = (2 * inter) / (pred_bin.sum(dim=(1,2,3)) + y.sum(dim=(1,2,3)) + 1e-8)
            iou = inter / (union + 1e-8)

            for i in range(x.shape[0]):
                results.append({
                    "dice": dice[i].item(),
                    "iou": iou[i].item(),
                    "image": x[i].cpu().numpy(),
                    "pred": pred_bin[i].cpu().numpy()[0],
                    "gt": y[i].cpu().numpy()[0],
                })

    # sort by Dice
    results = sorted(results, key=lambda r: r["dice"], reverse=True)

    mid = len(results) // 2
    selected = [
        random.choice(results[:3]),
        random.choice(results[mid-1:mid+2]),
        random.choice(results[-3:])
    ]

    def add_label(img, text):
        out = img.copy()
        cv2.putText(out, text, (10,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255,255,255), 2)
        return out

    for idx, s in enumerate(selected):

        img = s["image"]
        pred = s["pred"]
        gt = s["gt"]

        # image
        img_rgb = (np.transpose(img[:3], (1,2,0)) * 255).astype(np.uint8)

        # prediction
        pred_img = cv2.cvtColor((pred*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # GT (real OR pseudo)
        gt_img = cv2.cvtColor((gt*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        panels = [add_label(img_rgb, "Image")]

        # -----------------------------
        # Approach 1: Image → GT
        # -----------------------------
        if not use_heatmap and not use_pseudo:
            panels += [
                add_label(pred_img, "Prediction"),
                add_label(gt_img, "Ground Truth")
            ]

        # -----------------------------
        # Approach 2: Image + Heatmap → GT
        # -----------------------------
        elif use_heatmap and not use_pseudo:
            heatmap = img[3]
            heatmap = (heatmap * 255).astype(np.uint8)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)

            panels += [
                add_label(heatmap, "Heatmap (Input)"),
                add_label(pred_img, "Prediction"),
                add_label(gt_img, "Ground Truth")
            ]

        # -----------------------------
        # Approach 3: Image → Heatmap (Pseudo GT)
        # -----------------------------
        elif not use_heatmap and use_pseudo:
            panels += [
                add_label(pred_img, "Prediction"),
                add_label(gt_img, "Pseudo GT (Heatmap)")
            ]

        grid = np.concatenate(panels, axis=1)

        cv2.putText(
            grid,
            f"Dice: {s['dice']:.4f} | IoU: {s['iou']:.4f}",
            (10,45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,255),
            2
        )

        save_path = os.path.join(save_dir, f"sample_{idx}.png")
        cv2.imwrite(save_path, grid)

        print(f"Saved {save_path}")

# ---------------------------
# MAIN
# ---------------------------

if __name__ == "__main__":

    ## Currently dropped this heatmap as groundtruth approach, do not consider this approach anywhere
    # print("\n--- Proposed: Image --> Heatmap as Groundtruth ---\n")

    # dice3, iou3 = run_experiment(use_heatmap=False, use_pseudo = True, save_dir="outputs/pseudo_gt")

    print("\n--- Baseline: Image only --> Actual Groundtruth ---\n")

    dice1, iou1 = run_experiment(use_heatmap=False, use_pseudo = False, save_dir="outputs/baseline")

    print("\n--- Proposed: Image + Heatmap --> Actual Groundtruth ---\n")

    dice2, iou2 = run_experiment(use_heatmap=True, use_pseudo = False,  save_dir="outputs/heatmap_guided")

    print("\n===== FINAL RESULTS =====")

    print(f"Baseline Dice: {dice1:.4f}")
    print(f"Baseline IoU : {iou1:.4f}")

    print(f"Heatmap-Guided Dice: {dice2:.4f}")
    print(f"Proposed IoU : {iou2:.4f}")

    print(f"Pseudo-GT Dice: {dice3:.4f}")
    print(f"Pseudo-GT IoU : {iou3:.4f}")