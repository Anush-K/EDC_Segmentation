import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import random
import cv2
import numpy as np

# 🔥 FIXED IMPORT (important)
from segmentation.dataset_busi import BUSISegDataset
from segmentation.unet import UNet


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 🔥 USE ABSOLUTE PATH (avoid errors)
BASE_DIR = "/home/cs24d0008/EDC_Segmentation/EDC-master"
DATASET_PATH = os.path.join(BASE_DIR, "datasets/BUSI_SEG")

BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-4


# ---------------- Dice ----------------
def dice_score(pred, target):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + 1e-8) / (pred.sum() + target.sum() + 1e-8)


# ---------------- IoU ----------------
def iou_score(pred, target):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + 1e-8) / (union + 1e-8)


# ---------------- Dice Loss ----------------
def dice_loss(pred, target):
    smooth = 1e-8
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice


# ---------------- Train ----------------
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


# ---------------- Evaluate ----------------
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

    return dice_total / len(loader), iou_total / len(loader)


# ---------------- Experiment ----------------
def run_experiment(use_heatmap, use_pseudo, save_dir):

    dataset = BUSISegDataset(
        DATASET_PATH,
        use_heatmap=use_heatmap,
        use_pseudo=use_pseudo
    )

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=4)

    in_channels = 4 if use_heatmap else 3

    model = UNet(in_channels).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

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

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss {loss:.4f} | Dice {dice:.4f} | IoU {iou:.4f}")

    save_grid_predictions(model, test_loader, save_dir, use_heatmap)

    return best_dice, best_iou


# ---------------- Visualization ----------------
def save_grid_predictions(model, loader, save_dir, use_heatmap):

    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):

            x = x.to(DEVICE)
            pred = model(x)
            pred = (pred > 0.5).float()

            for i in range(x.shape[0]):

                img = x[i].cpu().numpy()
                gt = y[i].numpy()[0]
                pr = pred[i].cpu().numpy()[0]

                img_rgb = (np.transpose(img[:3], (1,2,0)) * 255).astype(np.uint8)
                gt_img = (gt * 255).astype(np.uint8)
                pr_img = (pr * 255).astype(np.uint8)

                gt_img = cv2.cvtColor(gt_img, cv2.COLOR_GRAY2BGR)
                pr_img = cv2.cvtColor(pr_img, cv2.COLOR_GRAY2BGR)

                panels = [img_rgb]

                if use_heatmap:
                    heatmap = (img[3] * 255).astype(np.uint8)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
                    panels.append(heatmap)

                panels.extend([pr_img, gt_img])

                grid = np.concatenate(panels, axis=1)

                save_path = os.path.join(save_dir, f"sample_{idx}_{i}.png")
                cv2.imwrite(save_path, grid)

            break


# ---------------- MAIN ----------------
if __name__ == "__main__":

    print("\n--- Baseline ---\n")
    dice1, iou1 = run_experiment(False, False, "outputs/busi_baseline")

    print("\n--- Heatmap Guided ---\n")
    dice2, iou2 = run_experiment(True, False, "outputs/busi_heatmap")

    print("\n===== FINAL RESULTS =====")
    print(f"Baseline Dice: {dice1:.4f}")
    print(f"Baseline IoU : {iou1:.4f}")
    print(f"Heatmap Dice: {dice2:.4f}")
    print(f"Heatmap IoU : {iou2:.4f}")