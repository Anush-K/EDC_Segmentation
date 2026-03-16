import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import LGGSegDataset
from unet import UNet


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_PATH = "../LGG_SEG"
BATCH_SIZE = 8
EPOCHS = 30
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

def run_experiment(use_heatmap):

    dataset = LGGSegDataset(DATASET_PATH, use_heatmap=use_heatmap)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=4)

    in_channels = 4 if use_heatmap else 3

    model = UNet(in_channels).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    criterion = nn.BCELoss()

    for epoch in range(EPOCHS):

        loss = train_epoch(model, train_loader, optimizer, criterion)

        dice, iou = evaluate(model, test_loader)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Loss {loss:.4f} | Dice {dice:.4f} | IoU {iou:.4f}"
        )

    return dice, iou


# ---------------------------
# MAIN
# ---------------------------

if __name__ == "__main__":

    print("\n--- Baseline: Image only ---\n")

    dice1, iou1 = run_experiment(use_heatmap=False)

    print("\n--- Proposed: Image + Heatmap ---\n")

    dice2, iou2 = run_experiment(use_heatmap=True)

    print("\n===== FINAL RESULTS =====")

    print(f"Baseline Dice: {dice1:.4f}")
    print(f"Baseline IoU : {iou1:.4f}")

    print(f"Proposed Dice: {dice2:.4f}")
    print(f"Proposed IoU : {iou2:.4f}")