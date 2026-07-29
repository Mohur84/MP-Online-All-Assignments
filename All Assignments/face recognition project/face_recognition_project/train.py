"""
Training Script
================
Trains a face recognition model on the LFW dataset.
Uses transfer learning (ResNet18) for high accuracy.
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.cnn_model import build_model
from utils.data_loader import load_lfw_dataset, IMG_SIZE


CONFIG = {
    "model"        : "resnet18",
    "min_faces"    : 25,
    "max_classes"  : 25,
    "epochs"       : 50,
    "batch_size"   : 16,
    "learning_rate": 3e-4,
    "weight_decay" : 1e-4,
    "target_acc"   : 90.0,
    "save_dir"     : "checkpoints",
    "img_size"     : IMG_SIZE,
}


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 20 == 0:
            print(f"    Batch {batch_idx+1}/{len(loader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Acc: {100.*correct/total:.1f}%")

    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), 100. * correct / total


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\nLoading LFW dataset...")
    train_loader, val_loader, class_names = load_lfw_dataset(
        root_dir="dataset/lfw-deepfunneled",
        min_faces=CONFIG["min_faces"],
        max_classes=CONFIG["max_classes"],
        batch_size=CONFIG["batch_size"],
        img_size=CONFIG["img_size"],
    )

    num_classes = len(class_names)
    print(f"\n{num_classes} classes | "
          f"{len(train_loader.dataset)} train | "
          f"{len(val_loader.dataset)} val images\n")

    model = build_model(CONFIG["model"], num_classes, pretrained=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {CONFIG['model']} ({total_params:,} parameters)\n")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    target_reached = False

    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    print("=" * 55)
    print(f"  Training for up to {CONFIG['epochs']} epochs "
          f"(target: {CONFIG['target_acc']:.0f}% val accuracy)")
    print("=" * 55)

    for epoch in range(1, CONFIG["epochs"] + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{CONFIG['epochs']}  |  LR: {current_lr:.6f}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(f"\n  Train -> Loss: {train_loss:.4f}  Acc: {train_acc:.1f}%")
        print(f"  Val   -> Loss: {val_loss:.4f}  Acc: {val_acc:.1f}%  ({elapsed:.1f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch"       : epoch,
                "model_state" : model.state_dict(),
                "optimizer"   : optimizer.state_dict(),
                "val_acc"     : val_acc,
                "num_classes" : num_classes,
                "class_names" : class_names,
                "model_name"  : CONFIG["model"],
                "img_size"    : CONFIG["img_size"],
            }, os.path.join(CONFIG["save_dir"], "best_model.pth"))
            print(f"  Saved best model (val_acc={val_acc:.1f}%)")

        if val_acc >= CONFIG["target_acc"]:
            print(f"\nTarget accuracy {CONFIG['target_acc']:.0f}% reached!")
            target_reached = True
            break

    with open(os.path.join(CONFIG["save_dir"], "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    status = "Target met" if target_reached else "Finished"
    print(f"\n{status}. Best val accuracy: {best_val_acc:.1f}%")
    return history, best_val_acc


if __name__ == "__main__":
    train()
