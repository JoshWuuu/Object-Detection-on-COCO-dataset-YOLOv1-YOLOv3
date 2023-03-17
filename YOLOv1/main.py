import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from model_build import YOLOv1
from model_train import train_fn
from metrics import *
from datasets import *
from metrics import *
import config

def main():
    transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])

    model = YOLOv1(split_size=7, num_boxes=2, num_classes=20).to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    loss_fn = YoLoLoss()

    if config.LOAD_MODEL:
        load_checkpoint(torch.load(config.LOAD_MODEL_FILE), model, optimizer)

    train_dataset = VOCDataset(
        "data/train.csv", config.IMG_DIR, config.LABEL_DIR, transform=transform
    )
    test_dataset = VOCDataset(
        "data/test.csv", config.IMG_DIR, config.LABEL_DIR, transform=transform
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
    )

    for epoch in range(config.EPOCHS):
        print(f"Epoch {epoch+1}/{config.EPOCHS}")
        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4
        )
        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Train mAP: {mean_avg_prec}")

        train_fn(train_loader, model, optimizer, loss_fn)

        if save_checkpoint(model, optimizer, filename=f"overfit.pth.tar"):
            print("Checkpoint saved!")

if __name__ == "__main__":
    main()
