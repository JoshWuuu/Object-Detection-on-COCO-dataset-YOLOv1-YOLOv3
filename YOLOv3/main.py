import torch
import torch.optim as optim

from utils import (
    mean_average_precision,
    get_evaluation_bboxes,
    load_checkpoint,
    check_class_accuracy,
    get_loaders
)
from loss import YoloLoss
from model_build import YOLOv3
from model_train import train_fn
import config

def main():
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    train_loader, test_loader = get_loaders(
        train_csv_path=config.DATASET + "train.csv",
        test_csv_path=config.DATASET + "test.csv",
    )

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
        )
    
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    for epoch in range(config.NUM_EPOCHS):
        #plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors)
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)

        if epoch > 0 and epoch % 3 == 0:
            check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
            pred_boxes, true_boxes = get_evaluation_bboxes(
                test_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
            )
            mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            )
            print(f"MAP: {mapval.item()}")
            model.train()

if __name__ == '__main__':
    main()
