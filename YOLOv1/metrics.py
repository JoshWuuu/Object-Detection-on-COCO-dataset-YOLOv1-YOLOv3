import torch
import torch.nn as nn
from collections import Counter

class YoLoLoss(nn.Module):
    """YOLO Loss function, calculates the loss for all the cells in the grid for the YOLOv1 model."""
    def __init__(self, S=7, B=2, C=20) -> None:
        super(YoLoLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        """
        Args:
            predictions (torch.Tensor): (N, S * S * (C + B * 5)) tensor
            target (torch.Tensor): (N, S, S, C + B * 5) tensor
        Returns:
            torch.Tensor: scalar tensor containing the loss
        """
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, bestbox = torch.max(ious, dim=0) # bestbox is the index of the best box, 0 or 1
        exists_box = target[..., 20].unsqueeze(3)  # 1 if there is an object in the cell, 0 otherwise

        # Loss for box coordinates (x, y, w, h)
        box_predictions = exists_box * (
            # determine which box is the best box
            (
                bestbox * predictions[..., 26:30]
                + (1 - bestbox) * predictions[..., 21:25]
            )
        )

        box_targets = exists_box * target[..., 21:25]

        # width and height of the box if there is a target box in the cell
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6))
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        # (N, S, S, 4) -> (N*S*S, 4)
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # Loss of object confidence
        pred_box = (
            bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21]
        )
        # (N, S, S, 1) -> (N*S*S)
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21]),
        )

        # Loss for no object confidence
        # (N, S, S, 1) -> (N, S*S)
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )
        
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )


        # Loss for classes
        # (N, S, S, 20) -> (N*S*S, 20)
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
            torch.flatten(exists_box * target[..., :20], end_dim=-2),
        )

        loss = (self.lambda_coord * box_loss 
                + object_loss 
                + self.lambda_noobj * no_object_loss 
                + class_loss)

        return loss
    
def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union

    Input:
    - boxes_preds: tensor, Predictions of Bounding Boxes (BATCH_SIZE, 4)
    - boxes_labels: tensor, Correct Labels of Boxes (BATCH_SIZE, 4)
    - box_format: str, midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
    - tensor: Intersection over union for all examples
    """

    # Slicing idx:idx+1 in order to keep tensor dimensionality
    # Doing ... in indexing if there would be additional dimensions
    # Like for Yolo algorithm which would have (N, S, S, 4) in shape
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # Need clamp(0) in case they do not intersect, then we want intersection to be 0
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

def non_max_suppression(
        predictions,
        iou_threshold,
        threshold,
        box_format="corners"
):
    """
    Does Non Max Suppression given bboxes

    Inputs:
    - bboxes: list, list of lists containing all bboxes with each bboxes specified as [class_pred, prob_score, x1, y1, x2, y2]
    - iou_threshold: float, threshold where predicted bboxes is correct
    - threshold: float, threshold to remove predicted bboxes (independent of IoU) 
    - box_format: str, "midpoint" or "corners" used to specify bboxes

    Returns:
    - bboxes_after_nms: list, bboxes after performing NMS given a specific IoU threshold
    """
    assert type(predictions) == list
    bboxes = [box for box in predictions if box[1] > threshold]
    # sort bboxes with highest probability score first  
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [
            # compare same class bboxes only, and if iou is less than iou threshold, keep the box for the next iteration
            box for box in bboxes if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format
            ) < iou_threshold
        ]
        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

def mean_average_precision(
        pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    """
    Calculates mean average precision for single iou threshold
    
    Input:
    - pred_boxes: list, list of lists containing all bboxes with each bboxes, [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
    - true_boxes: list, similar as pred_boxes except all the correct ones 
    - iou_threshold: float, threshold where predicted bboxes is correct
    - box_format: str, "midpoint" or "corners" used to specify bboxes
    - num_classes: int, number of classes

    Returns:
    - res: float, mAP value across all classes given a specific IoU threshold 
    """
    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # find all bboxes with the same class
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)
        
        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # count bbox of c class for each image, img 0 has 2 bbox, img 1 has 3 bbox, amount_bboxes = {0: 2, 1: 3}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # amount_boxes = {0: tensor([0., 0.]), 1: tensor([0., 0., 0.])}
        # update amount_boxes with the amount of bboxes in each image
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)
        # sort detections by prob_score
        detection.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # start with the highest prob_score detection bbox
        for detection_idx, detection in enumerate(detections):
            # number of target bboxes in the image for the current class
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]
            num_gts = len(ground_truth_img)
            # only the best iou can be true positive
            best_iou = 0
            
            # idx is the index of the target bbox in the image
            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )
                # find the best iou between the detection bbox and all the target bboxes in the image for the current class
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # maybe the target bbox has already been found by another detection bbox with higher prob_score
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        # calculate precision and recall
        # [1,1,0,1,0,0,1,0,0,0] -> [1,2,2,3,3,3,4,4,4,4]
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.div(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon)) 
        # add 0 to the beginning of precisions and recalls
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # calculate average precision
        average_precision = torch.trapz(precisions, recalls)
        average_precisions.append(average_precision)
    
    res = sum(average_precisions) / len(average_precisions)
    return res

def unit_test1():
    pass

if __name__ == "__main__":
    unit_test1()