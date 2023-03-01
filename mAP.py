import torch
from collections import Counter
from IOU import intersection_over_union

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