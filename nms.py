import torch

from IOU import intersection_over_union

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

def unit_test1():
    pass    

if __name__ == "__main__":
    unit_test1()