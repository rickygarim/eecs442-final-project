import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from utils import get_coco_datasets
from models import get_faster_rcnn_model

# Hyperparameters
COCO_ROOT = "data/coco"  # Path to COCO dataset
NUM_CLASSES = 91         # COCO has 80 classes + 1 background
BATCH_SIZE = 4

def collate_fn(batch):
    """
    Custom collate function for handling variable size tensors.
    """
    return tuple(zip(*batch))

def evaluate_model(model, val_loader, device, iou_threshold=0.5):
    """
    Evaluate the model on the validation dataset.

    Args:
        model: The Faster R-CNN model.
        val_loader: DataLoader for validation data.
        device: The device to run evaluation on (CPU or GPU).
        iou_threshold: Threshold for Intersection over Union (IoU).

    Returns:
        float: Mean Average Precision (mAP).
    """
    model.eval()
    all_iou_scores = []
    all_predictions = 0
    all_true_positives = 0

    with torch.no_grad():
        for images, targets in val_loader:
            # Move images and targets to the device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Get model predictions
            outputs = model(images)

            for target, output in zip(targets, outputs):
                true_boxes = target['boxes']
                pred_boxes = output['boxes']
                pred_scores = output['scores']

                # Filter predictions by confidence score
                high_conf_idx = pred_scores > 0.5
                pred_boxes = pred_boxes[high_conf_idx]

                # Calculate IoU and count true positives
                if len(pred_boxes) > 0 and len(true_boxes) > 0:
                    iou_matrix = box_iou(pred_boxes, true_boxes)
                    max_ious, _ = torch.max(iou_matrix, dim=0)
                    all_iou_scores.append(max_ious.mean().item())
                    all_true_positives += (max_ious > iou_threshold).sum().item()
                all_predictions += len(pred_boxes)

    # Compute mAP
    mean_iou = sum(all_iou_scores) / len(all_iou_scores) if all_iou_scores else 0
    precision = all_true_positives / all_predictions if all_predictions > 0 else 0

    return mean_iou, precision

def main():
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load validation dataset
    print("Loading COCO validation dataset...")
    _, val_dataset = get_coco_datasets(COCO_ROOT)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_fn)

    # Initialize the model
    print("Loading Faster R-CNN model...")
    model = get_faster_rcnn_model(NUM_CLASSES)
    model.load_state_dict(torch.load("faster_rcnn_epoch_10.pth"))  # Load the last checkpoint
    model.to(device)

    # Evaluate the model
    print("Evaluating the model...")
    mean_iou, precision = evaluate_model(model, val_loader, device)
    print(f"Evaluation Results:\nMean IoU: {mean_iou:.4f}\nPrecision: {precision:.4f}")

if __name__ == "__main__":
    main()
