import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils import get_coco_datasets
from models import get_faster_rcnn_model

# Hyperparameters
COCO_ROOT = "data/coco"  # Path to COCO dataset
NUM_CLASSES = 91         # COCO has 80 classes + 1 background
BATCH_SIZE = 1           # Use a batch size of 1 for visualization

def visualize_predictions(model, dataset, device, num_images=5, confidence_threshold=0.5):
    """
    Visualize predictions from the model on the validation dataset.

    Args:
        model: The Faster R-CNN model.
        dataset: COCO validation dataset.
        device: The device to run inference on (CPU or GPU).
        num_images: Number of images to visualize.
        confidence_threshold: Minimum confidence score for displaying a box.
    """
    model.eval()

    # Select random images from the dataset
    for idx in range(num_images):
        image, target = dataset[idx]
        image_tensor = image.to(device).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            predictions = model(image_tensor)[0]

        # Extract predicted boxes, labels, and scores
        pred_boxes = predictions['boxes']
        pred_labels = predictions['labels']
        pred_scores = predictions['scores']

        # Filter predictions by confidence threshold
        high_conf_idx = pred_scores > confidence_threshold
        pred_boxes = pred_boxes[high_conf_idx]
        pred_labels = pred_labels[high_conf_idx]
        pred_scores = pred_scores[high_conf_idx]

        # Plot the image with predictions
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image.permute(1, 2, 0).cpu())  # Convert from CxHxW to HxWxC

        # Draw ground truth boxes
        for box in target['boxes']:
            x_min, y_min, x_max, y_max = box.cpu().numpy()
            rect = patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=2,
                edgecolor='blue',
                facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(
                x_min,
                y_min - 5,
                "GT",
                color='blue',
                fontsize=12,
                bbox=dict(facecolor='white', alpha=0.5)
            )

        # Draw predicted boxes
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            x_min, y_min, x_max, y_max = box.cpu().numpy()
            rect = patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(
                x_min,
                y_min - 10,
                f"Pred: {label.item()} ({score:.2f})",
                color='red',
                fontsize=12,
                bbox=dict(facecolor='white', alpha=0.5)
            )

        plt.axis('off')
        plt.show()

def main():
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load validation dataset
    print("Loading COCO validation dataset...")
    _, val_dataset = get_coco_datasets(COCO_ROOT)

    # Initialize the model
    print("Loading Faster R-CNN model...")
    model = get_faster_rcnn_model(NUM_CLASSES)
    model.load_state_dict(torch.load("faster_rcnn_epoch_10.pth"))  # Load the last checkpoint
    model.to(device)

    # Visualize predictions
    print("Visualizing predictions...")
    visualize_predictions(model, val_dataset, device)

if __name__ == "__main__":
    main()
