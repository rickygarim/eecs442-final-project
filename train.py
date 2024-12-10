import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from models import get_faster_rcnn_model
from utils import get_coco_datasets

# Hyperparameters
COCO_ROOT = "data/coco"  # Path to COCO dataset
NUM_CLASSES = 91         # COCO has 80 classes + 1 background
BATCH_SIZE = 4
NUM_EPOCHS = 10
LEARNING_RATE = 0.005
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9

def collate_fn(batch):
    """
    Custom collate function for handling variable size tensors.
    """
    return tuple(zip(*batch))

def train_one_epoch(model, optimizer, train_loader, device):
    """
    Train the model for one epoch.
    """
    model.train()
    epoch_loss = 0

    for images, targets in train_loader:
        # Move images and targets to the device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()

    return epoch_loss / len(train_loader)

def main():
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    print("Loading COCO datasets...")
    train_dataset, val_dataset = get_coco_datasets(COCO_ROOT)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_fn)

    # Initialize the model
    print("Initializing Faster R-CNN model...")
    model = get_faster_rcnn_model(NUM_CLASSES)
    model.to(device)

    # Define optimizer and learning rate scheduler
    optimizer = SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    lr_scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    # Train the model
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        epoch_loss = train_one_epoch(model, optimizer, train_loader, device)
        lr_scheduler.step()

        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}")

        # Save the model checkpoint
        checkpoint_path = f"faster_rcnn_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")

    print("Training complete.")

if __name__ == "__main__":
    main()
