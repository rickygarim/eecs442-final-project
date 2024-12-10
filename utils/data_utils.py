from torchvision.datasets import CocoDetection
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip

def get_coco_datasets(coco_root):
    """
    Load the COCO dataset with transformations.

    Args:
        coco_root (str): Path to the COCO dataset directory.

    Returns:
        tuple: Training and validation datasets.
    """
    # Define transformations for training and validation
    train_transform = Compose([
        ToTensor(),
        RandomHorizontalFlip(0.5)
    ])
    val_transform = Compose([
        ToTensor()
    ])

    # Load the datasets
    train_dataset = CocoDetection(
        root=f"{coco_root}/train2017",
        annFile=f"{coco_root}/annotations/instances_train2017.json",
        transform=train_transform
    )
    val_dataset = CocoDetection(
        root=f"{coco_root}/val2017",
        annFile=f"{coco_root}/annotations/instances_val2017.json",
        transform=val_transform
    )
    return train_dataset, val_dataset
