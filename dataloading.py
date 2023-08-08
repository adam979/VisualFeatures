import torch
from torchvision import datasets, transforms
from skimage import exposure
import os
import numpy as np
from swinv2 import SwinV2, extract_visual_features
import torch.multiprocessing as mp
import multiprocessing
from tqdm import tqdm

# Set the paths for the dataset directories
root_dir = "C:\\Users\\hassa\\Desktop\\SWIN\\archive"
train_dir = os.path.join(root_dir, "train")
val_dir = os.path.join(root_dir, "val")
test_dir = os.path.join(root_dir, "test")

# Define the transformation applied to each image
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Lambda(apply_equalize_adapthist),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Load the train dataset
train_dataset = datasets.ImageFolder(train_dir, transform=transform)

# Load the validation dataset
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

# Load the test dataset
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

# Create data loaders to iterate over the datasets
batch_size = 8
num_workers = 4

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
)


# Define the main function for the script
def main():
    # Initialize the Swin Transformer model
    model = SwinV2()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Extract visual features using the Swin Transformer model
    # Modify the extract_swin_features function
    def extract_swin_features(loader, model):
        global_features_list = []
        local_features_list = []
        labels = []
        print("Starting feature extraction...")
        for i, (images, targets) in enumerate(loader):
            images = images.to(device)  # Move images to the GPU if available
            global_features, local_features = extract_visual_features(images, model)
            global_features_list.append(global_features)
            local_features_list.append(local_features)
            labels.append(targets)
            print(f"Iteration {i+1}/{len(loader)} completed.")
        global_features = torch.cat(global_features_list, dim=0)
        local_features = torch.cat(local_features_list, dim=0)
        labels = torch.cat(labels, dim=0)
        print("Feature extraction completed.")
        return global_features, local_features, labels

    # Extract global and local features from the train, validation, and test datasets
    train_global_features, train_local_features, train_labels = extract_swin_features(
        train_loader, model
    )
    val_global_features, val_local_features, val_labels = extract_swin_features(
        val_loader, model
    )
    test_global_features, test_local_features, test_labels = extract_swin_features(
        test_loader, model
    )

    # Merge the global and local features
    train_features = torch.cat((train_global_features, train_local_features), dim=1)
    val_features = torch.cat((val_global_features, val_local_features), dim=1)
    test_features = torch.cat((test_global_features, test_local_features), dim=1)

    # Save the extracted features and labels for future use
    torch.save(train_features, "train_features.pt")
    torch.save(val_features, "val_features.pt")
    torch.save(test_features, "test_features.pt")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    mp.set_start_method("spawn")
    main()
