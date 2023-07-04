import torch
from torchvision import datasets, transforms
from skimage import exposure
import os
from swin_model import extract_visual_features
import swin_model
from swin_model import SwinTransformer


# Set the paths for the dataset directories
root_dir = "C:\\Users\\hassa\\Desktop\\SWIN\\archive"
train_dir = os.path.join(root_dir, "train")
val_dir = os.path.join(root_dir, "val")
test_dir = os.path.join(root_dir, "test")

# Define the transformation applied to each image
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize the image to a fixed size
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Lambda(
            lambda x: exposure.equalize_adapthist(x.numpy(), clip_limit=0.03)
        ),  # Apply adaptive histogram equalization
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalize the image
    ]
)

# Load the train dataset
train_dataset = datasets.ImageFolder(train_dir, transform=transform)

# Load the validation dataset
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

# Load the test dataset
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

# Create data loaders to iterate over the datasets
batch_size = 32
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


model = SwinTransformer()


# Extract visual features using the Swin Transformer model
def extract_swin_features(loader, model):
    features = []
    labels = []
    for images, targets in loader:
        visual_features = extract_visual_features(images, model)
        features.append(visual_features)
        labels.append(targets)
    features = torch.cat(features)
    labels = torch.cat(labels)
    return features, labels


# Extract visual features from the train, validation, and test datasets
train_features, train_labels = extract_swin_features(train_loader, model)
val_features, val_labels = extract_swin_features(val_loader, model)
test_features, test_labels = extract_swin_features(test_loader, model)

# Save the extracted features and labels for future use
torch.save(train_features, "train_features.pt")
torch.save(train_labels, "train_labels.pt")
torch.save(val_features, "val_features.pt")
torch.save(val_labels, "val_labels.pt")
torch.save(test_features, "test_features.pt")
torch.save(test_labels, "test_labels.pt")
