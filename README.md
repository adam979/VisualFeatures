# Visual Feature Extraction Using Swin Transformer

This project demonstrates the extraction of visual features using the Swin Transformer model. It provides Python scripts for dataset preparation, data loading, and the SwinV2 model implementation. Additionally, it includes the necessary dataset files: `indiana_reports.csv` and `indiana_projections.csv`.

## Table of Contents

- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Scripts](#scripts)
- [Dataset](#dataset)
- [Model](#model)
- [References](#references)

## Introduction

Visual feature extraction is a fundamental step in various computer vision tasks. The Swin Transformer is a cutting-edge deep learning architecture that has demonstrated remarkable performance across multiple vision applications, including image classification. This project showcases the utilization of the Swin Transformer model to extract essential visual features from images.

The project encompasses three primary scripts:

1. **datasetprep.py**: Organizes the dataset images into train, validation, and test directories based on provided CSV files containing projection information and reports.
2. **dataloading.py**: Loads the dataset, applies necessary transformations, creates data loaders for training, validation, and testing, and extracts visual features using the SwinV2 model.
3. **swinv2.py**: Implements the SwinV2 model, a customized version of the Swin Transformer for visual feature extraction.

## Dependencies

To successfully run the scripts in this project, the following dependencies are required:

- Python 3.x
- PyTorch
- torchvision
- scikit-learn
- tqdm
- pandas

## Usage

To utilize this project, follow these steps:

1. **Dataset Preparation**: The first step involves preparing the dataset by executing the `datasetprep.py` script. This script reads the provided CSV files (`indiana_reports.csv` and `indiana_projections.csv`), organizes the images into appropriate directories, and performs the train-test split.

2. **Data Loading and Feature Extraction**: After dataset preparation, run the `dataloading.py` script. This script loads the dataset, applies transformations, creates data loaders for training, validation, and testing, and extracts visual features using the SwinV2 model.

3. **Model Customization (Optional)**: Modify the `swinv2.py` script if required, such as altering the model architecture or adjusting the pre-trained weights.

4. **Extracting Features**: Execute the `dataloading.py` script again to extract visual features from images using the SwinV2 model. The extracted features will be saved as `.pt` files for future use.

## Scripts

### datasetprep.py

This script prepares the dataset by organizing images into train, validation, and test directories based on `indiana_reports.csv` and `indiana_projections.csv` files. 
This script forms the basis for dataset preparation. It reads projection and report information from CSV files, splits UIDs into train, validation, and test sets, and copies images to respective directories based on UIDs and projection information. Running this script is the first step towards preparing the dataset for subsequent feature extraction.The main function undertakes the following steps:

- Reads projection information from the CSV file.
- Reads report information from another CSV file.
- Splits UIDs into train, validation, and test sets.
- Copies images to respective directories based on their UIDs and projection information. This script forms the foundation for subsequent data loading and feature extraction.

### dataloading.py

This script loads the dataset, applies transformations, and establishes data loaders for training, validation, and testing. The main function follows these steps:

- Initializes the SwinV2 model.
- Extracts visual features using the SwinV2 model and provided data loaders.
- Saves extracted features and labels as `.pt` files for future use.

### swinv2.py

This script implements the SwinV2 model, which is a tailored version of the Swin Transformer designed explicitly for visual feature extraction.

The SwinV2 class encapsulated in this script provides:

- **Swin Transformer Backbone**: This core component extracts visual features from input images. It utilizes the Swin Transformer architecture with specified configurations for patch size, embedding dimension, depths, number of heads, window size, MLP ratio, stochastic depth probability, and number of classes.

- **Pre-trained Weights Loading (optional)**: The constructor of SwinV2 allows loading pre-trained weights for the backbone using the `weights` parameter. It employs the `torch.hub.load_state_dict_from_url` function to download pre-trained weights from a URL if provided.

- **Forward Pass and Feature Extraction**:
  The `forward` method is where the magic happens. It takes input images and passes them through the Swin Transformer backbone to extract two types of features:
  
  - **Global Features**: These features are extracted directly from the Swin Transformer. They capture high-level contextual information from the entire image. After passing through the backbone, these features undergo adaptive average pooling to resize them to a fixed spatial size.
  
  - **Local Features**: These features are obtained by applying an additional convolutional layer to the global features. This convolutional operation serves to enhance and refine features that are more localized within the image. The `LocalFeatureExtractor` class encapsulates this process, involving a convolutional layer followed by a ReLU activation.

- **Extract Function**:
  The script provides an `extract_visual_features` function that simplifies the process of extracting global and local features using the SwinV2 model. This function is used within the `dataloading.py` script to extract features from input images.

### Dataset

The dataset for this project consists of medical images and associated reports. It includes two CSV files:

- `indiana_reports.csv`: Contains information about reports related to the images, including UIDs, findings, and impressions.
- `indiana_projections.csv`: Contains information about the projections of the images, including UIDs and projection types.

### Model

The SwinV2 model implemented in this project is a customized version of the Swin Transformer, tailored for visual feature extraction. It leverages the Swin Transformer architecture as the backbone to extract crucial visual features from input images.

The SwinV2 class encapsulated in the `swinv2.py` script serves as the SwinV2 model interface for feature extraction. The backbone of the model can be loaded with pre-trained weights, allowing for customization by providing a URL through the `weights` parameter in the constructor. The script demonstrates how to utilize the SwinV2 model to extract global and local features from images.
## References

- Liu, Z., et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. _arXiv preprint arXiv:2103.14030_.

- PyTorch Documentation: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)

- torchvision Documentation: [https://pytorch.org/vision/stable/index.html](https://pytorch.org/vision/stable/index.html)




