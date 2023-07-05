# Visual Features Extraction using Swin Transformer

This project demonstrates the extraction of visual features using the Swin Transformer model. It provides Python scripts for data loading, dataset preparation, and the SwinV2 model implementation. Additionally, it includes the necessary dataset files: `indiana_reports.csv` and `indiana_projections.csv`.

## Table of Contents

- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Scripts](#scripts)
- [Dataset](#dataset)
- [Model](#model)
- [References](#references)

## Introduction

Visual feature extraction is a crucial step in many computer vision tasks. The Swin Transformer is a state-of-the-art deep learning model that has shown excellent performance in various vision tasks, including image classification. This project demonstrates how to use the Swin Transformer model to extract visual features from images.

The project consists of three main scripts:

1. `dataloading.py`: Loads the dataset, applies necessary transformations, and creates data loaders for training, validation, and testing.
2. `datasetprep.py`: Prepares the dataset by organizing the images into train, validation, and test directories based on CSV files containing projection information and reports.
3. `swinv2.py`: Implements the SwinV2 model, which is a modified version of the Swin Transformer for visual feature extraction.

## Dependencies

To run the scripts in this project, the following dependencies are required:

- Python 3.x
- PyTorch
- torchvision
- scikit-learn
- tqdm
- pandas


## Usage

To use this project, follow the steps below:

1. Prepare the dataset by organizing the images into the appropriate directories using the `datasetprep.py` script. Make sure to place the `indiana_reports.csv` and `indiana_projections.csv` files in the same directory as the script.

2. Run the `dataloading.py` script to load the dataset, apply transformations, and create data loaders for training, validation, and testing. Adjust the batch size, number of workers, and other parameters according to your needs.

3. Customize the `swinv2.py` script if necessary, such as modifying the model architecture or adjusting the pre-trained weights.

4. Execute the `dataloading.py` script to extract visual features from the images using the SwinV2 model. The extracted features will be saved as `.pt` files for future use.

## Scripts

### `dataloading.py`

This script loads the dataset, applies transformations, and creates data loaders for training, validation, and testing. The main function performs the following steps:

1. Initialize the SwinV2 model.

2. Extract visual features using the SwinV2 model and the provided data loaders.

3. Save the extracted features and labels as `.pt` files for future use.

### `datasetprep.py`

This script prepares the dataset by organizing the images into train, validation, and test directories based on the `indiana_reports.csv` and `indiana_projections.csv` files. The main function performs the following steps:

1. Read the projection information from `indiana_projections.csv` and store it in a dictionary.

2. Read the report information from `indiana_reports.csv` and store it in a dictionary.

3. Split the UIDs into train, validation, and test sets.

4. Copy the images to the appropriate directories based on their UIDs and projection information(continued)

### `swinv2.py`

This script implements the SwinV2 model, which is a modified version of the Swin Transformer specifically designed for visual feature extraction. The SwinV2 class inherits from `torch.nn.Module` and consists of the following components:

1. The Swin Transformer backbone: This is the core of the model and is responsible for extracting visual features from input images. It utilizes the Swin Transformer architecture with specific configurations for patch size, embedding dimension, depths, number of heads, window size, MLP ratio, stochastic depth probability, and number of classes.

2. Loading pre-trained weights (optional): The SwinV2 constructor allows you to load pre-trained weights for the backbone using the `weights` parameter. It uses the `torch.hub.load_state_dict_from_url` function to download the pre-trained weights if a URL is provided.

3. Forward pass: The forward method takes input images and passes them through the backbone to extract visual features. It returns the extracted features.

4. Feature extraction helper function: The `extract_visual_features` function takes a batch of images and the SwinV2 model and extracts visual features using the model. It returns the extracted features.

## Dataset

The dataset used in this project consists of medical images and associated reports. It includes two CSV files:

- `indiana_reports.csv`: This file contains information about the reports associated with the images. It includes UIDs, findings, and impressions.

- `indiana_projections.csv`: This file contains information about the projections of the images. It includes UIDs and projection types.

## Model

The SwinV2 model implemented in this project is a modified version of the Swin Transformer specifically designed for visual feature extraction. It utilizes the Swin Transformer architecture as the backbone to extract visual features from input images.

The `SwinV2` class in the `swinv2.py` script encapsulates the SwinV2 model and provides an interface for feature extraction. The backbone of the model is loaded with pre-trained weights, which can be customized by providing a URL to the `weights` parameter in the constructor.

## References

- Liu, Z., et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. _arXiv preprint arXiv:2103.14030_.

- PyTorch Documentation: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)

- torchvision Documentation: [https://pytorch.org/vision/stable/index.html](https://pytorch.org/vision/stable/index.html)

Please note that this README provides a high-level overview of the project. For more detailed explanations, refer to the source code and relevant documentation.

---

*Note: Make sure to include the necessary license information, acknowledgments, and any additional details specific to your project in the README.md file.*


