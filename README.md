# README

This repository contains the code for preparing and loading the Indiana Chest X-ray dataset using PyTorch and the Swin Transformer model.

## Files

1. `datasetprep.py`: This script prepares the dataset by organizing the images into train, validation, and test directories based on the provided CSV files. It reads the projection and report information from the CSV files and copies the images to the appropriate directories. The split of the dataset into train, validation, and test sets is done using the `train_test_split` function from scikit-learn.

2. `dataloader.py`: This script loads and preprocesses the prepared dataset using PyTorch's `datasets` module. It defines the transformations applied to each image, loads the train, validation, and test datasets, creates data loaders for iterating over the datasets, and extracts visual features using the Swin Transformer model.

3. `swin_model.py`: This script contains the implementation of the Swin Transformer model. It includes the definition of the SwinTransformer class and the function `extract_visual_features` for extracting visual features from images using the Swin Transformer model.

## Dataset Preparation

The `datasetprep.py` script is used to prepare the dataset for training the Swin Transformer model. It expects the following files:

- `indiana_projections.csv`: This CSV file contains the projection information for the images, including the UID, filename, and projection type.

- `indiana_reports.csv`: This CSV file contains the report information for the images, including the UID, findings, and impression.

The script creates the train, validation, and test directories if they don't exist. It reads the projection and report information from the CSV files and organizes the images into the appropriate directories based on the UID and the split of the dataset.

## Data Loading and Preprocessing

The `dataloader.py` script is responsible for loading and preprocessing the prepared dataset using PyTorch. It performs the following steps:

1. Sets the paths for the dataset directories.

2. Defines the transformations applied to each image, including resizing, conversion to tensor, adaptive histogram equalization, and normalization.

3. Loads the train, validation, and test datasets using PyTorch's `ImageFolder` class and applies the defined transformations.

4. Creates data loaders for iterating over the datasets, specifying the batch size and number of workers.

5. Defines the Swin Transformer model using the `SwinTransformer` class from `swin_model.py`.

6. Extracts visual features from the train, validation, and test datasets using the Swin Transformer model and the `extract_visual_features` function.

7. Saves the extracted features and labels as PyTorch tensors for future use.

## Model Training

To train the Swin Transformer model using the prepared dataset, you can follow these steps:

1. Run the `datasetprep.py` script to prepare the dataset.

2. Run the `dataloader.py` script to load and preprocess the prepared dataset.

3. Implement the training loop and model training code based on your specific requirements. You can use the extracted features and labels obtained from the data loader for training the model.

4. Evaluate the trained model using the validation set and make predictions on the test set.

5. Fine-tune the model and iterate on the training process as needed.
