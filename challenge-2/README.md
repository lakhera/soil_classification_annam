# Soil-Classification - Soil Image Classification Challenge Part-2

This repository contains a PyTorch-based image classification pipeline to i to classify the images to if they're soil image or not.

Organized by Annam.ai at IIT Ropar.

# Structure

1. download.sh – Checks if it's in the Kaggle environment and skips downloading.
2. preprocessing.py – Loads the dataset, encodes labels, and splits into train/val.
3. training.ipynb – Trains a ResNet-18 model on the soil images.
4. inference.ipynb – Loads the best model and performs inference on the test set.
5. postprocessing.py – Generates a submission CSV from predictions.

# Overview

The project was developed and run on Kaggle Notebooks with a P100 GPU. It includes:
- Data preprocessing and augmentation
- Custom PyTorch 'Dataset' loaders
- Transfer learning using pretrained ResNet-18
- Training with evaluation metrics
- Test-time inference and CSV submission generation

# Dataset (On Kaggle)

The dataset used is available in the Kaggle Soil Classification 2025 Challenge. It contains:
- train/ folder: training images
- train_labels.csv`: training labels
- test/ folder: test images
- test_ids.csv: test image names

# Directory structure:

/soil_classification-2025/
|____ train/
|____ train_labels.csv
|____ test/
|____ test_ids.csv

# Dependencies

This project runs on Kaggle which has most packages pre-installed. However, if you're running locally or on another platform, install the following:

bash
pip install torch torchvision pandas scikit-learn matplotlib pillow tqdm

# Setup and Usage (On Kaggle)

1. Create a new notebook on Kaggle.
2. Attach the Soil Classification 2025 dataset to your notebook.
3. Set the accelerator to GPU (P100):
4. Settings → Accelerator → GPU (P100)
5. Copy the code from soil_classification.py into a cell.
6. Run all cells. The training will start and the model will:
7. Save the best model as best_model.pth
8. Create the prediction file submission.csv