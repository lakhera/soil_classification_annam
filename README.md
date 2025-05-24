# Soil-Classification - Soil Image Classification Challenge

This repository contains a PyTorch-based image classification pipeline to identify soil types (Alluvial, Black, Clay, and Red) from soil images using a ResNet-18 model and transfer learning.

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
├── train/
├── train_labels.csv
├── test/
└── test_ids.csv

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