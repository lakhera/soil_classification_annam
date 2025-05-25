# Soil-Classification - Soil Image Classification Challenge
# Team Name: RootCoders (Amit Lakhera, Vikramjeet, Jyoti Ghungru, Pradipta Das, Sukanya Saha)
# Last Modified: May 25, 2025
# Hackathon organized by Annam.ai at IIT Ropar.

This repository contains a PyTorch-based image classification pipeline to identify soil types (Alluvial, Black, Clay, and Red) from soil images using a ResNet-18 model and transfer learning.

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

# Dependencies

Install all dependencies using the provided `requirements.txt`:

```
numpy==1.24.3
pandas==1.5.3
matplotlib==3.7.1
Pillow==9.5.0
tqdm==4.65.0
torch==2.0.0
torchvision==0.15.1
scikit-learn==1.2.2
```

Install with:

```bash
pip install -r requirements.txt
```
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

---

For questions or issues, please open an issue on the repository.

<div style="text-align: center">⁂</div>