# Soil Image Single-Class Binary Classification

## Project Overview

This project provides a robust solution for **single-class binary image classification** using anomaly detection, designed for highly imbalanced soil image datasets. The pipeline leverages feature engineering and a One-Class SVM to identify "soil" images (normal class) and flag "not soil" images as anomalies, making it suitable for real-world cases with limited negative samples.

## Features

- **Advanced Feature Extraction:** Combines grayscale histogram, Local Binary Pattern (LBP), and Histogram of Oriented Gradients (HOG) for comprehensive image representation.
- **Anomaly Detection Approach:** Uses One-Class SVM to model the "soil" class and flag outliers.
- **Flexible Data Handling:** Supports training and testing using folders and CSV files for image management.
- **Comprehensive Evaluation:** Provides accuracy, precision, recall, F1-score, confusion matrix, and visualization plots.
- **Sample Visualization:** Visualizes predictions and errors for interpretability.
- **Model Persistence:** Save and load trained models for reuse.


## Dependencies

Install all dependencies using the provided `requirements.txt`:

```
numpy
opencv-python
pandas
matplotlib
seaborn
scikit-image
scikit-learn
```

Install with:

```bash
pip install -r requirements.txt
```


## Project Structure

```
.
├── final.py                  # Main code file
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── train/                    # Folder with training images
├── test/                     # Folder with test images
├── train_labels.csv          # CSV with columns: image_id, label
├── test_ids.csv              # CSV with column: image_id
```
---

## Data Preparation \& Feature Transformation

Images are loaded in grayscale and resized to 128x128 pixels. For each image, three sets of features are extracted:

- **Grayscale Histogram:** Captures the distribution of pixel intensities.
- **Local Binary Pattern (LBP):** Encodes local texture patterns using a radius of 3 and 24 points.
- **Histogram of Oriented Gradients (HOG):** Extracts shape and edge information with 9 orientations and 8x8 pixel cells.

These features are concatenated into a single 1D feature vector per image and standardized using `StandardScaler` to ensure uniform contribution across features. Images can be loaded from folders or filtered using a CSV file listing image IDs and labels[^1].

---

## Model and Training Strategy Used

The model uses a **One-Class Support Vector Machine (OneClassSVM)** with an RBF kernel, which is well-suited for anomaly detection in imbalanced datasets. Training is performed only on the "soil" class, learning its feature distribution to distinguish it from outliers. The pipeline allows for an optional validation split to assess model performance before full training. The main workflow is encapsulated in the `SoilClassificationModel` class, which manages feature extraction, scaling, model fitting, prediction, and evaluation[^1].

---

## Loss Function and Evaluation Metrics

The OneClassSVM does not use a traditional loss function; it optimizes a margin-based objective to separate the normal class from anomalies. For evaluation, the code computes accuracy, precision, recall, and F1-score. It also generates a classification report and confusion matrix, with visualizations for confusion matrix, metric comparison, and prediction distribution. These metrics help assess the model's ability to correctly identify soil versus not-soil images in an unbalanced setting[^1].

---

## Our Approach

- **Problem Framing:** The task is approached as anomaly detection, treating "soil" as the normal class and "not soil" as anomalies.
- **Feature Engineering:** Combines grayscale histogram, LBP, and HOG features for rich image representation.
- **Training:** One-Class SVM is trained only on the soil class, with optional validation to tune parameters.
- **Prediction \& Evaluation:** Converts SVM output to binary labels, evaluates with standard metrics, and provides visual diagnostics and sample image predictions for interpretability.
- **Model Persistence:** Includes utilities to save and load trained models for reuse[^1].

---

## How It Works

### 1. Data Preparation \& Feature Extraction

- Images are loaded in grayscale and resized to 128x128 pixels.
- Features are extracted using:
    - **Grayscale Histogram:** Captures intensity distribution.
    - **LBP:** Encodes local texture.
    - **HOG:** Captures shape and edge information.
- Features are concatenated and standardized for model input.


### 2. Model Training

- The `SoilClassificationModel` class wraps all logic.
- One-Class SVM (RBF kernel) is trained on "soil" images only.
- Optional validation split for performance tuning.


### 3. Prediction \& Evaluation

- Predicts on new/test images, labeling as "soil" (1) or "not soil" (0).
- Outputs metrics, confusion matrix, and visualizations.
- Generates `submission.csv` for test predictions.


### 4. Model Saving \& Loading

- Trained models can be saved and reloaded for later use.


## Usage

### Prepare your data

 Organize images in train/test folders and provide corresponding CSV files (`train_labels.csv`, `test_ids.csv`).


### Training and Evaluation

Update paths as needed and run:

```python
from final import SoilClassificationModel

train_folder = "path/to/train"
test_folder = "path/to/test"
train_csv = "path/to/train_labels.csv"
test_csv = "path/to/test_ids.csv"

model = SoilClassificationModel(nu=0.1)
trained_model, validation_metrics = model.train_and_evaluate_model(
    train_folder=train_folder,
    test_folder=test_folder,
    train_csv=train_csv,
    test_csv=test_csv,
    validation_split=0.2
)
model.save_model("soil_classification_model.pkl")
```


### Predict on New Data

```python
model = SoilClassificationModel.load_model("soil_classification_model.pkl")
image_ids, predictions = model.predict(test_folder, test_csv)
#image_ids, predictions = model.predict("path/to/test", "path/to/test_ids.csv")

```


### Evaluate Model

```python
metrics = model.evaluate_predictions(true_labels, predictions)
```


## Input CSV Formats

- **train_labels.csv:**


| image_id | label |
| :-- | :-- |
| img_00001.jpg | 1 |
| img_00002.jpg | 1 |

- **test_ids.csv:**


| image_id |
| :-- |
| img_10001.jpg |
| img_10002.jpg |


## Outputs

- `submission.csv`: Predictions for test images.
- `soil_classification_model.pkl`: Saved model.
- Visualizations: Confusion matrix, metrics comparison, prediction distribution, sample predictions.


## Notes

- Ensure all image files are present in the specified folders.
- The code assumes **single-class training** (all training labels must be the same).
- Adjust parameters (e.g., `nu`, `gamma`) as needed for your dataset.

---

For questions or issues, please open an issue on the repository.

<div style="text-align: center">⁂</div>


