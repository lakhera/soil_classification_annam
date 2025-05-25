#%%
# src/postprocessing.py

# -------------------------
# Soil Image Classification Challenge - Part 2
# The Soil Image Classification Challenge is a machine learning competition organised by Annam.ai at IIT Ropar, serving as an initial task for shortlisted hackathon participants.
# Task: Classify provided image as Soil or NotSoil image , given Single label Binary Image classification.
# Team Name: RootCoders
# Team Members : Amit Lakhera, Vikramjeet, Pradipta Das, Jyoti Ghungru, Sukanya Saha
# -------------------------

from soil_classification import SoilClassificationModel

# Update these paths to your actual directories
train_folder = "/kaggle/input/soil-classification-part-2/soil_competition-2025/train"         # Modify to your train image folder path
test_folder = "/kaggle/input/soil-classification-part-2/soil_competition-2025/test"           # Modify to your test image folder path
train_csv = "/kaggle/input/soil-classification-part-2/soil_competition-2025/train_labels.csv" # Modify to your train csv path
test_csv = "/kaggle/input/soil-classification-part-2/soil_competition-2025/test_ids.csv"      # Modify to your test csv path

# Initialize model with parameters from code
model = SoilClassificationModel(
    kernel='rbf',
    gamma='auto',
    nu=0.1  # Anomaly detection sensitivity (0.1 from code example)
)

# Full training and testing pipeline
trained_model, validation_metrics = model.train_and_evaluate_model(
    train_folder=train_folder,
    test_folder=test_folder,
    train_csv=train_csv,
    test_csv=test_csv,
    validation_split=0.2  # 20% validation split as in code
)

# To load saved model and test separately (alternative approach)
model = SoilClassificationModel.load_model("soil_classification_model.pkl")
image_ids, predictions = model.predict(test_folder, test_csv)
print("Complete")

# Output will be:
# - submission.csv with test predictions
# - Confusion matrix plot
# - Metrics comparison plot
# - Prediction distribution plot
# - Sample image predictions grid


