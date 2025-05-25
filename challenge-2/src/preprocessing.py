#%%
# src/training.py

# -------------------------
# Soil Image Classification Challenge - Part 2
# The Soil Image Classification Challenge is a machine learning competition organised by Annam.ai at IIT Ropar, serving as an initial task for shortlisted hackathon participants.
# Task: Classify provided image as Soil or NotSoil image , given Single label Binary Image classification.
# Team Name: RootCoders
# Team Members : Amit Lakhera, Vikramjeet, Pradipta Das, Jyoti Ghungru, Sukanya Saha
# -------------------------

from soil_classification import SoilClassificationModel

def main():
    # Update these paths to your actual data locations
    train_folder = "/kaggle/input/soil-classification-part-2/soil_competition-2025/train"          # Modify to your image folder path
    train_csv = "/kaggle/input/soil-classification-part-2/soil_competition-2025/train/train_labels.csv"  # Modify to your label csv

    # Initialize the model (you can adjust nu, kernel, gamma if needed)
    model = SoilClassificationModel(kernel='rbf', gamma='auto', nu=0.1)

    print("Starting training process...")

    # Train the model on all available training data
    model.fit(target_folder=train_folder, labels_csv=train_csv)

    # Save the trained model for future inference or testing
    model.save_model("soil_classification_model.pkl")

    print("Training completed and model saved as 'soil_classification_model.pkl'.")

if __name__ == "__main__":
    main()

