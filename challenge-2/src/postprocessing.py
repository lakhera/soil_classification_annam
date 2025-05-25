# src/postprocessing.py

# This module contains the postprocessing steps for the soil image dataset predictions.
import pandas as pd

# Function to decode predictions from model output
def decode_predictions(predictions, label_map):
    return [label_map[pred] for pred in predictions]

# Function to save the decoded predictions to a CSV file for submission
def save_submission(ids, decoded_preds, output_csv="submission.csv"):
    submission = pd.DataFrame({'id': ids, 'label': decoded_preds})
    submission.to_csv(output_csv, index=False)
    print(f"Saved submission to {output_csv}")
