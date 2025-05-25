# src/postprocessing.py

import pandas as pd

# generate submission csv from predictions

def generate_submission(predictions, label_encoder, output_path="/kaggle/working/submission.csv"):
    submission_df = pd.DataFrame(predictions, columns=['image_id', 'label_encoded'])
    submission_df['soil_type'] = label_encoder.inverse_transform(submission_df['label_encoded'])
    submission_df = submission_df[['image_id', 'soil_type']]
    submission_df.to_csv(output_path, index=False)
    return submission_df