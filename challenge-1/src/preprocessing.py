# src/preprocessing.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# load and prepare data
def load_and_prepare_data(train_csv_path):
    df = pd.read_csv(train_csv_path)
    df['image'] = df['image_id']
    df['label'] = df['soil_type']
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label_encoded'], random_state=42)
    return train_df, val_df, le
