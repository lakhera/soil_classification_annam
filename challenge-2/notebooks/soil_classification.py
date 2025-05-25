# -------------------------
# Soil Image Classification Challenge - Part 2
# The Soil Image Classification Challenge is a machine learning competition organised by Annam.ai at IIT Ropar, serving as an initial task for shortlisted hackathon participants. Competitors will build models to classify image as soil or not soil image.
# Task: Classify each provided soil image into one of the four soil types (Alluvial, Black, Clay, Red).
# Deadline: May 25, 2025, 11:59 PM IST
# Team Name: RootCoders
# Team Members : Amit Lakhera, Vikramjeet, Pradipta Das, Jyoti Ghungru, Sukanya Saha
# -------------------------

import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import local_binary_pattern, hog
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_curve, auc, precision_recall_curve)
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


class FeatureExtractor:
    """
    A class for extracting features from images using Histogram, Local Binary Pattern (LBP),
    and Histogram of Oriented Gradients (HOG).

    Attributes:
        lbp_radius (int): The radius for the Local Binary Pattern (LBP) feature extraction.
        lbp_n_points (int): The number of points for the LBP, calculated as 8 * lbp_radius.
    """

    def __init__(self):
        """
        Initializes the FeatureExtractor with default parameters for LBP.
        """
        self.lbp_radius = 3
        self.lbp_n_points = 8 * self.lbp_radius

    def extract_features(self, image_path):
        """
        Extracts a combination of histogram, LBP, and HOG features from a grayscale image for Feature Extraction.

        Args:
            image_path (str): The file path to the image.
        Returns:
            np.ndarray: A 1D array containing the concatenated features (histogram, LBP, and HOG).
        Raises:
            ValueError: If the image cannot be read from the provided path.
        """
        # Read the image in grayscale mode
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")

        # Resize the image to a fixed size of 128x128 pixels
        image = cv2.resize(image, (128, 128))

        # Extract histogram features
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])  # Compute grayscale histogram
        hist = cv2.normalize(hist, hist).flatten()  # Normalize and flatten the histogram

        # Extract Local Binary Pattern (LBP) features
        lbp = local_binary_pattern(image, self.lbp_n_points, self.lbp_radius, method="uniform")
        (lbp_hist, _) = np.histogram(
            lbp.ravel(),
            bins=np.arange(0, self.lbp_n_points + 3),
            range=(0, self.lbp_n_points + 2)
        )
        lbp_hist = lbp_hist.astype("float")
        lbp_hist /= (lbp_hist.sum() + 1e-6)  # Normalize the LBP histogram

        # Extract Histogram of Oriented Gradients (HOG) features
        hog_features = hog(
            image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            feature_vector=True
        )

        # Combine all features into a single feature vector
        return np.hstack([hist, lbp_hist, hog_features])


class SoilClassificationModel:
    """
    A wrapper class for the One-Class SVM model, designed for anomaly detection in soil image datasets.
    This class includes methods for loading images, extracting features, training the model,
    making predictions, and visualizing results.

    Attributes:
        model (OneClassSVM): The One-Class SVM model instance.
        scaler (StandardScaler): A scaler for normalizing feature data.
        feature_extractor (FeatureExtractor): An instance of the FeatureExtractor class
            for extracting features from images.
    """

    def __init__(self, kernel='rbf', gamma='auto', nu=0.1):
        """
        Initializes the SoilClassificationModel with a One-Class SVM model,
        a standard scaler for feature normalization, and a feature extractor for image processing.

        Args:
            kernel (str): Kernel type to be used in the SVM. Defaults to 'rbf'.
            gamma (str or float): Kernel coefficient. Defaults to 'auto'.
            nu (float): An upper bound on the fraction of training errors and a lower bound of the
                        fraction of support vectors. Should be in the interval (0, 1]. Defaults to 0.1.
        """
        self.model = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
        self.scaler = StandardScaler()
        self.feature_extractor = FeatureExtractor()

    def load_images_from_folder(self, folder, labels_csv=None):
        """
        Loads images from a specified folder and optionally filters them based on a CSV file.

        Args:
            folder (str): The path to the folder containing the images.
            labels_csv (str, optional): Path to a CSV file containing image_id column.
                For training: must contain both image_id and label columns.
                For testing: may contain only image_id column.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: An array of extracted features for each image.
                - list: A list of image file names corresponding to the extracted features.

        Raises:
            ValueError: If the CSV contains multiple unique labels in training mode.
        """
        features = []
        image_files = []

        if labels_csv is not None:
            # Load the CSV file
            try:
                df = pd.read_csv(labels_csv)

                # Check if this is a test CSV (with only image_id) or training CSV (with image_id and label)
                if 'image_id' not in df.columns:
                    raise ValueError("CSV must contain 'image_id' column")

                is_test_csv = 'label' not in df.columns

                if not is_test_csv:
                    # Training mode - verify single class for training data
                    unique_labels = df['label'].unique()
                    if len(unique_labels) > 1:
                        raise ValueError("Code is trained for Single Binary Classification for Unbalanced data")
                    print(f"Found {len(df)} image entries in CSV with label: {unique_labels[0]}")
                else:
                    # Testing mode
                    print(f"Found {len(df)} image entries in test CSV")

                # Process only the images listed in the CSV
                for _, row in df.iterrows():
                    filename = row['image_id']
                    path = os.path.join(folder, filename)

                    if os.path.exists(path):
                        try:
                            # Extract features from the image
                            feat = self.feature_extractor.extract_features(path)
                            features.append(feat)
                            image_files.append(filename)
                        except Exception as e:
                            print(f"Skipping {filename}: {e}")
                    else:
                        print(f"Warning: Image file not found: {path}")

            except Exception as e:
                print(f"Error loading CSV file: {e}")
                raise
        else:
            # Original behavior - load all images from folder
            for filename in sorted(os.listdir(folder)):
                # Check if the file has a valid image extension
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    path = os.path.join(folder, filename)
                    try:
                        # Extract features from the image
                        feat = self.feature_extractor.extract_features(path)
                        features.append(feat)
                        image_files.append(filename)
                    except Exception as e:
                        print(f"Skipping {filename}: {e}")

        print(f"Loaded {len(features)} valid images from {folder}")
        return np.array(features), image_files

    def predict(self, test_folder, test_csv=None):
        """
        Predicts the class labels for images in the test folder.

        Args:
            test_folder (str): Path to the folder containing test images.
            test_csv (str, optional): Path to CSV file with image_id column for test images.

        Returns:
            tuple: A tuple containing:
                - list: Image filenames in the test folder.
                - list: Predicted labels (1=soil, 0=not soil) for each image.

        Raises:
            ValueError: If no features can be extracted from the test folder.
        """
        features, image_names = self.load_images_from_folder(test_folder, test_csv)
        if features.size == 0:
            raise ValueError(" No features found in test folder.")
        scaled = self.scaler.transform(features)
        preds = self.model.predict(scaled)
        preds = [1 if p == 1 else 0 for p in preds]  # 1=normal (soil), -1=outlier (not soil) -> convert to 0
        return image_names, preds

    def evaluate_predictions(self, true_labels, predicted_labels, plot_results=True):
        """
        Comprehensive evaluation of model predictions with metrics and optional visualizations.

        Args:
            true_labels (list or np.ndarray): The ground truth labels for the dataset.
            predicted_labels (list or np.ndarray): The predicted labels from the model.
            plot_results (bool, optional): Whether to generate and display visualizations. Defaults to True.

        Returns:
            dict: A dictionary containing the calculated metrics:
                - 'Accuracy': The accuracy of the predictions.
                - 'Precision': The precision of the predictions.
                - 'Recall': The recall of the predictions.
                - 'F1 Score': The F1 score of the predictions.
        """
        print("\n" + "=" * 50)
        print(" COMPREHENSIVE MODEL EVALUATION")
        print("=" * 50)

        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)

        # Print detailed metrics
        print(f"\n PERFORMANCE METRICS:")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1 Score:  {f1:.4f}")

        print(f"\n CLASSIFICATION REPORT:")
        print(classification_report(true_labels, predicted_labels, digits=4))

        print(f"\n CONFUSION MATRIX:")
        cm = confusion_matrix(true_labels, predicted_labels)
        print(cm)

        # Create metrics dictionary for plotting
        metrics_dict = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }

        if plot_results:
            print("=" * 60)
            print(f"\n GENERATING VISUALIZATIONS...")
            print("=" * 60)
            self.plot_confusion_matrix(true_labels, predicted_labels)
            self.plot_metrics_comparison(metrics_dict)
            self.plot_prediction_distribution(predicted_labels)

        return metrics_dict

    def fit(self, target_folder, labels_csv=None):
        """
        Trains the model on images from the specified folder.

        Args:
            target_folder (str): Path to the folder containing training images.
            labels_csv (str, optional): Path to CSV file with image_id and label columns.

        Raises:
            ValueError: If no features can be extracted from the training folder.
        """
        features, _ = self.load_images_from_folder(target_folder, labels_csv)
        if features.size == 0:
            raise ValueError(" No features found in training folder.")
        scaled = self.scaler.fit_transform(features)
        self.model.fit(scaled)
        print(" Model trained successfully.")

    def train_and_evaluate_model(self, train_folder, test_folder, train_csv=None, test_csv=None, validation_split=0.2, random_state=42):
        """
        Trains the model using the provided training data, evaluates it on a validation set (if specified),
        and tests it on the test data. Generates evaluation metrics, visualizations, and saves predictions.

        Args:
            train_folder (str): Path to the folder containing training images.
            test_folder (str): Path to the folder containing test images.
            train_csv (str, optional): Path to CSV file with training image_id and label columns.
            test_csv (str, optional): Path to CSV file with test image_id column.
            validation_split (float, optional): Proportion of training data to use for validation. Defaults to 0.2.
            random_state (int, optional): Random seed for reproducibility of the train-validation split. Defaults to 42.

        Returns:
            tuple: A tuple containing:
                - SoilClassificationModel: The trained classifier instance.
                - dict or None: Validation metrics if validation_split > 0, otherwise None.

        Raises:
            ValueError: If no training data is found in the specified folder.
        """
        print(" STARTING SOIL CLASSIFICATION TRAINING AND EVALUATION")
        print("=" * 60)

        # Load training data
        print(f"\n Loading training data from: {train_folder}")
        if train_csv:
            print(f" Using labels from: {train_csv}")

        train_features, train_files = self.load_images_from_folder(train_folder, train_csv)

        if len(train_features) == 0:
            raise ValueError(" No training data found!")

        # Split training data for validation
        val_metrics = None
        if validation_split > 0:
            print(f"\n Splitting training data (validation split: {validation_split})")
            train_feat, val_feat, train_names, val_names = train_test_split(
                train_features, train_files, test_size=validation_split,
                random_state=random_state
            )

            # Train on training subset
            print(f" Training samples: {len(train_feat)}")
            print(f" Validation samples: {len(val_feat)}")

            # Fit scaler and model on training data
            scaled_train = self.scaler.fit_transform(train_feat)
            self.model.fit(scaled_train)
            print(" Model trained on training subset.")

            # Validate on validation set
            scaled_val = self.scaler.transform(val_feat)
            val_preds = self.model.predict(scaled_val)
            val_preds = [1 if p == 1 else 0 for p in val_preds]

            # Create true labels for validation (all should be 1 since they're from training folder)
            val_true_labels = [1] * len(val_preds)

            print(f"\n VALIDATION RESULTS:")
            val_metrics = self.evaluate_predictions(val_true_labels, val_preds, plot_results=True)
        else:
            # Train on all data
            scaled_train = self.scaler.fit_transform(train_features)
            self.model.fit(scaled_train)
            print(" Model trained on all training data.")

        # Test on test folder
        print(f"\n Testing on: {test_folder}")
        if test_csv:
            print(f" Using test IDs from: {test_csv}")
        image_ids, test_predictions = self.predict(test_folder, test_csv)

        # Save submission
        df = pd.DataFrame({"image_id": image_ids, "label": test_predictions})
        df.to_csv("submission.csv", index=False)
        print(" Saved submission.csv")

        # Show test predictions distribution
        self.plot_prediction_distribution(test_predictions, "test_predictions_distribution.png")

        # Show sample images with predictions
        print("\n" + "=" * 50)
        print(f" Generating sample image predictions...")
        print("=" * 50)
        self.plot_sample_images_with_predictions(test_folder, image_ids, test_predictions,
                                                 num_samples=20, save_path="sample_test_predictions.png")

        return self, val_metrics

    def plot_confusion_matrix(self, true_labels, predicted_labels, save_path="confusion_matrix.png"):
        """
        Plot and save confusion matrix heatmap.

        Args:
            true_labels (list or np.ndarray): The ground truth labels for the dataset.
            predicted_labels (list or np.ndarray): The predicted labels from the model.
            save_path (str, optional): Path where the plot image will be saved. Defaults to "confusion_matrix.png".
        """
        cm = confusion_matrix(true_labels, predicted_labels)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['NotSoil', 'Soil'],
                    yticklabels=['NotSoil', 'Soil'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f" Confusion matrix saved to {save_path}")

    def plot_metrics_comparison(self, metrics_dict, save_path="metrics_comparison.png"):
        """
        Plot comparison of different metrics.

        Args:
            metrics_dict (dict): Dictionary containing metric names as keys and their values.
            save_path (str, optional): Path where the plot image will be saved. Defaults to "metrics_comparison.png".
        """
        metrics = list(metrics_dict.keys())
        values = list(metrics_dict.values())

        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        plt.title('Model Performance Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f" Metrics comparison saved to {save_path}")

    def plot_prediction_distribution(self, predicted_labels, save_path="prediction_distribution.png"):
        """
        Plot distribution of predictions.

        Args:
            predicted_labels (list or np.ndarray): The predicted labels from the model.
            save_path (str, optional): Path where the plot image will be saved. Defaults to "prediction_distribution.png".
        """
        unique, counts = np.unique(predicted_labels, return_counts=True)

        # Create a dictionary mapping values to their counts
        counts_dict = dict(zip(unique, counts))

        # Get counts for each category (0 and 1), default to 0 if not present
        not_soil_count = counts_dict.get(0, 0)
        soil_count = counts_dict.get(1, 0)

        plt.figure(figsize=(8, 6))
        bars = plt.bar(['NotSoil', 'Soil'], [not_soil_count, soil_count], color=['lightcoral', 'lightgreen'])
        plt.title('Distribution of Predictions')
        plt.ylabel('Count')

        # Add value labels on bars
        for bar, count in zip(bars, [not_soil_count, soil_count]):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     str(count), ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f" Prediction distribution saved to {save_path}")

    def plot_sample_images_with_predictions(self, folder_path, image_names, predictions, num_samples=8, save_path="sample_predictions.png"):
        """
        Plot sample images with their predictions.

        Args:
            folder_path (str): Path to the folder containing images.
            image_names (list): List of image file names.
            predictions (list): List of predictions corresponding to the images.
            num_samples (int, optional): Number of sample images to plot. Defaults to 20.
            save_path (str, optional): Path where the plot image will be saved. Defaults to "sample_predictions.png".
        """
        if len(image_names) < num_samples:
            num_samples = len(image_names)

        # Select random samples
        indices = np.random.choice(len(image_names), num_samples, replace=False)

        # Calculate grid dimensions - make it more flexible
        grid_size = int(np.ceil(np.sqrt(num_samples)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(14, 14))  # Reduced figure size
        axes = axes.flatten()

        for i, idx in enumerate(indices):
            if i >= len(axes):  # Safety check
                break

            img_path = os.path.join(folder_path, image_names[idx])
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    axes[i].imshow(img)

                    # Set title with prediction - shorter titles
                    pred_label = "Soil" if predictions[idx] == 1 else "NotSoil"
                    color = "green" if predictions[idx] == 1 else "red"
                    axes[i].set_title(f"{image_names[idx][:10]}...\n{pred_label}", color=color, fontsize=8)  # Smaller font size
                    axes[i].axis('off')
                else:
                    axes[i].text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center')
                    axes[i].set_title(f"{image_names[idx][:10]}...")
                    axes[i].axis('off')
            except Exception as e:
                axes[i].text(0.5, 0.5, f'Error', ha='center', va='center')
                axes[i].set_title(f"{image_names[idx][:10]}...")
                axes[i].axis('off')

        # Turn off any unused subplots
        for i in range(len(indices), len(axes)):
            axes[i].axis('off')

        plt.tight_layout(pad=0.5)  # Reduce padding between subplots
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f" Sample predictions saved to {save_path}")

    def save_model(self, filepath="soil_classification_model.pkl"):
        """
        Save the trained model to a file using pickle.

        Args:
            filepath (str): Path where the model will be saved.
        """
        import pickle
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_extractor': self.feature_extractor
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f" Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath="soil_classification_model.pkl"):
        """
        Load a trained model from a file.

        Args:
            filepath (str): Path to the saved model file.
        Returns:
            SoilClassificationModel: A loaded model instance.
        """
        import pickle
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        instance = cls()
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_extractor = model_data['feature_extractor']
        print(f" Model loaded from {filepath}")
        return instance


# Main execution
if __name__ == "__main__":
    # Example usage paths - update these to your actual paths
    train_folder = "/kaggle/input/soil-classification-part-2/soil_competition-2025/train"
    test_folder = "/kaggle/input/soil-classification-part-2/soil_competition-2025/test"
    train_csv = "/kaggle/input/soil-classification-part-2/soil_competition-2025/train_labels.csv"
    test_csv = "/kaggle/input/soil-classification-part-2/soil_competition-2025/test_ids.csv"

    try:
        # Initialize and train the model
        model = SoilClassificationModel(nu=0.1)  # Adjust parameters as needed

        # Train and evaluate
        trained_model, validation_metrics = model.train_and_evaluate_model(
            train_folder=train_folder,
            test_folder=test_folder,
            train_csv=train_csv,
            test_csv=test_csv,
            validation_split=0.2
        )

        # Save the trained model
        model.save_model("soil_classification_model.pkl")

        print("\n TRAINING AND EVALUATION COMPLETED SUCCESSFULLY!")
        print("Check the generated plots and submission.csv file for results.")

    except Exception as e:
        print(f" Error: {e}")
        print("Please check that the folder paths exist and contain valid images.")
