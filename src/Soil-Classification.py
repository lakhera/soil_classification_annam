# -------------------------
# Soil Image Classification Challenge
# The Soil Image Classification Challenge is a machine learning competition organised by Annam.ai at IIT Ropar, serving as an initial task for shortlisted hackathon participants. Competitors will build models to classify each soil image into one of four categories: Alluvial soil, Black soil, Clay soil, or Red soil. Final submissions are due by May 25, 2025, 11:59 PM IST. Be sure to submit well before the deadline to avoid server overload or last-minute issues.
# Task: Classify each provided soil image into one of the four soil types (Alluvial, Black, Clay, Red).
# Deadline: May 25, 2025, 11:59 PM IST
# Team Name: RootCoders (Amit Lakhera, Vikramjeet, Jyoti Ghungru, Pradipta Das, Sukanya Saha)
# Last Modified: May 25, 2025
# -------------------------

# Import Required Libraries
import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# PyTorch Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

# Sklearn for splitting and label encoding
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load and Prepare Data
train_df = pd.read_csv('/kaggle/input/soil-classification/soil_classification-2025/train_labels.csv')
train_df['image'] = train_df['image_id']
train_df['label'] = train_df['soil_type']

le = LabelEncoder()
train_df['label_encoded'] = le.fit_transform(train_df['label'])

train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['label_encoded'], random_state=42)

# Image Transformations
img_size = 224
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

val_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Custom Dataset Classes
class SoilDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, 'image']
        label = self.df.loc[idx, 'label_encoded']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

train_img_dir = '/kaggle/input/soil-classification/soil_classification-2025/train/'
train_dataset = SoilDataset(train_df, train_img_dir, transform=train_transforms)
val_dataset = SoilDataset(val_df, train_img_dir, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

# Model Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 4)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Evaluation Function
def evaluate_model(model, data_loader, criterion=None):
    model.eval()
    correct, total, val_loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if criterion:
                loss = criterion(outputs, labels)
                val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    avg_loss = val_loss / len(data_loader) if criterion else None
    return acc, avg_loss


# Training Loop
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
    best_acc = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = correct / total
        val_acc, val_loss = evaluate_model(model, val_loader, criterion)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.state_dict()

        print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Loss: {running_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Acc: {val_acc:.4f}")


    torch.save(best_model_state, 'best_model.pth')
    print("Best model saved with validation accuracy:", best_acc)

# Train the Model
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=15)

# Load Test Data and Create Submission
test_df = pd.read_csv('/kaggle/input/soil-classification/soil_classification-2025/test_ids.csv')
test_df['image'] = test_df['image_id']
test_img_dir = '/kaggle/input/soil-classification/soil_classification-2025/test/'

# Define class TestSoilDataset
class TestSoilDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, 'image']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_name

test_dataset = TestSoilDataset(test_df, test_img_dir, transform=val_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Generate Predictions
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
predictions = []

with torch.no_grad():
    for images, image_names in tqdm(test_loader, desc="Validation"):
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(zip(image_names, predicted.cpu().numpy()))

submission_df = pd.DataFrame(predictions, columns=['image_id', 'label_encoded'])
submission_df['soil_type'] = le.inverse_transform(submission_df['label_encoded'])
submission_df = submission_df[['image_id', 'soil_type']]
submission_df.to_csv('submission.csv', index=False)

print(submission_df)