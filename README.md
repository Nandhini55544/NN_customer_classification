# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

<img width="898" height="917" alt="image" src="https://github.com/user-attachments/assets/4f917051-bbc9-4eed-812b-0608b3720e66" />


## DESIGN STEPS

### STEP 1: Data Collection and Understanding
Collect customer data from the existing market and identify the features that influence customer segmentation. Define the target variable as the customer segment (A, B, C, or D).

### STEP 2: Data Preprocessing
Remove irrelevant attributes, handle missing values, and encode categorical variables into numerical form. Split the dataset into training and testing sets.

### STEP 3: Model Design and Training
Design a neural network classification model with suitable input, hidden, and output layers. Train the model using the training data to learn patterns for customer segmentation.

### STEP 4: Model Evaluation and Prediction
Evaluate the trained model using test data and use it to predict the customer segment for new customers in the target market.


## PROGRAM

### Name: Nandhini M
### Register Number: 212224040211

```python
!pip install pandas scikit-learn openpyxl matplotlib seaborn torch -q

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from google.colab import files

uploaded = files.upload()

df = pd.read_excel("customers.xlsx")

print(df.head())
print("\nColumns:\n", df.columns)

df = df.drop(columns=["ID"])

for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

X = df.drop("Segmentation", axis=1).values
y = df["Segmentation"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test  = torch.tensor(y_test, dtype=torch.long)

class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

model = PeopleClassifier(X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, preds = torch.max(outputs, 1)

labels = np.unique(y_test)
cm = confusion_matrix(y_test, preds, labels=labels)

print("\nNAME: NANDHINI M")
print("REG NO: 212224040211")

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=labels,
            yticklabels=labels)

plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")

plt.text(-0.5, len(labels)+0.7,
         "NANDHINI M\n212224040211",
         fontsize=10)

plt.show()

print("\nNAME: NANDHINI M")
print("REG NO: 212224040211\n")

print("Classification Report:\n")
print(classification_report(y_test, preds))

sample_index = 0
sample = X_test[sample_index].unsqueeze(0)

with torch.no_grad():
    prediction = model(sample)
    predicted_class = torch.argmax(prediction, 1).item()
    actual_class = y_test[sample_index].item()

print("\nNew Sample Data Prediction")
print("NAME: NANDHINI M")
print("REG NO: 212224040211")
print(f"Predicted class for sample input: {predicted_class}")
print(f"Actual class for sample input: {actual_class}")

```

## Dataset Information

<img width="908" height="484" alt="image" src="https://github.com/user-attachments/assets/bed652b0-fd4e-4df4-95cf-09096ae46423" />

## OUTPUT

<img width="729" height="509" alt="image" src="https://github.com/user-attachments/assets/c68cd152-587b-465e-9def-ed28c2d21d5f" />


### Confusion Matrix

<img width="565" height="562" alt="image" src="https://github.com/user-attachments/assets/8371f11d-4f09-408f-aa4c-5eeae6abb6ef" />


### Classification Report

<img width="447" height="264" alt="image" src="https://github.com/user-attachments/assets/204fb028-02b2-40f4-b620-4a29965fe58e" />


### New Sample Data Prediction

<img width="290" height="88" alt="image" src="https://github.com/user-attachments/assets/32e5db1a-f2f4-425b-8a11-4f4a62114b89" />


## RESULT

Thus neural network classification model is developded for the given dataset. 
