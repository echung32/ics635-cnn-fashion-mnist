#!/usr/bin/env python
"""
Assignment 4: Convolutional Neural Networks (CNN) for FashionMNIST
This script implements a baseline CNN and an improved CNN (with dropout and batch normalization)
to classify images from the FashionMNIST dataset using PyTorch.

Usage:
    python assignment4.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix
import torchsummary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

transform = transforms.Compose([
    transforms.ToTensor() # this normalizes to [0, 1] and reshapes the data if necessary
])

full_train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

print("Shape of dataset:", full_train_dataset.data.shape)
print("Labels in the dataset:", full_train_dataset.classes)
print("Number of training samples:", len(full_train_dataset))
print("Number of test samples:", len(test_dataset))

train_dataset, val_dataset = random_split(full_train_dataset, [50000, 10000])

print("Number of training samples after split:", len(train_dataset))
print("Number of validation samples after split:", len(val_dataset))

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class CNNBaseline(nn.Module):
    def __init__(self):
        super(CNNBaseline, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1   = nn.Linear(64 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, 10)
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNNImproved(nn.Module):
    def __init__(self):
        # adds dropout and batch normalization
        # all other parameters are the same as the baseline
        super(CNNImproved, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10):
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        print(f'Epoch {epoch+1}/{num_epochs}: Training Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}')
    return train_losses, val_losses

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = correct / total
    return accuracy, all_preds, all_labels

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', filename='confusion_matrix.png'):
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(filename)

def plot_loss_curves(train_losses, val_losses, title='Loss Curves', filename='loss_curves.png'):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8,6))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.savefig(filename)

#------------------------------------
# Main 
#------------------------------------

def main():
    num_epochs = 15

    model_baseline = CNNBaseline().to(device)
    model_improved = CNNImproved().to(device)

    print("Baseline Model:")
    torchsummary.summary(model_baseline, (1, 28, 28))
    print("Improved Model:")
    torchsummary.summary(model_improved, (1, 28, 28))

    print("Training Baseline CNN Model...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model_baseline.parameters(), lr=0.001)

    baseline_train_losses, baseline_val_losses = train_model(model_baseline, criterion, optimizer, train_loader, val_loader, num_epochs)
    plot_loss_curves(baseline_train_losses, baseline_val_losses, title='Baseline Model Loss Curves', filename='baseline_loss_curves.png')
    print("Baseline Model Loss Curves saved as 'baseline_loss_curves.png'")
    accuracy_baseline, preds_baseline, labels_baseline = evaluate_model(model_baseline, test_loader)
    print(f'Baseline Test Accuracy: {accuracy_baseline * 100:.2f}%')

    classes = full_train_dataset.classes
    cm = confusion_matrix(labels_baseline, preds_baseline)
    plot_confusion_matrix(cm, classes, title='Baseline Model Confusion Matrix', filename='baseline_confusion_matrix.png')
    print("Confusion matrix saved as 'baseline_confusion_matrix.png'")

    print("\nTraining Improved CNN Model with Dropout and Batch Normalization...")
    criterion_improved = nn.CrossEntropyLoss()
    optimizer_improved = optim.AdamW(model_improved.parameters(), lr=0.001)

    improved_train_losses, improved_val_losses = train_model(model_improved, criterion_improved, optimizer_improved, train_loader, val_loader, num_epochs)
    plot_loss_curves(improved_train_losses, improved_val_losses, title='Improved Model Loss Curves', filename='improved_loss_curves.png')
    print("Improved Model Loss Curves saved as 'improved_loss_curves.png'")
    accuracy_improved, preds_improved, labels_improved = evaluate_model(model_improved, test_loader)
    print(f'Improved Model Test Accuracy: {accuracy_improved * 100:.2f}%')

    cm_improved = confusion_matrix(labels_improved, preds_improved)
    plot_confusion_matrix(cm_improved, classes, title='Improved Model Confusion Matrix', filename='improved_confusion_matrix.png')
    print("Confusion matrix saved as 'improved_confusion_matrix.png'")

if __name__ == "__main__":
    main()