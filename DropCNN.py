import os
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# -----------------------------
# Model definition
# -----------------------------
class DropoutCNN(nn.Module):
    def __init__(self, num_classes: int = 7):
        super().__init__()

        self.features = nn.Sequential(
            # Conv block 1 (smaller)
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),

            # Conv block 2 (smaller)
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),

            # Conv block 3 (smaller)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),
        )

        # 48x48 -> 24 -> 12 -> 6, channels = 64, so 64 * 6 * 6 = 2304
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(64 * 6 * 6, 128),  # smaller FC layer
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# -----------------------------
# Training / validation loops
# -----------------------------
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# -----------------------------
# Plotting & confusion matrix
# -----------------------------
def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="Train Acc")
    plt.plot(epochs, val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()


def compute_confusion_matrix(model, dataloader, device, class_names):
    model.eval()
    preds_list = []
    true_list = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            preds_list.extend(preds.cpu().numpy())
            true_list.extend(labels.cpu().numpy())

    cm = confusion_matrix(true_list, preds_list)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=class_names)
    disp.plot(xticks_rotation=45, cmap="Blues")
    plt.title("Confusion Matrix (Validation Set)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()


# -----------------------------
# Main script
# -----------------------------
def main():
  
    DATA_DIR = r"C:\Users\quint\OneDrive\Documents\GitHub\FacialExpressionRecognition\data\fer\images\images"
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "validation")

    batch_size = 64
    num_epochs = 5
    learning_rate = 1e-3
    num_workers = 0  # keep 0 on Windows to avoid multiprocessing issues

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train dir does not exist: {train_dir}")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Validation dir does not exist: {val_dir}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    # Datasets
    train_dataset = datasets.ImageFolder(root=train_dir,
                                         transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir,
                                       transform=val_transform)

    class_names = train_dataset.classes
    num_classes = len(class_names)

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    # Model / loss / optimizer
    model = DropoutCNN(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # For curves
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_acc = 0.0

    # Training loop
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        epoch_time = time.time() - start_time

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch [{epoch}/{num_epochs}] "
            f"- {epoch_time:.1f}s | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_cnn_model.pt")
            print(f"  -> New best model saved (val_acc = {best_val_acc:.4f})")

    print("\nTraining complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # Plots
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)

    # Confusion matrix on validation set
    compute_confusion_matrix(model, val_loader, device, class_names)


# -----------------------------
# Windows-safe entry point
# -----------------------------
if __name__ == "__main__":
    main()
