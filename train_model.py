#!/usr/bin/env python3
"""
Model training script for marine acoustic classifier.

Trains MobileNetV2 on spectrogram images and exports to ONNX format.
Per spec: 3 classes (Background, Vessel, Cetacean), input [1, 3, 224, 224].
"""

import os
import sys
from pathlib import Path

# Add src to path for config import
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# --- CONFIGURATION ---
TRAIN_DIR = Path("./data/processed_spectrograms/train")
VAL_DIR = Path("./data/processed_spectrograms/val")
MODEL_OUTPUT = Path("./models/classifier.onnx")
NUM_CLASSES = 3
NUM_EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 0.001
IMAGE_SIZE = 224


def get_data_loaders():
    """Create training and validation data loaders."""
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.3),  # Time reversal augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # No augmentation for validation
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=VAL_DIR, transform=val_transform)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")
    print(f"Class to index: {train_dataset.class_to_idx}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    return train_loader, val_loader, train_dataset.class_to_idx


def create_model():
    """Create MobileNetV2 model modified for 3 classes."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Modify classifier for 3 classes
    # Original: classifier[1] = Linear(1280, 1000)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)

    return model


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = running_loss / len(train_loader)
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = running_loss / len(val_loader)
    return avg_loss, accuracy


def export_to_onnx(model, device):
    """Export trained model to ONNX format."""
    model.eval()

    # Create dummy input matching expected shape: [1, 3, 224, 224]
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)

    # Ensure output directory exists
    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # Export with dynamic batch size
    torch.onnx.export(
        model,
        dummy_input,
        MODEL_OUTPUT,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"Model exported to: {MODEL_OUTPUT}")


def main():
    """Main training loop."""
    print("=" * 60)
    print("Marine Acoustic Classifier - Model Training")
    print("=" * 60)

    # Check for data
    if not TRAIN_DIR.exists() or not any(TRAIN_DIR.iterdir()):
        print(f"ERROR: Training data not found at {TRAIN_DIR}")
        print("Run generate_dummy_data.py first to create training data.")
        sys.exit(1)

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_loader, val_loader, class_to_idx = get_data_loaders()

    # Verify class order matches spec: background=0, cetacean=1, vessel=2
    # ImageFolder sorts alphabetically, so: background, cetacean, vessel
    print(f"\nClass mapping from ImageFolder: {class_to_idx}")
    print("Note: Model outputs will use this ordering")

    # Create model
    model = create_model()
    model = model.to(device)
    print(f"\nModel: MobileNetV2 ({sum(p.numel() for p in model.parameters())} parameters)")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print(f"\nTraining for {NUM_EPOCHS} epochs...")
    print("-" * 60)

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    print("-" * 60)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

    # Export to ONNX
    print("\nExporting model to ONNX format...")
    export_to_onnx(model, device)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
