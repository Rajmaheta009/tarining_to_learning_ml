import os
import time
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from tqdm import tqdm

# ========== Logging Setup ==========
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/train_log.log",
    filemode='w',
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

# ========== Device Setup ==========
device = torch.device("cpu")
logging.info(f"Using device: {device}")

# ========== Data Load ==========
def get_data_loaders(data_dir, batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    class_names = train_dataset.classes
    logging.info(f"Classes found: {class_names}")
    return train_loader, test_loader, len(class_names)

# ========== Model Build ==========
def build_model(num_classes):
    model = resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

# ========== Train ==========
def train(model, train_loader, val_loader, num_epochs=10, lr=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        avg_train_loss = running_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}%")

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_acc = 100 * correct / total
        avg_val_loss = val_loss / len(val_loader)
        logging.info(f"Epoch {epoch+1}: Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_car_model_cpu.pth")
            logging.info(f"✅ Saved new best model with Val Acc={val_acc:.2f}%")

    logging.info("Training complete.")

# ========== Main ==========
def main():
    logging.info(f"=== Job started: {datetime.now()} ===")
    base_path = "csv_database/Cars Dataset"
    train_loader, val_loader, num_classes = get_data_loaders(base_path)
    model = build_model(num_classes)
    train(model, train_loader, val_loader)
    logging.info(f"=== Job finished: {datetime.now()} ===")

    print("💾 Saving model to 'pkl_files/car_image_model.pth'...")
    os.makedirs("pkl_files", exist_ok=True)
    torch.save(model.state_dict(), "pkl_files/car_image_model2.pth")
    print("✅ Model saved successfully.")

if __name__ == "__main__":
    main()
