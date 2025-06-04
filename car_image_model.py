import os
import logging

import joblib
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet18
from torch.utils.data import DataLoader

# ========== Logging Setup ==========
logging.basicConfig(
    filename='train.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

# ========== Device Setup ==========
device = torch.device("cpu")
logging.info("Using CPU for training.")
logging.info(f"Using device: {device}")

# ========== Data Setup ==========
data_dir = "./csv_database/Cars Dataset"  #Replace with your dataset path
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "test")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

num_classes = len(train_dataset.classes)
logging.info(f"Detected {num_classes} car classes")

# ========== Model Setup ==========
model = resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ========== Training ==========
best_acc = 0.0
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    avg_loss = train_loss / len(train_loader)
    logging.info(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Train Acc={acc:.2f}%")

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

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_car_model_cpu.pth")
        logging.info(f"Saved new best model with Val Acc={val_acc:.2f}%")

logging.info("Training complete.")

print("💾 Saving model to 'pkl_files/fraud.pkl'...")
torch.save(model.state_dict(), "pkl_files/car_image_model.pth")
print("✅ Model saved successfully.")