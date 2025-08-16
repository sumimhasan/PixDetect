import torch
from src.data_loader import get_dataloaders
from src.model import get_model
from src.train import train_model
from src.utils import save_model

# Configuration
DATA_DIR = "data/train"
BATCH_SIZE = 32
EPOCHS = 5
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "checkpoints/nudity_resnet18.pth"

# Load data
train_loader, val_loader = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)

# Get model
model = get_model(num_classes=3)

# Train model
model = train_model(model, train_loader, val_loader, DEVICE, epochs=EPOCHS, lr=LR)

# Save model
save_model(model, SAVE_PATH)
print(f"Model saved to {SAVE_PATH}")
