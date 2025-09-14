import torch
from src.data_loader import get_dataloaders
from src.model import get_model
from src.train import train_model
from src.utils import save_model
import json 

CONFIG_PATH = "training-config.json"
with open(CONFIG_PATH, "r") as f:
    data = json.load(f)

# Configuration
data_dir = data["train_data_path"]
batch_size = data["batch_size"]
epochs = data["num_epochs"]
learning_rate = data["learning_rate"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path = data["save_checkpoints_path"]
validation_split = data["validation_split"]
image_width = data["image_width"]
image_height = data["image_height"]

# Load data
train_loader, val_loader = get_dataloaders(image_width,image_height,data_dir, batch_size=batch_size,val_split=validation_split)

# Get model
model = get_model(num_classes=3)

# Train model
model = train_model(model, train_loader, val_loader, device, epochs=epochs, lr=learning_rate)

# Save model
save_model(model, save_checkpoints_path)
print(f"Model saved to {save_checkpoints_path}")
