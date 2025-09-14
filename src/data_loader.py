import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(imageW,imageH,data_dir, batch_size, val_split):
    # Transformations
    transform = transforms.Compose([
        transforms.Resize((imageW, imageH)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    # Train/Validation split
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
