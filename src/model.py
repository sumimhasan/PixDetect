import torch.nn as nn
import torchvision.models as models

def get_model(num_classes, pretrained=True):
    model = models.resnet50(pretrained=pretrained)
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model
