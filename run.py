import torch
from PIL import Image
from torchvision import transforms
from model import MyModel  # or your actual model class

def main():
    # Initialize the model
    model = torch.load('model.pth')
    model.eval()

    # Load an image with PIL
    img_path = "test_image.jpg"  # replace with your image path
    image = Image.open(img_path).convert("RGB")  # ensure 3 channels

    # Transform the image to tensor and resize
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),   # CNN input size
        transforms.ToTensor(),           # convert to [0,1] tensor
    ])
    input_tensor = preprocess(image).unsqueeze(0)  # add batch dimension

    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)

    print("Model output:", output)

if __name__ == "__main__":
    main()
