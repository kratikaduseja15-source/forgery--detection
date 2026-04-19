import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

# Load model
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# Test any image
IMAGE_PATH = r"C:\Users\victus\OneDrive\New folder\forgery project\data\findit2\test\X00016469619.png"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

img = Image.open(IMAGE_PATH).convert("RGB")
tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(tensor)
    prob = torch.softmax(output, dim=1)
    classes = ['forged', 'real']
    predicted = classes[prob.argmax()]
    confidence = prob.max().item() * 100

print(f"Result: {predicted.upper()}")
print(f"Confidence: {confidence:.1f}%")