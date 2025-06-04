# server.py
from fastmcp import FastMCP
from pydantic import BaseModel
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import io
import base64

# Initialize MCP
mcp = FastMCP()

# Define car classes
class_names = [
    "audi",
    "hyundai_creta",
    "mahindra_scorpio",
    "rolls_royce",
    "swift",
    "tata_safari",
    "toyota_innova"
]
num_classes = len(class_names)

# Load the trained model
device = torch.device("cpu")
model = resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("../pkl_files/car_image_model.pth", map_location=device))
model.eval()

# Preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Define input schema for MCP
class CarImageInput(BaseModel):
    filename: str
    file_data: str  # base64-encoded image

# Define the tool
@mcp.tool()
def car_model_connecter(input: CarImageInput) -> str:
        # Handle padding issue for base64 string
        file_data = input.file_data
        padding_needed = len(file_data) % 4
        if padding_needed:
            file_data += '=' * (4 - padding_needed)

        # Decode and preprocess image
        image_bytes = base64.b64decode(file_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)

        # Prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            class_index = int(predicted.item())
            class_name = class_names[class_index]

        print(f"Predicted car brand: {class_name}")
        return f"Predicted car brand: {class_name}"

# Run the MCP server
if __name__ == "__main__":
    mcp.run(transport="sse")
