import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io

app = FastAPI()

# Load model
num_classes = 7  # set to your actual number of car classes
device = torch.device("cpu")
model = resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("../pkl_files/car_image_model.pth", map_location=device))
model.eval()

# Transform for inference
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

class_names = ['audi', 'hyundai', 'Scorpio', 'rolls royal', 'swift', 'tata', 'toyota']  # match your folders

@app.post("/predict_car_info")
async def predict_car_info(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        class_index = int(predicted.item())
        class_name = class_names[class_index]

    return JSONResponse({
        "prediction_index": class_index,
        "prediction_name": class_name
    })


