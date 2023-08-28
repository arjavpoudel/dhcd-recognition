import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io

device = torch.device("mps")
HIDDEN_SIZE = 512
OUT_SIZE = 46
BATCH_SIZE = 128
PARAM_PATH = "dhcd_model.pth"


def get_char_pred(class_label):
    with open("labels.txt") as f:
        labels = [line.strip() for line in f]
    itos = {i: s for i, s in enumerate(labels)}
    return itos[class_label.item()]


model = nn.Sequential(
    nn.Conv2d(1, 8, 3, padding=1),
    nn.BatchNorm2d(8),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(8, 16, 3, padding=1),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(16, 32, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(32, 32, 3, padding=1),
    nn.Conv2d(32, 64, 3, padding=1),
    nn.BatchNorm2d(64),
    nn.Conv2d(64, 64, 3, padding=1),
    nn.Flatten(),
    nn.Linear(64 * 64, 128),
    nn.ReLU(),
    nn.Linear(128, HIDDEN_SIZE),
    nn.ReLU(),
    nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
    nn.ReLU(),
    nn.Linear(HIDDEN_SIZE, OUT_SIZE),
)


def transform_image(image_bytes):
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Resize((32, 32), antialias=True),
        ]
    )
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)


def get_prediction(image_tensor):
    image = image_tensor.reshape(1, 1, 32, 32)
    output = model(image)
    prediction = torch.argmax(output)
    return prediction


model.load_state_dict(torch.load(PARAM_PATH))
