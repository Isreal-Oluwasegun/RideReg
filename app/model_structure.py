import torch.nn as nn
import torch
import gdown
import os

height = 224
width = 224

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(3 * height * width, 512),
    nn.ReLU(),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Linear(128, 2)
)
MODEL_PATH = "model1.pth" 
GOOGLE_DRIVE_FILE_ID = "1S7lWfjhtH0YBdAMB2iW2u6Z_bkfFtwXc"  

def download_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    else:
        print("Model already exists locally.")


download_model()
model.load_state_dict(torch.load('model1.pth', map_location=torch.device('cpu')))
model.eval()
