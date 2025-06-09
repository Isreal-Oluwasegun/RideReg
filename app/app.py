import streamlit as st
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from model_structure import model



CLASS_NAMES = ["Bus", "Cng"]

import torch
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
])


st.title("Bus vs. Napep(Tricycle) Image Classifier")
st.write("Upload an image, and I'll predict whether it's a Bus or a Tricycle (Napep)!")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=250)

    
    input_tensor = transform(image).unsqueeze(0)

    
    with torch.no_grad():
        output = model(input_tensor)            # One value (logit)
        prob = torch.sigmoid(output).item()    # Converts to probability between 0 and 1
        if prob > 0.85:
            print("Predicted: Tricycle")
        elif prob < 0.15:
            print("Predicted: Bus")
        else:
            print("Unknown or uncertain prediction")




