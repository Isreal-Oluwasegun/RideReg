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


st.title("Bus vs. Napep (Tricycle) Image Classifier")
st.write("Upload an image, and I'll predict whether it's a Bus or a Tricycle (Napep)!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])


if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=250)

    
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
        predicted_name = CLASS_NAMES[predicted_class]  # Match class index with label

    st.write(f"**Predicted Vehicle:** {predicted_name}")

