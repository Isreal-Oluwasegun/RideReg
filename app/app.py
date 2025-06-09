import streamlit as st
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from model_structure import model



CLASS_NAMES = ["Bus", "Cng"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Match model input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize values to expected range
])



st.title("Bus vs. Napep(Tricycle) Image Classifier")
st.write("Upload an image, and I'll predict whether it's a Bus or a Tricycle (Napep)!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=250)

    
    input_tensor = transform(image).unsqueeze(0)

    
    with torch.no_grad():
        output = model(input_tensor)
        confidence_scores = torch.softmax(output, dim=1)  # Normalize outputs to confidence values
        predicted_class = torch.argmax(output, dim=1).item()
        predicted_name = CLASS_NAMES[predicted_class]  # Match class index with label
        confidence = confidence_scores[0][predicted_class].item()

    # Introduce an "Unknown" category for low-confidence predictions
    if confidence < 0.7:  # Adjust threshold as needed
        predicted_name = "Unknown"

    st.write(f"**Predicted Vehicle:** {predicted_name} (Confidence: {confidence:.2f})")
