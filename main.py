import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.utils import load_img, img_to_array  # type: ignore
from PIL import Image
import io

# Load your pre-trained model
model = load_model('CNN_plantdiseases_model.keras')

# Define class labels
class_labels = [
    "Apple Scab Leaf", "Apple Black Rot Leaf", "Apple Cedar Apple Rust Leaf", "Apple Healthy Leaf",
    "Blueberry Healthy Leaf", "Cherry Powdery Mildew Leaf", "Cherry Healthy Leaf",
    "Corn Cercospora Leaf Spot (Gray Leaf Spot)", "Corn Common Rust Leaf", "Corn Northern Leaf Blight Leaf",
    "Corn Healthy Leaf", "Grape Black Rot Leaf", "Grape Esca (Black Measles) Leaf",
    "Grape Leaf Blight (Isariopsis Leaf Spot)", "Grape Healthy Leaf", "Orange Huanglongbing (Citrus Greening) Leaf",
    "Peach Bacterial Spot Leaf", "Peach Healthy Leaf", "Bell Pepper Bacterial Spot Leaf", "Bell Pepper Healthy Leaf",
    "Potato Early Blight Leaf", "Potato Late Blight Leaf", "Potato Healthy Leaf", "Raspberry Healthy Leaf",
    "Soybean Healthy Leaf", "Squash Powdery Mildew Leaf", "Strawberry Leaf Scorch", "Strawberry Healthy Leaf",
    "Tomato Bacterial Spot Leaf", "Tomato Early Blight Leaf", "Tomato Late Blight Leaf", "Tomato Leaf Mold",
    "Tomato Septoria Leaf Spot", "Tomato Spider Mites (Two-Spotted Spider Mite) Leaf", "Tomato Target Spot Leaf",
    "Tomato Yellow Leaf Curl Virus Leaf", "Tomato Mosaic Virus Leaf", "Tomato Healthy Leaf"
]

# Preprocess the uploaded image
def preprocess_image(image):
    img = load_img(image, target_size=(224, 224))  # Load and resize the image
    img = img_to_array(img) / 255.0  # Convert to array and normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function for prediction with confidence
def predict_image(img):
    img = preprocess_image(img)
    prediction = model.predict(img)
    predicted_class_idx = np.argmax(prediction, axis=1)[0]
    predicted_class = class_labels[predicted_class_idx]
    confidence = np.max(prediction) * 100  # Confidence percentage
    return predicted_class, confidence

# Main Page with Custom Colors
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🌱 Plant Disease Detection System for Sustainable Agriculture</h1>", 
            unsafe_allow_html=True)

image_path = "Diseases.jpg"  # Replace with your image file path or URL
img = Image.open(image_path)
st.image(img, use_container_width=True)

# Using st.markdown for header with colors
st.markdown("<h2 style='text-align: center; color: #FF6347;'>Upload a Leaf image for Disease Recognition</h2>", 
            unsafe_allow_html=True)

# Create a placeholder for dynamic elements
placeholder = st.empty()

# Add a unique key to the file uploader to reset it
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="image_uploader")

if uploaded_file is not None:
    # Create a layout with three columns
    col1, col2, col3 = st.columns([1, 2, 1])  # Adjusting column sizes

    with col2:
        # Display the uploaded image with a controlled width (adjust as needed)
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=False, width=400)  # Adjust the width here

        # Predict Button (centered horizontally in column2)
        predict_button = st.button("Predict")
        
    # Add Snowy Effect with custom CSS
    snowy_css = """
    <style>
    @keyframes snow {
      0% { top: -10px; }
      100% { top: 100%; }
    }
    .snow {
      position: absolute;
      top: -10px;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: 9999;
    }
    </style>
"""
    st.markdown(snowy_css, unsafe_allow_html=True)

    if predict_button:
        with st.spinner("Analyzing the image..."):
            predicted_class, confidence = predict_image(uploaded_file)
            # Display prediction and confidence below the image preview with custom colors
            st.markdown(f"<h3 style='color: #4CAF50;'>Prediction: {predicted_class}</h3>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='color: #FFD700; font-size: 18px;'><b>Confidence:</b> {confidence:.2f}%</h4>", 
                        unsafe_allow_html=True)
            st.snow()