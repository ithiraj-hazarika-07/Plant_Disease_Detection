import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.utils import load_img, img_to_array  # type: ignore
from PIL import Image

model = load_model('CNN_plantdiseases_model.keras')

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

# Preprocessing the uploaded image
def preprocess_image(image):
    img = load_img(image, target_size=(224, 224))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Function for prediction with confidence
def predict_image(img):
    img = preprocess_image(img)
    prediction = model.predict(img)
    predicted_class_idx = np.argmax(prediction, axis=1)[0]
    predicted_class = class_labels[predicted_class_idx]
    confidence = np.max(prediction) * 100
    return predicted_class, confidence

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="icons8-trees-64.png",
)

st.markdown("""
    <style>
        [data-testid="stHeaderActionElements"] {
            display: none !important;
        }
        .st-emotion-cache-1cqe6qn, .e1obcldf23 {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🌱 Plant Disease Detection System for Sustainable Agriculture</h1>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

image_path = "Diseases.jpg"
img = Image.open(image_path)
st.image(img, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    "<p style='text-align: center; font-size: 20px;'>This system provides the capability to identify whether a plant is healthy or afflicted with disease. It supports detection for various plant species including <b>Apple, Bell Pepper, Blueberry, Cherry, Corn, Grape, Orange, Peach, Potato, Raspberry, Soybean, Squash, Strawberry,</b> and <b>Tomato</b>.</p>",
    unsafe_allow_html=True
)

placeholder = st.empty()

st.markdown(
    "<h2 style='text-align: center; color: #FF6347; font-weight: bolder'>Upload a Leaf image for Disease Recognition</h2>",
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    "<h3 style='font-size: 24px; text-align: center; font-weight: 900;'>Choose an Image</h3>", 
    unsafe_allow_html=True
)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="image_uploader")

if uploaded_file is not None:

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
    
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=False, width=400)
        
        predict_button = st.button("Predict")

    if predict_button:
        st.snow()
        with st.spinner("Analyzing the image..."):
            predicted_class, confidence = predict_image(uploaded_file)
        
            st.success("Model is predicting it as a {}".format(predicted_class))
            st.markdown(f"<h3 style='color: #4CAF50;'>Prediction: {predicted_class}</h3>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='color: #FFD700; font-size: 18px;'><b>Confidence:</b> {confidence:.2f}%</h4>", 
                        unsafe_allow_html=True)