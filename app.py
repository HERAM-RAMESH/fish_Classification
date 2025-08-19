import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("model_efficientnetv2b0.keras")

# Class labels (replace with your dataset class names in order)
class_names = ['class1', 'class2', 'class3', 'class4', 'class5',
               'class6', 'class7', 'class8', 'class9', 'class10', 'class11']

st.title("🐟 Fish Species Classification (EfficientNetV2B0)")
st.write("Upload a fish image and the model will predict its species.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = image.resize((224,224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    pred_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    st.success(f"Prediction: **{class_names[pred_class]}** ({confidence*100:.2f}% confidence)")
