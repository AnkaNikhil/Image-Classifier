import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# Load model
model = load_model('cifar10_image_classifier.h5')

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

st.title("CIFAR-10 Image Classifier")
st.write("Drag and drop an image to classify it.")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((32, 32))
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess
    img = img_to_array(image)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Prediction
    prediction = model.predict(img)
    class_idx = np.argmax(prediction)
    confidence = prediction[0][class_idx]
    
    st.write(f"Prediction: **{class_names[class_idx]}**")
    st.write(f"Confidence: {confidence:.2f}")
