import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np

# Load the trained model
model = keras.models.load_model('Citrus_Model.h5', compile=False)

# Function to preprocess the input image
def preprocess_image(image):
    img = image.resize((64, 64))  # Resize the image to (64, 64)
    img = np.array(img)  # Convert the image to numpy array
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Scale to [0, 1]
    return img

# Map the prediction to the corresponding class label
class_labels = ['Citrus_Canker', 'Nutrient_Deficiency', 'Healthy_Leaf_Orange', 'Multiple_Diseases', 'Young_Healthy']

# Streamlit UI
st.title("Citrus Disease Classification")
st.write("Upload an image to classify it into one of the following classes:")

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the input image
    input_image = preprocess_image(Image.open(uploaded_image))

    # Make predictions using the loaded model
    predictions = model.predict(input_image)

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class_index]

    # Display the predicted class label
    st.write("Predicted class label:", predicted_class_label)
