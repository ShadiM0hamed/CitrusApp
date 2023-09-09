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

# User Database (For demonstration purposes only)
user_database = {'user1': 'password1', 'user2': 'password2'}

# Streamlit UI
page = st.sidebar.selectbox("Select Page", ["Login", "Signup"])

if page == "Login":
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in user_database and user_database[username] == password:
            st.success("Logged in as {}".format(username))
            st.title("Citrus Disease Classification")
            st.write("Upload an image to classify it into one of the following classes:")
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
        else:
            st.error("Invalid username or password")

elif page == "Signup":
    st.title("Signup Page")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")

    if st.button("Signup"):
        if new_username and new_password:
            user_database[new_username] = new_password
            st.success("Signup successful! You can now log in.")
        else:
            st.error("Please provide a username and password")

