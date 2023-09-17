import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Load credentials from JSON file
creds = ServiceAccountCredentials.from_json_keyfile_name(
    r'C:\Users\Shady\Downloads\booming-order-399315-4aebce2babfd.json',  # Replace with your credentials file path
    ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
)

# Authorize with Google Sheets API
client = gspread.authorize(creds)

# Open the Google Sheet
sheet = client.open('Database')  # Replace with your actual sheet name

# Get the default (first) sheet
worksheet = sheet.get_worksheet(0)

# Example: Writing data to the sheet
data = [['Username', 'Password'],
        ['user1', 'password1'],
        ['user2', 'password2']]

worksheet.insert_rows(data, 2)  # Insert data starting from the 2nd row

# Example: Reading data from the sheet
all_data = worksheet.get_all_values()
print(all_data)
this code worked, can you rewrite the next code to work as the previous
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import gspread

from oauth2client.service_account import ServiceAccountCredentials

# Load credentials from JSON file
creds = ServiceAccountCredentials.from_json_keyfile_name(
    'booming-order-399315-4aebce2babfd.json',  # Replace with your credentials file path
    ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
)

# Authorize with Google Sheets API
client = gspread.authorize(creds)

# Open the Google Sheet
sheet = client.open('Database')  # Replace with your actual sheet name


# Function to add username to the database in the Google Sheet
def add_username_to_sheet(username, password):
    sheet = client.open('Database')
    worksheet = sheet.get_worksheet(0)  # Assuming the data is in the first sheet

    # Get all values from column A (assuming usernames are in column A)
    existing_usernames = worksheet.col_values(1)

    # Check if username already exists
    if username in existing_usernames:
        return "Username already exists. Please choose a different username."
    else:
        # Append the new username and password
        new_row = [username, password]
        worksheet.append_row(new_row)
        return "Signup successful! You can now log in."


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
class_labels = ['Lemon Canker', 'Nutrient Deficiency', 'Healthy Leaf', 'Multiple Diseases', 'Young & Healthy']

# Initialize session state
if 'user_database' not in st.session_state:
    st.session_state.user_database = {'user1': 'password1', 'user2': 'password2'}

# Initialize login state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Streamlit UI
page = st.sidebar.selectbox("Select Page", ["Login", "Signup"])

if page == "Login":
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in st.session_state.user_database and st.session_state.user_database[username] == password:
            st.session_state.logged_in = True
            st.success("Logged in as {}".format(username))

elif page == "Signup":
    st.title("Signup Page")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")

    if st.button("Signup"):
        if new_username and new_password:
            st.session_state.user_database[new_username] = new_password
            st.success("Signup successful! You can now log in.")
        else:
            st.error("Please provide a username and password")

if st.session_state.logged_in:
    st.title("Lemon Disease Classification")
    st.write("Upload an image to classify it into one of the following classes:")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:


        # Preprocess the input image
        input_image = preprocess_image(Image.open(uploaded_image))

        # Make predictions using the loaded model
        predictions = model.predict(input_image)

        # Get the predicted class index
        predicted_class_index = np.argmax(predictions)
        predicted_class_label = class_labels[predicted_class_index]

        # Display the predicted class label
        st.write("Predicted class label:", predicted_class_label)
        # Display the uploaded image
        st.markdown(f'<h1 style="color:#33ff33;font-size:24px;text-align:center;">{predicted_class_label}</h1>', unsafe_allow_html=True)
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
