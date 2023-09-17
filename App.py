import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Load credentials from JSON file
creds = ServiceAccountCredentials.from_json_keyfile_name(
    'booming-order-399315-4aebce2babfd.json',
    ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
)

# Authorize with Google Sheets API
client = gspread.authorize(creds)

# Open the Google Sheet
sheet = client.open('Database')


# Function to add username to the database in the Google Sheet
def add_username_to_sheet(username, password):
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

# Function to get the user database
def get_user_database():
    return {'user1': 'password1', 'user2': 'password2'}

# Initialize login state
def initialize_login_state():
    return {'logged_in': False, 'username': None}

# Define a cache function to store the database
@st.cache_data()
def get_database():
    return {'users': get_user_database(), 'login_state': initialize_login_state()}

# Function to check if the username and password match
def check_credentials(username, password):
    database = get_database()  # Retrieve the database
    return username in database['users'] and database['users'][username] == password

# Streamlit UI
page = st.sidebar.selectbox("Select Page", ["Login", "Signup"])

if page == "Login":
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        database = get_database()  # Retrieve the database
        if check_credentials(username, password):
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.success("Logged in as {}".format(username))
