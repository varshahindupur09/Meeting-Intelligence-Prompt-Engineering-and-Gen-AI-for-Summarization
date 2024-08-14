import streamlit as st
import os

cwd = os.getcwd()
print("cwd: ", cwd)

# # Define the directory to store uploaded files relative to the current working directory
upload_dir = os.path.join(cwd, "audio_files")
# upload_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audio_files")
# print("**** upload dir: ", upload_dir)

# Ensure the directory exists
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

# Define the directory to store uploaded files
upload_dir = "./audio_files/"
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

# Streamlit file uploader
uploaded_file = st.file_uploader("Upload your text file", type=["txt"])

if uploaded_file is not None:
    # Save the uploaded file to the directory
    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File saved to {file_path}")
    
    