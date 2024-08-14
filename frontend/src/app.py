import streamlit as st
import os
from dotenv import load_dotenv
import requests

env_path = os.path.join(os.getcwd(), ".env")
load_dotenv(dotenv_path=env_path)

# FastAPI backend URL
backend_url = os.getenv("BACKEND_URL")

cwd = os.getcwd()
cwd = cwd.replace('/frontend','')
print("cwd: ", cwd)

# # Define the directory to store uploaded files relative to the current working directory
upload_dir = os.path.join(cwd, "files")

# Ensure the directory exists
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

# Define the directory to store uploaded files
upload_dir = "./files/"
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

# Streamlit file uploader
uploaded_file = st.file_uploader("Upload your audio file", type=["mp3"])

# Processing on uploaded_file
if uploaded_file is not None:
    # Save the uploaded file to the directory
    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File saved to {file_path}")

    # Send the file to the FastAPI backend for processing
    with open(file_path, "rb") as f:
        files = {"file": (uploaded_file.name, f, "multipart/form-data")}
        response = requests.post(f"{backend_url}/ask_from_audio/", files=files, data={"question": "What is the content of this audio?"})

    if response.status_code == 200:
        result = response.json()
        st.write("Transcription:", result["transcription"])
        st.write("Answer:", result["answer"])
    else:
        st.error("Failed to process the audio file.")
    


    