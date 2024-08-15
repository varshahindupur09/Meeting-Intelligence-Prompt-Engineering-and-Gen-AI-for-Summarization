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

# Set the title and description
# st.title("Meetin: Meeting Intelligence App")
# st.write("Enhance your meeting productivity with advanced audio transcription and AI-driven insights.")

st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50; font-size: 36px;'>
    🚀 Meetin: <br> AI-Powered Meeting Intelligence & Transcription 📋
    </h1>
    <p style='text-align: center; color: #555; font-size: 18px;'>
    Elevate your meetings with cutting-edge transcription and smart insights
    </p>
    """,
    unsafe_allow_html=True
)

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
st.header("Upload Audio File")
uploaded_file = st.file_uploader("Upload your audio file", type=["mp3"])

if uploaded_file is not None:
    # Save the uploaded file to the directory
    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File saved to {file_path}")
    st.success(f"File saved successfully!")

    # Transcribe the audio file by sending it to the FastAPI backend
    with open(file_path, "rb") as f:
        files = {"file": (uploaded_file.name, f, "multipart/form-data")}
        response = requests.post(f"{backend_url}/transcribe/", files=files)

    if response.status_code == 200:
        result = response.json()
        st.subheader("Transcription")
        st.write("Transcription:", result["transcription"])

        # Ask questions based on the transcription
        # Interactive Q&A based on the transcription
        st.subheader("Ask Questions")
        question = st.text_input("Ask a question about the audio:")
        if question:
            response = requests.post(
                f"{backend_url}/ask_from_audio/",
                data={"file_name": uploaded_file.name, "question": question}
            )

            if response.status_code == 200:
                result = response.json()
                st.write("Answer:", result["answer"])
            else:
                st.error("Failed to get an answer from the backend.")
    else:
        st.error("Failed to transcribe the audio file.")


# Footer or additional information
st.markdown("---")
st.write("© 2024 Meetin. Happy to Share!")