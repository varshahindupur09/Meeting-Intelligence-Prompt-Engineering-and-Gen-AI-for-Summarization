# MeetingIntelligenceGenAI

Project by: Varsha Hindupur (NUID: 002935078) & Raunak Balchandani (NUID: 002964444)

Meeting Intelligence GenAI is an application designed to take input in the form of transcripts or audio from any meeting and allow users to ask questions about the meeting. The application uses a combination of transcription services, embeddings, and a language model to generate accurate and relevant responses based on the meeting's context.

# Table of Contents

1. [Features](#Features)
2. [Installation](#Installation)
3. [Setup](#Setup)
4. [RunTheApp](#RunTheApp)
5. [Usage](#Usage)
6. [Troubleshooting](#Troubleshooting)
7. [Dependencies](#Dependencies)
8. [Contributing](#Contributing)
9. [License](#License)
10. [ContactUs](#ContactUs)
11. [YouTube](#YouTube)
    

# Features

Audio Transcription: Automatically transcribe audio files into text using OpenAI's Whisper model. Contextual Q&A: Ask questions related to the meeting, and the app will generate answers based on the provided transcription. Storage Integration: Store and retrieve transcriptions in/from MongoDB for future reference. Vector Search: Use Pinecone for vector search to retrieve the most relevant meeting contex
Installation

# Installation

1. Clone the Repository: git clone https://github.com/yourusername/MeetingIntelligenceGenAI.git cd MeetingIntelligenceGenAI
2. Create a Virtual Environment: python3 -m venv .venv_llm source .venv_llm/bin/activate
3. Install the Required Libraries: pip install -r requirements.txt

# Setup

1. Create a .env file in the root directory and add the following environment variables:
OPENAI_API_KEY=your_openai_api_key PINECONE_API_KEY=your_pinecone_api_key PINECONE_ENVIRONMENT=your_pinecone_environment PINECONE_INDEX_NAME=your_pinecone_index_name MONGO_URI=your_mongodb_connection_string
2. Pinecone Setup: Ensure you have set up a Pinecone index with the correct name and have your API key and environment ready.
3. MongoDB Setup: Set up a MongoDB instance where the transcriptions will be stored. Use the connection string in the .env file.

# RunTheApp

1. Start the Application: Run the application ```strealit run file-name.py```
2. Start the Application: Run the application using Uvicorn: uvicorn codes:app --reload This will start the FastAPI server on http://127.0.0.1:8000.

# Usage

1. Transcribe Audio : To transcribe an audio file, send a POST request to /transcribe/ with the audio file: curl -X POST "http://127.0.0.1:8000/transcribe/" -F "file=@/path/to/your/audio/file.wav"
2. Ask Questions: To ask a question based on a transcribed meeting, send a POST request to /ask_from_audio/: curl -X POST "http://127.0.0.1:8000/ask_from_audio/" -F "file_name=your_audio_file_name.wav" -F "question=Your question here"

# Troubleshooting

1. Embeddings Not Reaching Pinecone: Ensure that your text inputs to the embedding function are strings. Validate inputs in the create_embeddings function.
2. Transcription Issues: Double-check the accuracy of the transcription and verify the API key and connection settings.
3. Relevant Results Not Returned: Review the prompt construction and the embeddings used for vector search. Fine-tuning may be required.

# Dependencies

1. Python 3.8+
2. OpenAI - Used for transcription and language models
3. Pinecone - Vector database for semantic search
4. LangChain - For chaining components together
5. MongoDB - Database for storing transcriptions
6. dotenv - For managing environment variables

# Contributing
Contributions are welcome! Please submit a pull request or create an issue to discuss any changes.

# License
This project is licensed under the MIT License. See the LICENSE file for more details.

# ContactUs:

1. Varsha Hindupur:
   Email: hindupur.v@northeastern.edu, GitHub: https://github.com/varshahindupur09, LinkedIn: https://www.linkedin.com/in/varsha-hindupur/

2. Raunak Balchandani:
   Email: balchandani.r@northeastern.edu, GitHub: https://github.com/raunakbalchandani, LinkedIn: https://www.linkedin.com/in/raunak-balchandani-7090b1128/

# YouTube:

# PDF: https://gamma.app/docs/Meeting-Gen-AI-Summarizing-and-Answering-Questions-4q52maqtrsgv6ud



   
   
