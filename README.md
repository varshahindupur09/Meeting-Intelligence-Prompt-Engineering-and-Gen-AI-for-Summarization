# MeetingIntelligenceGenAI


The backend is designed to handle two main tasks:

Transcribing an Audio File: Convert audio files into text using the Whisper model (via OpenAI's API).
Handling Questions on the Transcription: Store the transcription and allow users to ask questions based on the stored transcription, leveraging Pinecone for vector storage and LangChain for generating answers.

Running locally frontend:
(.venv-llm) varshahindupur@Varshas-MacBook-Air MeetingIntelligenceGenAI % streamlit run frontend/src/app.py 


Running locally backend:
(base) varshahindupur@Varshas-MacBook-Air backend % uvicorn routespath:app --reload

uvicorn trail_openai:app --reload

Commands:
pip3.12 freeze > requirements.txt 
Successfully installed openai-0.28.0
pip3.12 install openai==0.28.0
pip3.12 install -U langchain-openai
pip3.12 install --upgrade openai
pip3.12 install whisper-openai
pip3.12 install python-dotenv
pip3.12 install python-multipart
pip3.12 install boto3
pip3.12 install pymongo

conda deactivate
source .venv_llm/bin/activate
cd backend