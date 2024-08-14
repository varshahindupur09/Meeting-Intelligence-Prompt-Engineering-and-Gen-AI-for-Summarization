from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import os
from openai import OpenAI
import uvicorn
from dotenv import load_dotenv
from pinecone import Pinecone as client
from langchain_community.vectorstores import Pinecone
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import whisper
import openai
from pymongo import MongoClient

# Load the Whisper model
model = whisper.load_model("base")  # You can choose from "tiny", "base", "small", "medium", "large"

app = FastAPI()
env_path = os.path.join(os.getcwd(), ".env")
load_dotenv(dotenv_path=env_path)

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

pc = client(
    api_key=pinecone_api_key,
    environment=pinecone_environment
)

# Access the index
index = pc.Index(pinecone_index_name)
index.describe_index_stats()

mongo_uri = os.getenv("MONGO_URI")
client_mongo = MongoClient(mongo_uri)
db_mongo = client_mongo["storing_transcriptions"]
collection = db_mongo["transcriptions"]

# Setup LangChain components
embeddings = OpenAIEmbeddings(deployment="text-similarity-ada-001")
vectorstore = Pinecone.from_existing_index(index_name=pinecone_index_name, embedding=embeddings)

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
qa_chain = load_qa_chain(llm, chain_type="stuff")

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    file_location = f"temp_{file.filename}"

    try:
        # Check if transcription already exists in MongoDB
        transcription_doc = collection.find_one({"filename": file.filename})
        if transcription_doc:
            return {"transcription": transcription_doc["transcription"]}

        # Save the uploaded file temporarily
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())

        # Open the file and transcribe using the OpenAI client
        with open(file_location, "rb") as audio_file:
            transcript = client_openai.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                response_format="verbose_json"
            )

        # Assuming the transcript object has an attribute 'text' that contains the transcription
        transcription_text = transcript.text

        # Save the transcription to MongoDB
        collection.insert_one({
            "filename": file.filename,
            "transcription": transcription_text
        })

        return {"transcription": transcription_text}

    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

    finally:
        if os.path.exists(file_location):
            os.remove(file_location)


@app.post("/ask_from_audio/")
async def ask_from_audio(file_name: str = Form(...), question: str = Form(...)):
    try:
        # Retrieve the transcription from MongoDB using the filename
        transcription_doc = collection.find_one({"filename": file_name})
        if not transcription_doc:
            raise HTTPException(status_code=404, detail="Transcription not found in MongoDB")

        context = transcription_doc["transcription"]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving transcription: {str(e)}")

    # Store transcription in Pinecone (assuming store_in_pinecone and vectorstore are defined)
    store_in_pinecone(context)

    # Perform RAG: Retrieve relevant documents from Pinecone and generate a response
    docs = vectorstore.similarity_search(question, k=5)
    answer = qa_chain.run(input_documents=docs, question=question)

    return {"transcription": context, "answer": answer}

def store_in_pinecone(text):
    # vector = embeddings.embed_query(text)
    # vectorstore.add_texts([text], [vector], [{"metadata": {"text": text}}])
    # Assuming you want to store each piece of text along with its vector and metadata
    vector = embeddings.embed_query(text)
    metadata = {"text": text}  # Metadata should be a dictionary
    # Ensure that both `text` and `vector` are lists, and `metadata` is a list of dictionaries
    vectorstore.add_texts([text], [vector], [metadata])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)