from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import os
from openai import OpenAI
import uvicorn
from dotenv import load_dotenv
from pinecone import Pinecone as client
from langchain_community.vectorstores import Pinecone
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda
import openai
from pymongo import MongoClient

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

# RAG prompt
# def create_prompt(context, question):
#     return f"""You are an intelligent meeting assistant. Use the following minutes of the meeting to understand and answer the questions as accurately as possible based on the provided context.
#     Context: {context}
#     Question: {question}
#     Please provide the answer strictly based on the information in the context above.
#     """

def create_prompt(context, question):
    return f"""Answer the question based only on the following context:{context} Question: {question}"""


llm = ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo")
output_parser = StrOutputParser()

# Runnable chain for dynamic retrieval and question answering
def create_retriever(query: str):
    return vectorstore.similarity_search(query)

dynamic_retriever = RunnableLambda(lambda inputs: create_retriever(inputs['question']))

def get_question(inputs):
    return inputs['question']

# Correct the prompt creation and the chain's invocation logic
chain = (
    RunnableParallel({"context": dynamic_retriever, "question": get_question})
    | RunnableLambda(lambda inputs: create_prompt(inputs["context"], inputs["question"]))
    | RunnableLambda(lambda prompt: [{"role": "user", "content": prompt}])  # Convert the prompt to the format expected by the LLM
    | llm
    | output_parser
)

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

    # Run the RAG chain with the retrieved transcription and the question
    inputs = {"context": context, "question": question}
    result = chain.invoke(inputs)

    # Log inputs for debugging
    # print(f"Prompt: {prompt_text}")

    return {"transcription": context, "answer": result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
