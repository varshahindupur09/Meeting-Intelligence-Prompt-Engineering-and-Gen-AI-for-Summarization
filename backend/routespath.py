from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import os
from openai import OpenAI
import uvicorn
from dotenv import load_dotenv
from pinecone import Pinecone as client
from langchain_community.vectorstores import Pinecone
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import JinaEmbeddings
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda
import openai
from pymongo import MongoClient
import logging

logging.basicConfig(level=logging.INFO)

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

# Jina Embeddings
jina_api_key = os.getenv("JINA_EMBEDDINGS_API_KEY")

pc = client(
    api_key=pinecone_api_key,
    environment=pinecone_environment
)

# Access the index
index = pc.Index(pinecone_index_name)
index_stats = index.describe_index_stats()
logging.info(f"Pinecone Index: {index_stats}")

mongo_uri = os.getenv("MONGO_URI")
client_mongo = MongoClient(mongo_uri)
db_mongo = client_mongo["storing_transcriptions"]
collection = db_mongo["transcriptions"]

# Setup LangChain components
# embeddings = OpenAIEmbeddings(deployment="text-similarity-ada-001")
embeddings = JinaEmbeddings(jina_api_key=jina_api_key, model_name='jina-clip-v1')
vectorstore = Pinecone.from_existing_index(index_name=pinecone_index_name, embedding=embeddings)

# def create_prompt(context, question):
template = """Answer the question based only on the following context:{context} Question: {question}"""
prompt = ChatPromptTemplate.from_template(template)
# return prompt
logging.info(f"Generated prompt: {template} and the prompt is: {prompt}")

llm = ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo")
output_parser = StrOutputParser()

# Runnable chain for dynamic retrieval and question answering
def create_retriever(query: str):
    logging.info(f"Query used for retrieval: {query}")
    results = vectorstore.similarity_search(query)
    logging.info(f"Retrieved contexts: {results}")
    logging.info("")
    # all_documents = vectorstore.get_all_documents()  # Replace with the actual method to retrieve all documents
    # logging.info(f"Total documents in vector store: {len(all_documents)}")
    # logging.info("")
    return results

dynamic_retriever = RunnableLambda(lambda inputs: create_retriever(inputs['question']))
logging.info(f"dynamic_retriever: {dynamic_retriever}")

def get_question(inputs):
    return inputs['question']

chain = (
    RunnableParallel({"context": dynamic_retriever, "question": get_question})
    | prompt
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

        logging.info(f"Retrieved transcription: {context}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving transcription: {str(e)}")

    # Run the RAG chain with the retrieved transcription and the question
    inputs = {"context": context, "question": question}
    logging.info(f"Inputs for chain: {inputs}")
    result = chain.invoke(inputs)
    logging.info(f"Generated response: {result}")

    return {"transcription": context, "answer": result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


# Correct the prompt creation and the chain's invocation logic
# chain = (
#     RunnableParallel({"context": dynamic_retriever, "question": get_question})
#     | RunnableLambda(lambda inputs: create_prompt(inputs["context"], inputs["question"]))
#     | RunnableLambda(lambda prompt: [{"role": "user", "content": prompt}])  # Convert the prompt to the format expected by the LLM
#     | llm
#     | output_parser
# )

# RAG prompt
# def create_prompt(context, question):
#     return f"""The following is a transcript of a meeting. Please answer the question using only the information provided in this transcript.
#     Transcript:
#     {context}
#     Question:
#     {question}
#     Please provide a clear and concise answer based on the transcript above."""