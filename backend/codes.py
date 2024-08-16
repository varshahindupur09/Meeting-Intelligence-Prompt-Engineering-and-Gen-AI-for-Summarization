from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import os
from openai import OpenAI
import uvicorn
from dotenv import load_dotenv
from pinecone import Pinecone as client
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda
import openai
from pymongo import MongoClient
import logging
from pinecone import ServerlessSpec

app = FastAPI()
logging.basicConfig(level=logging.INFO)

env_path = os.path.join(os.getcwd(), ".env")
load_dotenv(dotenv_path=env_path)

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
client_openai = OpenAI(api_key=openai.api_key)

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME_2")

pc = client(
    api_key=pinecone_api_key,
    environment=pinecone_environment
)

# # Access the index
index = pc.Index(pinecone_index_name)

# MongoDB setup
mongo_uri = os.getenv("MONGO_URI")
client_mongo = MongoClient(mongo_uri)
db_mongo = client_mongo["storing_transcriptions"]
collection = db_mongo["transcriptions"]

# Setup LangChain components
embeddings = OpenAIEmbeddings(deployment="text-similarity-ada-001")
sample_text = "This is a sample text to check dimensions."
embedding_test = embeddings.embed_query(sample_text)
logging.info(f"Dimension of the embedding: {len(embedding_test)}")
embedding_dimension = len(embedding_test)

# Ensure the index is created with the correct dimension
# if index_name not in pc.list_indexes():
# pc.create_index(
#     name=pinecone_index_name,
#     dimension=1536,  # Use the dynamic embedding dimension
#     metric="cosine",
#     spec=ServerlessSpec(
#         cloud='aws', 
#         region=pinecone_environment
#     ) 
# )

vectorstore = Pinecone.from_existing_index(index_name=pinecone_index_name, embedding=embeddings)

# def create_prompt(context, question):
template = """Answer the question based only on the following context:{context} Question: {question}"""
prompt = ChatPromptTemplate.from_template(template)
# return prompt
logging.info(f"Generated prompt: {template} and the prompt is: {prompt}")

llm = ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo")
output_parser = StrOutputParser()

def create_retriever(query: str):
    logging.info(f"query: {query}")
    val = vectorstore.similarity_search(query)
    logging.info(f"vectorstore.similarity_search(query){val}")
    return val

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

def create_prompt(context, question):
    return f"""You are an expert on the following meeting's context: "{context}". Based on this context, answer the following question accurately: "{question}"."""

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    file_location = f"temp_{file.filename}"

    try:
        # Check if transcription already exists in MongoDB
        transcription_doc = collection.find_one({"filename": file.filename})
        if transcription_doc:
            logging.info("MongoDB transcription found!")
            store_transcription_in_pinecone(transcription_doc['transcription'], {"id": file.filename, "filename": file.filename})
            return {"transcription": transcription_doc["transcription"]}

        # Save the uploaded file temporarily
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())

        # Transcribe using the OpenAI client
        with open(file_location, "rb") as audio_file:
            transcript = client_openai.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                response_format="verbose_json"
            )
        logging.info("")
        logging.info("")
        logging.info("***************************")
        logging.info(f"Transcript: {type(transcript)}")
        logging.info(f"Transcript: {transcript}")

        # Check if the transcription was successful
        logging.info("")
        logging.info("")
        logging.info("***************************")
        # transcription_text = transcript['Transcription[text]']
        # transcription_text = transcript['text']
        transcription_text = transcript.text
        logging.info(f"2222 Transcript: {transcription_text}")


        # Save the transcription to MongoDB
        metadata = {"filename": file.filename, "id": file.filename}  # Additional metadata can be added if needed
        collection.insert_one({
            "filename": file.filename,
            "transcription": transcription_text
        })

        # Optionally store in Pinecone if needed
        logging.info("Storing transcription in Pinecone")
        store_transcription_in_pinecone(transcription_text, metadata)

        return {"transcription": transcription_text}

    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

    finally:
        if os.path.exists(file_location):
            os.remove(file_location)


def check_transcription_in_pinecone(filename):
    """
    Checks if a transcription with the given filename exists in Pinecone.
    Assumes that `filename` is stored as a key in the metadata of each indexed item.
    """
    try:
        logging.info(f"filename in check_transcription_in_pinecone: {filename} and {embedding_dimension}")
        # Query Pinecone by trying to fetch vectors where metadata matches the filename
        search_results = index.query(
            vector=[0]*embedding_dimension,  # A dummy vector since we focus on metadata
            filter={"filename": filename},  # Assumes Pinecone supports filtering by metadata
            top_k=1,
            include_metadata=True
        )
        logging.info(f"search_results in check_transcription_in_pinecone  {search_results}")

        # Check if there are any matches
        if search_results['matches']:
            match = search_results['matches'][0]
            if match.metadata['filename'] == filename:
                logging.info(f"Found transcription for filename: {filename}")
                return True
        return False
    
    except Exception as e:
        logging.error(f"Error checking transcription in Pinecone: {str(e)}")
        return False


def store_transcription_in_pinecone(text, metadata):
    embedding = embeddings.embed_query(text)
    metadata["text"] = text  
    logging.info(f"data inserted {metadata}")
    logging.info(f"data inserted {embedding}")
    index.upsert([(metadata["id"], embedding, metadata)])

def retrieve_context_from_pinecone(query):
    query_embedding = embeddings.embed_query(query)
    # logging.info(f"query_embedding {query_embedding}") working

    search_results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
    logging.info(f"search_results in retrieve_context_from_pinecone  {search_results}")

    combined_context = " ".join([result["metadata"]["text"] for result in search_results["matches"]])
    logging.info(f"combined_context in retrieve_context_from_pinecone  {combined_context}")

    return combined_context

@app.post("/ask_from_audio/")
async def ask_from_audio(file_name: str = Form(...), question: str = Form(...)):
    try:
        logging.info(f"file_name: {file_name}")
        logging.info(f"question: {question}")

        # Check transcription availability and retrieve combined context
        if not check_transcription_in_pinecone(file_name):
            raise HTTPException(status_code=404, detail="Transcription not found in Pinecone")
        
        # Retrieve the specific transcription context from Pinecone
        base_context = retrieve_context_from_pinecone(file_name)
        logging.info(f"Base context from Pinecone: {base_context}")

    except Exception as e:
        logging.error(f"Error retrieving transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving transcription: {str(e)}")

    try:
        # Retrieve additional relevant context from Pinecone based on the question
        retrieved_context = retrieve_context_from_pinecone(question)
        logging.info(f"Retrieved additional context: {retrieved_context}")

        # Combine the base context with additional retrieved contexts
        full_context = f"{base_context} {retrieved_context}"
        logging.info("Full combined context: {full_context}")

        # Prepare inputs for the LangChain
        inputs = {"context": full_context, "question": question}
        logging.info(f"Inputs for LangChain: {inputs}")

        # Generate answer using LangChain
        result = chain.invoke(inputs)
        logging.info(f"Generated Response from LangChain: {result}")

        # Return the transcription and answer
        return {"transcription": base_context, "answer": result}

    except Exception as e:
        logging.error(f"Error during question answering: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve answer: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)