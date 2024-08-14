from fastapi import FastAPI, HTTPException, File, UploadFile
import os
import openai
from pinecone import Pinecone, ServerlessSpec
import uvicorn
from langchain import LLMChain, PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.embeddings.openai import OpenAIEmbeddings

app = FastAPI()

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
# pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )

# Check if the index exists, and create it if it doesn't
index_name = "audio-index"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name, 
        dimension=1536, 
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='gcp',  # or 'aws', depending on your preference
            region='us-west1'  # or another region where you want to host your index
        )
    )
# You can now access the index
index = pc.Index(index_name)

# Setup LangChain components
embeddings = OpenAIEmbeddings()
vectorstore = LangchainPinecone(index, embeddings.embed_query, "text")

llm = OpenAI(model_name="gpt-3.5-turbo")
qa_chain = load_qa_chain(llm, chain_type="stuff")

# Transcribe audio to text using Whisper API
@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())

    try:
        transcription = openai.Audio.transcribe("whisper-1", open(file_location, "rb"))
        return {"transcription": transcription["text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        os.remove(file_location)

# Embed the transcribed text and store in Pinecone
def store_in_pinecone(text):
    vector = embeddings.embed_query(text)
    vectorstore.add_texts([text], [vector])

# Generate a response using RAG with ChatGPT 3.5 and LangChain
@app.post("/ask_from_audio/")
async def ask_from_audio(file: UploadFile = File(...), question: str = ""):
    transcription_result = await transcribe_audio(file)
    context = transcription_result["transcription"]

    # Store transcription in Pinecone
    store_in_pinecone(context)

    # Perform RAG: Retrieve relevant documents from Pinecone and generate a response
    docs = vectorstore.similarity_search(question, k=5)
    answer = qa_chain.run(input_documents=docs, question=question)

    return {"transcription": context, "answer": answer}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
