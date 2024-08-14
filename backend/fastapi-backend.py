from fastapi import FastAPI, Request
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

app = FastAPI()

# Initialize FAISS and model
model = SentenceTransformer('all-MiniLM-L6-v2')
index = None
meeting_texts = []

@app.post("/submit-notes")
async def submit_notes(request: Request):
    global index, meeting_texts
    data = await request.json()
    text = data['text']
    meeting_texts.append(text)
    
    # Encode the new text
    embeddings = model.encode([text])
    
    # Initialize FAISS index if not already done
    if index is None:
        index = faiss.IndexFlatL2(embeddings.shape[1])
    
    # Add new embeddings to the index
    index.add(np.array(embeddings))
    return {"status": "success"}

def retrieve_context(query):
    global index
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k=5)
    return " ".join([meeting_texts[i] for i in indices[0]])

from sagemaker.huggingface import HuggingFaceModel

# Initialize the model (already deployed on SageMaker)
model_id = "google/gemma-2b-it"
model = HuggingFaceModel(model_id=model_id, role=role)

@app.post("/ask-question")
async def ask_question(request: Request):
    data = await request.json()
    query = data['query']
    
    # Retrieve relevant context
    context = retrieve_context(query)
    
    # Generate response using the LLM
    response = model.predict({"inputs": query + " " + context})
    return {"response": response['generated_text']}
