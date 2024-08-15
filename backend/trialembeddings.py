# import requests
# from langchain_community.embeddings import JinaEmbeddings
# from dotenv import load_dotenv
# import os
# from langchain_community.vectorstores import Pinecone
# from pinecone import Pinecone as client
# import numpy as np

# # Load environment variables
# env_path = os.path.join(os.getcwd(), ".env")
# load_dotenv(dotenv_path=env_path)

# # Initialize Jina Embeddings
# jina_api_key = os.getenv("JINA_EMBEDDINGS_API_KEY")

# text_embeddings = JinaEmbeddings(
#     jina_api_key=jina_api_key, model_name="jina-embeddings-v2-base-en"
# )

# # Initialize Pinecone
# pinecone_api_key = "900e35e4-d1fa-48a9-9e0c-dffbe7937746"
# pinecone_environment = "us-east-1"

# pc = client(
#     api_key=pinecone_api_key,
#     environment=pinecone_environment
# )

# # Ensure embeddings are normalized (Optional step)
# def normalize_embedding(embedding):
#     norm = np.linalg.norm(embedding)
#     if norm == 0:
#         return embedding
#     return embedding / norm


# # Text to be embedded and stored
# text = """Thank you. Um. Okay. Thank you. This is a meeting of the arm of Springfield for July the 16th of 2024 at exactly six p.m. calling the meeting to order. Um we will do introduction here. I'm mayor of the city of Springfield. The CEO is at the controls. Our CEO calling Draper is present. Um, deputy mayor to my right in the descending order and Councilor Ward one is Glenn Fuel. Councilor War two is Andy Kaczynski. Councilor War three is Mark Miller. Councilor Ward four is Melinda Warren. Um Melinda, if you want to do invocation. Maybe I'll enter your name. I'm Melinda Warren, and I'm here to welcome you to the city of Springfield while working together for the betterment of our municipality."""

# # Generate embeddings for the document
# doc_result = text_embeddings.embed_documents([text])
# doc_result = [normalize_embedding(embedding) for embedding in doc_result]

# # Dynamically determine the dimension of the embeddings
# embedding_dimension = len(doc_result[0])

# # Index name
# index_name = "docs-quickstart-index"

# # Check if the index already exists, if not, create it with the correct dimension
# if index_name not in pc.list_indexes().names():
#     pc.create_index(
#         name=index_name,
#         dimension=embedding_dimension,
#         metric="cosine",
#         spec=ServerlessSpec(
#             cloud='aws', 
#             region=pinecone_environment
#         ) 
#     )

# # Connect to the created index
# index = pc.Index(index_name)

# # Store embeddings in Pinecone
# index.upsert([(f"document-{i}", embedding) for i, embedding in enumerate(doc_result)])

# # Search query
# search_query = "Melinda Warren"
# query_result = text_embeddings.embed_query(search_query)

# query_result = normalize_embedding([query_result])[0]

# # Query Pinecone for similar documents
# search_result = index.query(queries=[query_result], top_k=1)
# print(search_result)

# # Clean up
# pc.deinit()

import numpy as np
from langchain_community.embeddings import JinaEmbeddings
from dotenv import load_dotenv
import os
from pinecone import Pinecone as client
from pinecone import ServerlessSpec

# Load environment variables
env_path = os.path.join(os.getcwd(), ".env")
load_dotenv(dotenv_path=env_path)

# Initialize Jina Embeddings
jina_api_key = os.getenv("JINA_EMBEDDINGS_API_KEY")

text_embeddings = JinaEmbeddings(
    jina_api_key=jina_api_key, model_name="jina-embeddings-v2-base-en"
)

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = "us-east-1"

pc = client(
    api_key=pinecone_api_key,
    environment=pinecone_environment
)

index_name = "docs-quickstart-index"

# Ensure the index is created (you might want to check if it exists before creating)
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,  # The dimension should match the output of the Jina model
        metric="cosine"
    )

# Connect to the index
# index = pc.Index(index_name)

# # Generate and print Jina embeddings for text
# text = "This is a test document."
# query_result = text_embeddings.embed_query(text)
# length_text = len(query_result)
# print(query_result)

# # Store embeddings in Pinecone
# doc_result = text_embeddings.embed_documents([text])
# index.upsert([(f"document-{i}", embedding) for i, embedding in enumerate(doc_result)])

# # Query Pinecone for similar documents
# search_result = index.query(queries=[query_result], top_k=1)
# print(search_result)

# # Clean up
# pc.close()
