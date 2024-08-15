import numpy as np
from langchain_community.embeddings import JinaEmbeddings
from dotenv import load_dotenv
import os
from pinecone import Pinecone as client
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Pinecone as PineconeStore
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

# Generate a sample embedding to determine the dimension
sample_text = "This is a sample text to determine embedding dimension."
sample_embedding = text_embeddings.embed_documents([sample_text])
embedding_dimension = len(sample_embedding[0])
print("embedding length: ", embedding_dimension)

# Ensure the index is created with the correct dimension
# if index_name not in pc.list_indexes():
#     pc.create_index(
#         name=index_name,
#         dimension=embedding_dimension,  # Use the dynamic embedding dimension
#         metric="cosine",
#         spec=ServerlessSpec(
#             cloud='aws', 
#             region=pinecone_environment
#         ) 
#     )

# Connect to the index
index = PineconeStore.from_existing_index(index_name=index_name, embedding=text_embeddings)

# Generate and print Jina embeddings for the text
text = "This is a test document."
doc_result = text_embeddings.embed_documents([text])

# Store embeddings in Pinecone
index.add_texts([text])

# Query Pinecone for similar documents
query_result = text_embeddings.embed_query(text)
search_result = index.similarity_search(text, k=1)
print(search_result)

# Initialize LLM and QA Chain
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
chain = load_qa_chain(llm, chain_type="stuff")

# Function to retrieve answers from the Pinecone index
def retrieve_answers(query):
    doc_search = index.similarity_search(query, k=2)
    print(doc_search)
    response = chain.run(input_documents=doc_search, question=query)
    return response

# Example query
our_query = "Melinda Warren"
answer = retrieve_answers(our_query)
print(answer)

