from pinecone import Pinecone
from pinecone import ServerlessSpec

pinecone_api_key = "900e35e4-d1fa-48a9-9e0c-dffbe7937746"
pinecone_environment = "us-east-1"

pc = Pinecone(api_key=pinecone_api_key)

index_name = "docs-quickstart-index"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'
        ) 
    ) 