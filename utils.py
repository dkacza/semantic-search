from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

def recreate_collection(client, name, vector_size):
    client.recreate_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

def embed_data(client, collection_name, model, sub_model, query, id):
    embedding = model.get_embedding(sub_model, query)
    client.upsert(collection_name=collection_name, wait=True, points=[
        PointStruct(
            id = id,
            vector = embedding,
            payload = {"word": query}
        )])
    print(f"Word {query} has been embedded with {sub_model} model into {collection_name} collection")

def perform_query(client, collection_name, model, sub_model, query):
    embedding = model.get_embedding(sub_model, query)
    print(f'Query {query} converted to vector. Finding matching vectors in {collection_name} collection.')
    search_results = client.search(collection_name=collection_name, query_vector=embedding, limit=5)
    return search_results
