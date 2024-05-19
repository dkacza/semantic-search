import ollama

import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Initialize vector DB client and collection details
current_id = 0


def recreateCollection(client, collection_name, vector_length):
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_length, distance=Distance.COSINE),
    )

def embedData(client, sub_model, collection_name, word):
    global current_id
    embedding = ollama.embeddings(model=sub_model, prompt=word)['embedding']
    client.upsert(collection_name=collection_name, wait=True, points=[
        PointStruct(
            id = current_id,
            vector = embedding,
            payload = {"word": word}
        )])
    current_id += 1
    print(f"Word {word} has been embedded with {sub_model} model into {collection_name} collection")


def performQuery(client, query):
    print(query)
    embedding = ollama.embeddings(model='llama3', prompt=query)
    print(f'Query {query} converted to vector. Finding matching vectors...')
    search_results = client.search(collection_name=LLAMA2_COLLECTION, query_vector=embedding['embedding'], limit=5)
    return search_results
