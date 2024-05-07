import ollama

import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Initialize vector DB client and collection details
LLAMA2_VEC_LEN = 4096
LLAMA2_COLLECTION = 'llama3-test'
current_id = 0


def recreateCollection(client):
    client.recreate_collection(
        collection_name=LLAMA2_COLLECTION,
        vectors_config=VectorParams(size=LLAMA2_VEC_LEN, distance=Distance.COSINE),
    )

def embedData(client, data):
    print('Embedding data...')
    for word in data:
        global current_id
        embedding = ollama.embeddings(model='llama3', prompt=word)['embedding']
        client.upsert(collection_name=LLAMA2_COLLECTION, wait=True, points=[
            PointStruct(
                id = current_id,
                vector = embedding,
                payload = {"word": word}
            )])
        current_id += 1
        print(f'Vector for "{word}" embedded successfully')
    print(f'{len(data)} vectors successfully inserted')

def performQuery(client, query):
    print(query)
    embedding = ollama.embeddings(model='llama3', prompt=query)
    print(f'Query {query} converted to vector. Finding matching vectors...')
    search_results = client.search(collection_name=LLAMA2_COLLECTION, query_vector=embedding['embedding'], limit=5)
    return search_results
