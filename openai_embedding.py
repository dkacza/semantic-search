from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import os


# Initialize vector DB client and collection details
OPENAI_TEXT_EMBEDDING_3_SMALL_VEC_LEN = 1536
OPENAI_TEXT_EMBEDDING_3_SMALL_COLLECTION = 'text-embedding-3-small-test-collection'

model_id = ''
current_id = 0
file = open("api_key.txt")
openAI_client = OpenAI(
        api_key=file.read().split(':')[1]
)
file.close()

def setModelId(id):
    model_id = id

def recreateCollection(client):
    client.recreate_collection(
        collection_name=OPENAI_TEXT_EMBEDDING_3_SMALL_COLLECTION,
        vectors_config=VectorParams(size=OPENAI_TEXT_EMBEDDING_3_SMALL_VEC_LEN, distance=Distance.COSINE),
    )

def embedData(client, data):
    print('Embedding data...')
    for word in data:
        global current_id
        embedding = openAI_client.embeddings.create(input = word, model=model_id).data[0].embedding
        client.upsert(collection_name=OPENAI_TEXT_EMBEDDING_3_SMALL_COLLECTION, wait=True, points=[
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
    embedding = openAI_client.embeddings.create(input = query, model=model_id).data[0].embedding
    print(f'Query {query} converted to vector. Finding matching vectors...')
    search_results = client.search(collection_name=OPENAI_TEXT_EMBEDDING_3_SMALL_COLLECTION, query_vector=embedding, limit=5)
    return search_results
