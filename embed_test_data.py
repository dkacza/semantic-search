from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import llama3_embedding
import openai_embedding
import vertex_ai_embedding
import json

# Setup DB connection
client = QdrantClient(url="http://localhost:6333")
models = [
    ('llama3', 4096, llama3_embedding),
    ('text-embedding-3-small', 1536, openai_embedding),
    ('text-embedding-3-large', 3072, openai_embedding),
    ('text-embedding-ada-002', 1536, openai_embedding),
    ('text-embedding-004', 768, vertex_ai_embedding),
]

def recreateCollection(name, vector_size):
    client.recreate_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

def embedData(client, collection_name, model, sub_model, query, id):
    embedding = model.get_embedding(sub_model, query)
    client.upsert(collection_name=collection_name, wait=True, points=[
        PointStruct(
            id = id,
            vector = embedding,
            payload = {"word": query}
        )])
    print(f"Word {query} has been embedded with {sub_model} model into {collection_name} collection")

def performQuery(client, collection_name, sub_model, query):
    embedding = model.get_embedding(sub_model, query)
    print(f'Query {query} converted to vector. Finding matching vectors.')
    search_results = client.search(collection_name=collection_name, query_vector=embedding, limit=5)
    return search_results



# Load data from JSON file
input_file = open('test_data.json')
json_string = input_file.read()
test_data = json.loads(json_string)


# Create separate collection for each test and for each model

for model_data in models:
    sub_model = model_data[0]
    sub_model_vec_len = model_data[1]
    model = model_data[2]
    print(f'Using {sub_model} model')

    for key in test_data:

        # Extract data about model
        collection_name = sub_model + '-' + key
        present_collections = list(map(lambda entry: entry.name, client.get_collections().collections))
        if collection_name in present_collections:
            print(f'Collection {collection_name} already present')
            continue
        recreateCollection(collection_name, sub_model_vec_len)

        # Embed the data into the collection
        # First word is embedded, second word will be used as query later
        pairs = test_data[key]
        current_id = 0
        for pair in pairs:
            current_id += 1
            # Call embedding function for specified submodel
            first_word = pair.split(':')[0]
            embedData(client, collection_name, model, sub_model, first_word, current_id)
        

