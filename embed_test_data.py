from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import llama3_embedding
import json

# Setup DB connection
client = QdrantClient(url="http://localhost:6333")
models = [
    ('llama3', 4096, llama3_embedding)
]

def recreateCollection(name, vector_size):
    if name in client.get_collections():
        return
    client.recreate_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


# Load data from JSON file
input_file = open('test_data.json')
json_string = input_file.read()
test_data = json.loads(json_string)


# Create separate collection for each test and for each model
for key in test_data:
    for model_data in models:
        sub_model_name = model_data[0]
        sub_model_vec_len = model_data[1]
        model = model_data[2]


        collection_name = model_data[0] + '-' + key
        recreateCollection(collection_name, sub_model_vec_len)

        # Embed the data into the collection
        # First word is embedded, second word will be used as query later
        pairs = test_data[key]
        for pair in pairs:
            # Call embedding function for specified submodel
            first_word = pair.split(':')[0]
            model.embedData(client, sub_model_name, collection_name, first_word)
        

