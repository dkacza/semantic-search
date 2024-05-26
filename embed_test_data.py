from qdrant_client import QdrantClient
import llama3_embedding
import openai_embedding
import vertex_ai_embedding
import json
import utils

# Setup DB connection
client = QdrantClient(url="http://localhost:6333")
models = [
    ('llama3', 4096, llama3_embedding),
    ('text-embedding-3-small', 1536, openai_embedding),
    ('text-embedding-3-large', 3072, openai_embedding),
    ('text-embedding-ada-002', 1536, openai_embedding),
    ('text-embedding-004', 768, vertex_ai_embedding),
]

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
        utils.recreate_collection(client, collection_name, sub_model_vec_len)

        # Embed the data into the collection
        # First word is embedded, second word will be used as query later
        pairs = test_data[key]
        current_id = 0
        for pair in pairs:
            current_id += 1
            # Call embedding function for specified submodel
            first_word = pair.split(':')[0]
            utils.embed_data(client, collection_name, model, sub_model, first_word, current_id)
        

