from qdrant_client import QdrantClient
import llama3_embedding
import openai_embedding
import vertex_ai_embedding
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

# Test data
european_capitals = [
    "Amsterdam", "Andorra la Vella", "Athens", "Baku", "Belgrade", "Berlin", "Bern", "Bratislava",
    "Brussels", "Bucharest", "Budapest", "Chisinau", "Copenhagen", "Dublin", "Helsinki", "Kyiv",
    "Lisbon", "Ljubljana", "London", "Luxembourg", "Madrid", "Minsk", "Monaco", "Moscow", "Nicosia",
    "Nuuk", "Oslo", "Paris", "Podgorica", "Prague", "Reykjavik", "Riga", "Rome", "San Marino",
    "Sarajevo", "Skopje", "Sofia", "Stockholm", "Tallinn", "Tbilisi", "Tirana", "Vaduz", "Valletta",
    "Vatican City", "Vienna", "Vilnius", "Warsaw", "Yerevan", "Zagreb"
]

model_data = None
while model_data == None:
    current_index = 0
    print('Select embedding model:')
    for model_data in models:
        print(f'{current_index + 1}. {model_data[0]}')
        current_index += 1
    user_input = int(input('> ')) - 1
    try:
        model_data = models[user_input]
    except:
        model_data = None 

sub_model = model_data[0]
vector_size = model_data[1]
model = model_data[2]
collection_name = 'Demo'

utils.recreate_collection(client, collection_name, vector_size)
for word in european_capitals:
    utils.embed_data(client, collection_name, model, sub_model, word, current_index)
    current_index += 1

while True:
    query = input("Enter a word or a sentence for matching\n> ")
    result = utils.perform_query(client, collection_name, model, sub_model, query)
    print(result)