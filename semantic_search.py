from qdrant_client import QdrantClient
import llama3_embedding
import openai_embedding


client = QdrantClient(url="http://localhost:6333")

# Test data
european_capitals = [
    "Amsterdam", "Andorra la Vella", "Athens", "Baku", "Belgrade", "Berlin", "Bern", "Bratislava",
    "Brussels", "Bucharest", "Budapest", "Chisinau", "Copenhagen", "Dublin", "Helsinki", "Kyiv",
    "Lisbon", "Ljubljana", "London", "Luxembourg", "Madrid", "Minsk", "Monaco", "Moscow", "Nicosia",
    "Nuuk", "Oslo", "Paris", "Podgorica", "Prague", "Reykjavik", "Riga", "Rome", "San Marino",
    "Sarajevo", "Skopje", "Sofia", "Stockholm", "Tallinn", "Tbilisi", "Tirana", "Vaduz", "Valletta",
    "Vatican City", "Vienna", "Vilnius", "Warsaw", "Yerevan", "Zagreb"
]
embedding_model = None
while True:
    print('''
    Select embedding model:
          1. llama3 - Local
          2. OpenAI - text-embedding-3-small
          3. OpenAI - text-embedding-3-large
          4. OpenAI - text-embedding-ada-002
    ''')
    user_input = input("> ")
    if user_input == "1":
        embedding_model = llama3_embedding
        break
    if user_input == "2":
        embedding_model = openai_embedding
        embedding_model.setModelId("text-embedding-3-small")
        break
    if user_input == "3":
        embedding_model = openai_embedding
        embedding_model.setModelId("text-embedding-3-large")
        break
    if user_input == "4":
        embedding_model = openai_embedding
        embedding_model.setModelId("text-embedding-ada-002")
        break

embedding_model.recreateCollection(client)
embedding_model.embedData(client, european_capitals)

while True:
    query = input("Enter a word or a sentence for matching\n> ")
    result = embedding_model.performQuery(client, query)
    print(result)

