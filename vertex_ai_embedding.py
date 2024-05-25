import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

file = open("api_key.txt")
vertexai.init(
    project = file.readlines()[1].split(':')[1]
)
file.close()

def get_embedding(sub_model, query):
    model = TextEmbeddingModel.from_pretrained(sub_model)
    input = TextEmbeddingInput(query, "RETRIEVAL_DOCUMENT")
    embeddings = model.get_embeddings([input])
    return [embedding.values for embedding in embeddings][0]
