import ollama

def get_embedding(sub_model, query):
    return ollama.embeddings(model=sub_model, prompt=query)['embedding']