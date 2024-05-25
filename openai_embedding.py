from openai import OpenAI

file = open("api_key.txt")
openAI_client = OpenAI(
        api_key=file.readlines()[0].split(':')[1][:-1]
)
file.close()


def get_embedding(sub_model, query):
    return openAI_client.embeddings.create(input = query, model=sub_model).data[0].embedding

