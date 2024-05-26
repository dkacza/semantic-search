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

final_scores = {}
categories = []

# Create separate collection for each test and for each model
for model_data in models:
    sub_model = model_data[0]
    sub_model_vec_len = model_data[1]
    model = model_data[2]
    current_model_scores = {}

    print(f'Using {sub_model} model')

    for key in test_data:

        categories.append(key)
        
        # Get current collection 
        collection_name = sub_model + '-' + key
        collection_score = 0

        # Query with second word
        pairs = test_data[key]
        max_score = len(pairs)
        current_model_scores[collection_name] = {}
        current_model_scores[collection_name]['max_score'] = max_score
        current_model_scores[collection_name]['random_threshold_percentage'] = (1 / max_score) + (0.5 / max_score) + (0.25 / max_score) + (0.125 / max_score) + (0.0625 / max_score) 

        for pair in pairs:
            # Call embedding function for specified submodel
            [correct_word, query] = pair.split(':')
            results = utils.perform_query(client, collection_name, model, sub_model, query)
            matchingWords = list(map(lambda result: result.payload['word'], results))
            print(f'Correct answer: {correct_word}, result: {matchingWords}')

            # Calculate score
            query_score = 1
            if correct_word in matchingWords:
                for i in range(len(matchingWords)):
                    if matchingWords[i] == correct_word:
                        break
                    query_score /= 2
            else:
                query_score = 0
            collection_score += query_score

            print(f'Scored points: {query_score}')
            print('\n')
        
        # Set score and percentage for a collection
        current_model_scores[collection_name]['score'] = collection_score
        current_model_scores[collection_name]['correct_percentage'] = collection_score / max_score

    # Set scores for model
    final_scores[sub_model] = current_model_scores 


# Save to JSON file
with open("results.json", "w") as outfile: 
    json.dump(final_scores, outfile)


# Save to CSV file for diagrams
# Create heading
csv_output_string = 'model,'
for category in categories:
    csv_output_string += category + ','
csv_output_string = csv_output_string[:-1] + '\n'

# Scores of each model and threshold score
threshold_scores = []
for model in final_scores:
    csv_output_string += model + ','
    for category in categories:
        collection_name = model + '-' + category
        score = final_scores[model][collection_name]['correct_percentage']
        threshold_scores.append(final_scores[model][collection_name]['random_threshold_percentage'])
        csv_output_string += str(score) + ','
    csv_output_string = csv_output_string[:-1] + '\n'

# Threshold scores
csv_output_string += 'threshold,'
for threshold in threshold_scores:
    csv_output_string += str(threshold) + ','
csv_output_string = csv_output_string[:-1] + '\n'
        

with open('results.csv', 'w') as file:
    file.write(csv_output_string)



            
