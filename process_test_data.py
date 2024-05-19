import json

# Initial data from http://download.tensorflow.org/data/questions-words.txt
f = open('test_data.txt')
input = f.readlines()
f.close()
current_colection_name = ''
data = {}
for text_line in input:
    if text_line.startswith(': '):
        current_colection_name = text_line[2:-1]
        data[current_colection_name] = set()
        continue
    words = text_line.split(' ')
    pair = [words[0], words[1]]
    data[current_colection_name].add(':'.join(pair))

def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError

with open("test_data.json", "w") as outfile: 
    json.dump(data, outfile, default=set_default)