import random
import json

random.seed(1234)
data = [json.loads(line) for line in open("sample_not_be.jsonl", "r")]
random.shuffle(data)
with open("sample_not_be_shuffled.jsonl", "w") as f:
    for line in data[:500]:
        f.write(json.dumps(line) + "\n")