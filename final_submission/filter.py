import json

negatedlama = [json.loads(line) for line in open('negatedlama.jsonl')]
negatedlama = [item for item in negatedlama if item['metadata']['dataset'] in ["ConceptNet", "TREx"]]

obqa = [json.loads(line) for line in open('obqa.jsonl')]
obqa = [item for item in obqa if item['negation_rule'] in ["not/be", "not/because"]]

all_data = negatedlama + obqa

with open('final_submission.jsonl', 'w') as f:
    for item in all_data:
        f.write(json.dumps(item) + '\n')