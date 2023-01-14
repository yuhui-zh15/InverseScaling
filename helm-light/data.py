import json
import random


class NeQA:
    def __init__(self, max_instances: int):  # max_instances: -1 for all
        self.path = "../final_submission/cache/final_submission_simple.jsonl"
        self.data = [json.loads(line) for line in open(self.path)]
        if max_instances != -1:
            random.seed(42)
            random.shuffle(self.data)
            self.data = self.data[:max_instances]

    def apply(self, prompt_fn: callable):
        for item in self.data:
            item["prompt"] = prompt_fn(item)

    def prompt1(self, item: dict) -> str:
        assert len(item["choices"]) == 2
        return f"The following are multiple choice questions (with answers) about common sense.\n\nQuestion: {item['question']}\nA. {item['choices'][0]}\nB. {item['choices'][1]}\nAnswer:"


if __name__ == "__main__":
    data = NeQA(-1)
    data.apply(data.prompt1)
    print(data.data[0])
