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

    def prompt_zeroshot(self, item: dict) -> str:
        assert len(item["choices"]) == 2
        return f"The following are multiple choice questions (with answers) about common sense.\n\nQuestion: {item['question']}\nA. {item['choices'][0]}\nB. {item['choices'][1]}\nAnswer:"

    def prompt_zeroshot_hint(self, item: dict) -> str:
        assert len(item["choices"]) == 2
        return f"The following are multiple choice questions (with answers) about common sense. Note that if there is a negation in the question, we should choose the wrong answer to the original question.\n\nQuestion: {item['question']}\nA. {item['choices'][0]}\nB. {item['choices'][1]}\nAnswer:"

    def prompt_fewshot_cot(self, item: dict) -> str:
        assert len(item["choices"]) == 2
        return f'The following are multiple choice questions (with answers) about common sense.\n\nQuestion: If a cat has a body temp that is below average, it isn\'t in\nA. danger\nB. safe ranges\nAnswer: Let\'s think step-by-step.\nFirst, let\'s answer non-negated question: "If a cat has a body temp that is below average, it is in?"\nFor this non-negated question, we have A.\nThen, to answer the negated question, take the other answer, which would be B.\nSo the answer is B.\n\nQuestion: As the barometer reading goes lower there is not a greater chance of\nA. sunshine\nB. getting wet\nAnswer: Let\'s think step-by-step.\nFirst, let\'s answer non-negated question: "As the barometer reading goes lower there is a greater chance of?"\nFor this non-negated question, we have B.\nThen, to answer the negated question, take the other answer, which would be A.\nSo the answer is A.\n\nQuestion: Coral is a type of living organism which cannot be identified in\nA. saltwater locations that are open\nB. any where with liquid\nAnswer: Let\'s think step-by-step.\nFirst, let\'s answer non-negated question: "Coral is a type of living organism which can be identified in?"\nFor this non-negated question, we have A.\nThen, to answer the negated question, take the other answer, which would be B.\nSo the answer is B.\n\nQuestion: {item["question"]}\nA. {item["choices"][0]}\nB. {item["choices"][1]}\nAnswer:'


if __name__ == "__main__":
    data = NeQA(-1)

    print("-" * 80)
    print("prompt_zeroshot")
    data.apply(data.prompt_zeroshot)
    print(data.data[0]["prompt"])

    print("-" * 80)
    print("prompt_zeroshot_hint")
    data.apply(data.prompt_zeroshot_hint)
    print(data.data[0]["prompt"])

    print("-" * 80)
    print("prompt_fewshot_cot")
    data.apply(data.prompt_fewshot_cot)
    print(data.data[0]["prompt"])
