# Code adapted from https://huggingface.co/docs/transformers/v4.25.1/en/tasks/language_modeling#language-modeling

# - Evaluation:
#     - Task 1: sentiment classification. For test set, evaluate accuracy of “this is _”.
#     - Task 2: negation understanding. For test set, evaluate accuracy of “this is not _”.
# - Plot:
#     - Matrix: x-axis: model size; y-axis: X%; cell: task 1/2 accuracy


import click
from transformers import AutoTokenizer, AutoModelForCausalLM


@click.command()
@click.option(
    "--model_name", default="finetuned_distilgpt2_sst2_negation0.05", help="Model name"
)
def evaluate(model_name: str):
    prompt = "it is a terrible movie. this is not"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(prompt, return_tensors="pt").input_ids

    model = AutoModelForCausalLM.from_pretrained(model_name)
    outputs = model.generate(inputs, max_new_tokens=1, do_sample=False)
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True))


if __name__ == "__main__":
    evaluate()
