# Code adapted from https://huggingface.co/docs/transformers/v4.25.1/en/tasks/language_modeling#language-modeling

# - Evaluation:
#     - Task 1: sentiment classification. For test set, evaluate accuracy of “this is _”.
#     - Task 2: negation understanding. For test set, evaluate accuracy of “this is not _”.
# - Plot:
#     - Matrix: x-axis: model size; y-axis: X%; cell: task 1/2 accuracy



from transformers import AutoTokenizer, AutoModelForCausalLM

prompt = "it is a terrible movie. this is not"

tokenizer = AutoTokenizer.from_pretrained("finetuned_distilgpt2_sst2_negation0.05")
inputs = tokenizer(prompt, return_tensors="pt").input_ids

model = AutoModelForCausalLM.from_pretrained("finetuned_distilgpt2_sst2_negation0.05")
outputs = model.generate(
    inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95
)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))


# if __name__ == "__main__":
#     evaluate()
    