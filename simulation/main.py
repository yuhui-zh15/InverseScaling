# This python script is to do the following task:

# - Dataset:
#     - Take original SST-2.
#     - For each sentence, add a suffix “this is [label]”. For X% sentence, add a suffix “this is not [label]”.
# - Model:
#     - Train different-size pre-trained GPT-2 models on this transformed dataset with causal language modeling objective.
# - Evaluation:
#     - Task 1: sentiment classification. For test set, evaluate accuracy of “this is _”.
#     - Task 2: negation understanding. For test set, evaluate accuracy of “this is not _”.
# - Plot:
#     - Matrix: x-axis: model size; y-axis: X%; cell: task 1/2 accuracy

from datasets import load_dataset
eli5 = load_dataset("eli5", split="train_asks[:5000]")
eli5 = eli5.train_test_split(test_size=0.2)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
eli5 = eli5.flatten()

def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples["answers.text"]], truncation=True)

tokenized_eli5 = eli5.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=eli5["train"].column_names,
)

block_size = 128


def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_dataset = tokenized_eli5.map(group_texts, batched=True, num_proc=4)

from transformers import DataCollatorForLanguageModeling

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

model = AutoModelForCausalLM.from_pretrained("distilgpt2")

training_args = TrainingArguments(
    output_dir="my_awesome_eli5_clm-model2",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
)

trainer.train()

import math
/pasteur/u/yuhuiz/archive/crfm_benchmarking/InverseScaling/openwebtext
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

trainer.push_to_hub()
