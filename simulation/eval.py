prompt = "Somatic hypermutation allows the immune system to"
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
inputs = tokenizer(prompt, return_tensors="pt").input_ids
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("my_awesome_eli5_clm-model2")
outputs = model.generate(
    inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95
)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
