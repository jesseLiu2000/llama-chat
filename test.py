from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", cache_dir="/scratch/cache")

text = "My name is Julien and I like to "

output = model({
    "inputs": text,
})
print(output)
generated_text = output[0]['generated_text']

print(generated_text.replace(text, '').replace('\n', '').replace('   	',' '))
