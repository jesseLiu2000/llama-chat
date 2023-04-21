from flask import Flask, render_template, request
import requests
import torch
from transformers import AutoTokenizer, LlamaForCausalLM
import pickle

from llama_chat import our_fast_chat
from use_alpaca_ex import load_alpaca_data 

app = Flask(__name__)
app.static_folder = 'static'
app.knowledge_pool = []

model = LlamaForCausalLM.from_pretrained("/scratch/zijie/github/weights/llama-13b", device_map="auto", load_in_8bit=True) 
tokenizer = AutoTokenizer.from_pretrained("/scratch/zijie/github/weights/llama-13b", use_fast=False)
alpaca_instructions_full, alpaca_outputs = load_alpaca_data() # load_dolly_data() 
# Saving n-grams cuts time down by ~10 seconds
file_name_instruction = "alpaca_instruction_ngrams_1.pkl"
with open(file_name_instruction, "rb") as f:
    instruction_ngrams = pickle.load(f) 
file_name_output = "alpaca_output_ngrams_1.pkl"
with open(file_name_output, "rb") as f:
    output_ngrams = pickle.load(f) 

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/demo')
def register():
   # Ensure the user reached path via GET
   if request.method == "GET":
      return render_template("demo.html")

   else:
      pass # Pass is a Python way to say 'do nothing'

@app.route("/get")
def get_bot_response():
    prompt = request.args.get('msg')

    #inputs = tokenizer(prompt, return_tensors="pt")
    # Generate
    #generate_ids = model.generate(inputs.input_ids, max_length=30)
    #result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    def generate_funct(inputs):    
        inputs = tokenizer([inputs])        
        # print("shape inputs: ", torch.as_tensor(inputs.input_ids).shape)
        output_ids = model.generate(
            torch.as_tensor(inputs.input_ids).cuda(),
            do_sample=True,
            #penalty_alpha=0.6,
            #top_k=6,
            temperature=0.7,
            max_new_tokens=256) # Changed from 1024 due to few-shot examples.
        chat_output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        return chat_output
    result, local_knowledge_pool = our_fast_chat(
        generate_funct, prompt, instruction_ngrams, 
        alpaca_outputs, alpaca_fs=True, most_similar=True, 
        k=2, min_length=50, knowledge_pool=app.knowledge_pool, 
        instructions_full=alpaca_instructions_full, dialogue_chat=True,
        ngram_outputs=output_ngrams
    )
    app.knowledge_pool = local_knowledge_pool
    # Replace newlines with <br> so that the text is displayed properly
    result = result.replace("\n", "<br>")
    return result

if __name__ == "__main__":
    print("======== Flask Running =========")
    app.run(host="10.24.77.19", port="8080", debug=True)
