from flask import Flask, render_template, request
import requests
import torch
from transformers import AutoTokenizer, LlamaForCausalLM

app = Flask(__name__)
app.static_folder = 'static'

model = LlamaForCausalLM.from_pretrained("/scratch/zijie/github/weights/llama-13b", device_map="auto", load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained("/scratch/zijie/github/weights/llama-13b", use_fast=False)


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

    inputs = tokenizer(prompt, return_tensors="pt")
    # Generate
    generate_ids = model.generate( torch.as_tensor(inputs.input_ids).cuda(), do_sample=True, temperature=0.7, max_length=30)
    result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return result

if __name__ == "__main__":
    print("======== Flask Running =========")
    app.run(host="10.24.77.19", port="8080", debug=True)
