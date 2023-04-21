from flask import Flask, render_template, request
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda"

model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", cache_dir="/scratch/cache", device_map="balanced_low_0")
model.save_pretrained("/scratch/zijie/gpt-j/")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")


app = Flask(__name__)
app.static_folder = 'static'

API_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-j-6B"
headers = {"Authorization": "Bearer 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"}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')

    prompt = ("Question:", +
        userText +
              "Answer: "
    )

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.9,
        max_length=100,
    )
    gen_text = tokenizer.batch_decode(gen_tokens)[0]

    return gen_text

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
