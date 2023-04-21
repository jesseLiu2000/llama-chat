from flask import Flask, render_template, request
import requests

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
    output = query({
        "inputs": userText,
    })
    generated_text = output[0]['generated_text']
    return generated_text.replace(userText, '').replace('\n', '').replace('   	', ' ')

if __name__ == "__main__":
    app.run(host="10.24.77.19", port="8080", debug=True)
