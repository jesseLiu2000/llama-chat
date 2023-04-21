# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json

from llama_chat import get_few_shot1, one_chat

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA
from flask import Flask, render_template, request
import requests

app = Flask(__name__)
app.static_folder = 'static'


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int = 2048,
    max_batch_size: int = 1,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


class main:
    def __init__(
            self,
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 2048,
    max_batch_size: int = 32,
    few_shot: list = [] 
):
        self.generator = None
        self.few_shot = get_few_shot1()

    def init(self):
        if self.generator:
            return self.generator
        self.few_shot = get_few_shot1()
        local_rank, world_size = setup_model_parallel()
        if local_rank > 0:
             sys.stdout = open(os.devnull, "w")

    # local_rank = 0
    # world_size = 1

    #global generator
        generator = load(
        "/scratch/llama/weights/7B", "/scratch/llama/weights/tokenizer.model",
        local_rank, world_size 
        #ckpt_dir, tokenizer_path, local_rank,
        #world_size, max_seq_len, max_batch_size
    )
        self.generator = generator
        return generator

MAIN = main("/scratch/llama/weights/65B", "/scratch/llama/weights/tokenizer.model")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    generator = MAIN.init()
    try:
        #generator = main("/scratch/llama/weights/7B", "/scratch/llama/weights/tokenizer.model")
        generator = MAIN.init()
    except Exception as e:
        pass
    # userText = list(request.args.get('msg'))
    userText = request.args.get('msg')
    generate_one_param = lambda x: generator.generate([x], max_gen_len=256, temperature=0.7, top_p=0.95)[0][len(x):]
    # NEED to store few-shot parameters somewhere.
    # print("few shot: ", MAIN.few_shot)
    results, few_shot = one_chat(generate_one_param, userText, MAIN.few_shot)
    MAIN.few_shot = few_shot
    return results



if __name__ == "__main__":
    #main("/scratch/llama/weights/7B", "/scratch/llama/weights/tokenizer.model")
    print("====>")
    app.run(host="10.24.77.19", port="8080", debug=True)
