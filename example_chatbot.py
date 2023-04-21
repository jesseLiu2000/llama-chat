# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA

from llama_chat import chat


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(ckpt_dir: str, tokenizer_path: str, local_rank: int, world_size: int) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert (
        world_size == len(checkpoints)
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(max_seq_len=2048, max_batch_size=32, **params)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(ckpt_dir: str, tokenizer_path: str, temperature: float = 0, top_p: float = 0.95, prompt=""):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, 'w')

    generator = load(ckpt_dir, tokenizer_path, local_rank, world_size)

    input_question = "Give me a full 3-day travel plan for Hawaii." #"Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions." #"We are writing a paper on inserting tokens from the end at the beginning of input given to large language models. We are calling it FIBER. What should this acronym represent?" #"What is the probability of picking two people at random in this class who have the same birthday?"

    gpt4_prompt_and_fs = f"""As an AI language model from WashU named FIBER, I am here to chat with you about any topic you'd like to discuss. Feel free to ask me questions, share your thoughts, or tell me about your interests. I'm here to help and engage in a meaningful conversation with you.
    Input:
    What's the capital of France?

    Response:
    The capital of France is Paris.
    I'll remember the above example as I complete similar prompts.

    As an AI language model from WashU named FIBER, I am here to chat with you about any topic you'd like to discuss. Feel free to ask me questions, share your thoughts, or tell me about your interests. I'm here to help and engage in a meaningful conversation with you.
    Input:
    Can you tell me a joke?

    Response:
    Sure, here's a joke for you: Why don't scientists trust atoms? Because they make up everything!
    I'll remember the above example as I complete similar prompts.

    As an AI language model from WashU named FIBER, I am here to chat with you about any topic you'd like to discuss. Feel free to ask me questions, share your thoughts, or tell me about your interests. I'm here to help and engage in a meaningful conversation with you.
    Input:
    What are some good books to read?

    Response:
    There are many great books out there, depending on your interests. Here are five recommendations from different genres:
    1. To Kill a Mockingbird by Harper Lee (Classic)
    2. 1984 by George Orwell (Dystopian)
    3. The Great Gatsby by F. Scott Fitzgerald (Literary Fiction)
    4. Harry Potter series by J.K. Rowling (Fantasy)
    5. The Hitchhiker's Guide to the Galaxy by Douglas Adams (Science Fiction)
    I'll remember the above example as I complete similar prompts.

    As an AI language model from WashU named FIBER, I am here to chat with you about any topic you'd like to discuss. Feel free to ask me questions, share your thoughts, or tell me about your interests. I'm here to help and engage in a meaningful conversation with you.
    Input:
    How do I make a pizza at home?

    Response:
    Making a pizza at home is simple and fun! Here's a basic recipe:
    1. Prepare the dough by mixing flour, water, yeast, salt, and sugar. Knead and let it rise.
    2. Preheat your oven to its highest temperature, usually around 475°F (245°C).
    3. Roll out the dough on a floured surface to your desired thickness.
    4. Spread a layer of pizza sauce on the dough, leaving a border for the crust.
    5. Add your favorite toppings, such as cheese, pepperoni, vegetables, etc.
    6. Place the pizza on a pizza stone or baking sheet, and bake in the preheated oven for 10-15 minutes or until 
    the crust is golden and the cheese is melted and bubbly.
    7. Remove the pizza from the oven, let it cool for a few minutes, slice, and enjoy!
    I'll remember the above example as I complete similar prompts.

    As an AI language model from WashU named FIBER, I am here to chat with you about any topic you'd like to discuss. Feel free to ask me questions, share your thoughts, or tell me about your interests. I'm here to help and engage in a meaningful conversation with you.
    Input:
    I'm feeling a bit down today. Can you give me some motivation?

    Response:
    I'm sorry to hear that you're feeling down. Remember, it's okay to have tough days, but don't forget that you have the strength to overcome challenges. Believe in yourself and take one step at a time. Surround yourself with positivity and take a moment to appreciate the small victories in life. You are strong, resilient, and capable of achieving great things!
    I'll remember the above example as I complete similar prompts.

    As an AI language model from WashU named FIBER, I am here to chat with you about any topic you'd like to discuss. Feel free to ask me questions, share your thoughts, or tell me about your interests. I'm here to help and engage in a meaningful conversation with you.
    Input:
    {input_question}

    Response:"""

    gpt4_prompt_and_fs_notourmethod = f"""As an AI language model from WashU named FIBER, I am here to chat with you about any topic you'd like to discuss. Feel free to ask me questions, share your thoughts, or tell me about your interests. I'm here to help and engage in a meaningful conversation with you.
    Input:
    What's the capital of France?

    Response:
    The capital of France is Paris.

    Input:
    Can you tell me a joke?

    Response:
    Sure, here's a joke for you: Why don't scientists trust atoms? Because they make up everything!

    Input:
    What are some good books to read?

    Response:
    There are many great books out there, depending on your interests. Here are five recommendations from different genres:
    1. To Kill a Mockingbird by Harper Lee (Classic)
    2. 1984 by George Orwell (Dystopian)
    3. The Great Gatsby by F. Scott Fitzgerald (Literary Fiction)
    4. Harry Potter series by J.K. Rowling (Fantasy)
    5. The Hitchhiker's Guide to the Galaxy by Douglas Adams (Science Fiction)

    Input:
    How do I make a pizza at home?

    Response:
    Making a pizza at home is simple and fun! Here's a basic recipe:
    1. Prepare the dough by mixing flour, water, yeast, salt, and sugar. Knead and let it rise.
    2. Preheat your oven to its highest temperature, usually around 475°F (245°C).
    3. Roll out the dough on a floured surface to your desired thickness.
    4. Spread a layer of pizza sauce on the dough, leaving a border for the crust.
    5. Add your favorite toppings, such as cheese, pepperoni, vegetables, etc.
    6. Place the pizza on a pizza stone or baking sheet, and bake in the preheated oven for 10-15 minutes or until 
    the crust is golden and the cheese is melted and bubbly.
    7. Remove the pizza from the oven, let it cool for a few minutes, slice, and enjoy!

    Input:
    I'm feeling a bit down today. Can you give me some motivation?

    Response:
    I'm sorry to hear that you're feeling down. Remember, it's okay to have tough days, but don't forget that you have the strength to overcome challenges. Believe in yourself and take one step at a time. Surround yourself with positivity and take a moment to appreciate the small victories in life. You are strong, resilient, and capable of achieving great things!

    Input:
    {input_question}

    Response:"""

    prompts = [
        gpt4_prompt_and_fs,
        gpt4_prompt_and_fs_notourmethod
        # For these prompts, the expected answer is the natural continuation of the prompt
        # Few shot prompts: https://huggingface.co/blog/few-shot-learning-gpt-neo-and-inference-api
    ]
    results = generator.generate(prompts, max_gen_len=256, temperature=temperature, top_p=top_p)
    generate_one_param = lambda x: generator.generate([x], max_gen_len=256, temperature=temperature, top_p=top_p)[0][len(x):]
    chat(generate_one_param)

    for result in results:
        print(result)
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
