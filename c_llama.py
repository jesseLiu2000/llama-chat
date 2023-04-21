from typing import Tuple
import os
import sys
import time
import json
from llama_cpp import Llama




from flask import Flask, render_template, request
import requests

app = Flask(__name__)
app.static_folder = 'static'
app.knowledge_pool = []
# recording_dict = {}
# count = 0


# alpaca_instructions_full, alpaca_outputs = load_alpaca_data() # load_dolly_data() 
# # Saving n-grams cuts time down by ~10 seconds
# file_name_instruction = "alpaca_instruction_ngrams_1.pkl"
# with open(file_name_instruction, "rb") as f:
#     instruction_ngrams = pickle.load(f) 
# file_name_output = "alpaca_output_ngrams_1.pkl"
# with open(file_name_output, "rb") as f:
#     output_ngrams = pickle.load(f) 

# def setup_model_parallel() -> Tuple[int, int]:
#     local_rank = int(os.environ.get("LOCAL_RANK", -1))
#     print("local_rank is ", local_rank)
#     world_size = int(os.environ.get("WORLD_SIZE", -1))

#     torch.distributed.init_process_group("nccl")
#     initialize_model_parallel(world_size)
#     torch.cuda.set_device(local_rank)

#     # seed must be the same in all processes
#     torch.manual_seed(1)
#     return local_rank, world_size


# def load(
#     ckpt_dir: str,
#     tokenizer_path: str,
#     local_rank: int,
#     world_size: int,
#     max_seq_len: int,
#     max_batch_size: int,
# ) -> LLaMA:
#     start_time = time.time()
#     checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
#     assert world_size == len(
#         checkpoints
#     ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
#     ckpt_path = checkpoints[local_rank]
#     print("Loading")
#     checkpoint = torch.load(ckpt_path, map_location="cpu")
#     with open(Path(ckpt_dir) / "params.json", "r") as f:
#         params = json.loads(f.read())

#     model_args: ModelArgs = ModelArgs(
#         max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
#     )
#     tokenizer = Tokenizer(model_path=tokenizer_path)
#     model_args.vocab_size = tokenizer.n_words
#     torch.set_default_tensor_type(torch.cuda.HalfTensor)
#     model = Transformer(model_args)
#     torch.set_default_tensor_type(torch.FloatTensor)
#     model.load_state_dict(checkpoint, strict=False)

#     generator = LLaMA(model, tokenizer)
#     print(f"Loaded in {time.time() - start_time:.2f} seconds")
#     return generator


# class main:
#     def __init__(
#             self,
#     ckpt_dir: str,
#     tokenizer_path: str,
#     temperature: float = 0.8,
#     top_p: float = 0.95,
#     max_seq_len: int = 2048,
#     max_batch_size: int = 32,
# ):
#         self.generator = None

#     def init(self):
#         if self.generator:
#             return self.generator
#         local_rank, world_size = setup_model_parallel()
#         if local_rank > 0:
#              sys.stdout = open(os.devnull, "w")

#     # local_rank = 0
#     # world_size = 1

#     #global generator
#         generator = load("/scratch/llama/weights/7B", "/scratch/llama/weights/tokenizer.model",
#         local_rank, world_size, 2048, 32
#         #ckpt_dir, tokenizer_path, local_rank,
#         #world_size, max_seq_len, max_batch_size
#     )
#         self.generator = generator
#         return generator

# MAIN = main("/scratch/llama/weights/65B", "/scratch/llama/weights/tokenizer.model")

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

# @app.route("/get")
# def get_bot_response():
#     temperature = 0.8
#     top_p = 0.95
#     dialogue_dict = {}
#     generator = MAIN.init()

#     prompts = request.args.get('msg')

#     results = generator.generate(prompts, max_gen_len=256, temperature=temperature, top_p=top_p)
#     generate_one_param = lambda x: generator.generate([x], max_gen_len=512, temperature=temperature, top_p=top_p)[0][len(x):]

#     result, local_knowledge_pool = our_fast_chat(
#         generate_one_param, prompts, instruction_ngrams, 
#         alpaca_outputs, alpaca_fs=True, most_similar=True, 
#         k=2, min_length=50, knowledge_pool=[], #app.knowledge_pool, 
#         instructions_full=alpaca_instructions_full, dialogue_chat=True,
#         ngram_outputs=output_ngrams
#     )
#     app.knowledge_pool = local_knowledge_pool
#     # Replace newlines with <br> so that the text is displayed properly
#     result = result.replace("\n", "<br>")

#     dialogue_dict['user'] = prompts
#     dialogue_dict['response'] = result

#     recording_dict[count] = dialogue_dict

#     with open("/scratch/zijie/github/recording/user_record_1.json", "a+") as f_write:
#         f_write.write(json.dumps(recording_dict, indent=5, ensure_ascii=False))

    
#     return result
llm = Llama(model_path="/scratch/zijie/github/weights/65B/ggml-model-q4_0.bin")

@app.route("/get")
def get_bot_response():
    # llm = Llama(model_path="/scratch/zijie/github/weights/65B/ggml-model-q4_0.bin")
    user_text = request.args.get('msg')
    prompts = "Question: " + user_text + "Answer: "
    output = llm(prompts, max_tokens=32, stop=["Q:", "\n"], echo=True)
    result = output["choices"][0]["text"].split("Answer: ")[-1]
    print(result)

    return result

if __name__ == "__main__":
    print("======== Flask Running =========")
    app.run()
