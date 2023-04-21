"""Using our few-shot structure, we will make LLaMA into an effective chatbot that produces 
similar output to ChatGPT or Alpaca at a fraction of the cost without any fine-tuning.

We will start with 5 few-shot examples written by us (or GPT-4) relevant to general chats, 
wrapped in our method.
"""
from use_alpaca_ex import load_alpaca_data, load_dolly_data, find_few_shot, INPUT_PREFIX, OUTPUT_PREFIX
import pickle

# INPUT_PREFIX = "### Input:\n"
# OUTPUT_PREFIX = "\n\n### Response:\n"
KNOWLEDGE_PREFIX = "### Memory ###"
KNOWLEDGE_SUFFIX = "Let's possibly continue the above conversation" #"The above is part of my previous conversations." #"[[I'll keep the above in my memory.]]"

KNOWLEDGE_INPUT_PREFIX = "\nInput: "
KNOWLEDGE_OUTPUT_PREFIX = "\nResponse: "
# KINPUT_PREFIX = "Memory:\n"
# KOUTPUT_PREFIX = "\n\nResponse:\n"

POST_FS_SUFFIX = "\n\n----------"
# POST_FS_SUFFIX = ""


PREPEND_TEXT = "As an AI language model from the Fiber team, I am here to chat with you about any topic you'd like to discuss. Feel free to ask me questions, share your thoughts, or tell me about your interests. I'm here to help and engage in a meaningful conversation with you. My responses will be descriptive and thorough."
POST_FS_TEXT = "I'll remember the above example as I complete similar prompts"

def get_few_shot1():
    # ls = [
    #     {"context": "What is your favorite color?", "response": "My favorite color is blue."},
    #     {"context": "What is your favorite food?", "response": "My favorite food is pizza."},
    #     {"context": "What is your favorite movie?", "response": "My favorite movie is The Matrix."},
    #     {"context": "What is your favorite book?", "response": "My favorite book is The Lord of the Rings."},
    #     {"context": "What is your favorite TV show?", "response": "My favorite TV show is The Office."},
    # ]
    chatGPT_dialogue = [
        {
            "context": "Hi, can you tell me what the weather will be like tomorrow?",
            "response": "I'm sorry, I don't have access to real-time weather updates. Please check your local weather forecast for the most up-to-date information.",
            "memory": []
        },
        {
            "context": "Alright, can you recommend a good restaurant near me?",
            "response": "Sure, I can recommend some restaurants in your area. Where are you located?",
            "memory": ["Hi, can you tell me what the weather will be like tomorrow?\nResponse: I'm sorry, I don't have access to real-time weather updates. Please check your local weather forecast for the most up-to-date information."]
        },
        {
            "context": "I'm in New York City.",
            "response": "Great! There are many amazing restaurants in NYC. Do you have any specific cuisine in mind?",
            "memory": ["Hi, can you tell me what the weather will be like tomorrow?\nResponse: I'm sorry, I don't have access to real-time weather updates. Please check your local weather forecast for the most up-to-date information.", "Alright, can you recommend a good restaurant near me?\nResponse: Sure, I can recommend some restaurants in your area. Where are you located?"]
        },
        {
            "context": "I'm in the mood for Italian food.",
            "response": "I recommend checking out Carbone, L'Artusi, or Il Buco. They all have great Italian food and are highly rated by locals and tourists alike. Let me know if you need any further assistance.",
            "memory": ["Hi, can you tell me what the weather will be like tomorrow?\nResponse: I'm sorry, I don't have access to real-time weather updates. Please check your local weather forecast for the most up-to-date information.", "Alright, can you recommend a good restaurant near me?\nResponse: Sure, I can recommend some restaurants in your area. Where are you located?", "I'm in New York City.\nResponse: Great! There are many amazing restaurants in NYC. Do you have any specific cuisine in mind?"]
        }
    ]
    # TODO: separate memory questions from responses.
    join_mem = lambda x: KNOWLEDGE_INPUT_PREFIX.join(x)
    formatted_fs = []
    for d in chatGPT_dialogue:
        if len(d['memory']) == 0: # <= 1
            formatted_fs.append(f"{INPUT_PREFIX}{d['context']}{OUTPUT_PREFIX}{d['response']}")
        else:
            formatted_fs.append(f"{KNOWLEDGE_PREFIX}\n{KNOWLEDGE_INPUT_PREFIX}{join_mem(d['memory'])}\n{KNOWLEDGE_SUFFIX}\n\n{INPUT_PREFIX}{d['context']}{OUTPUT_PREFIX}{d['response']}")    
    return formatted_fs

def format_input(user_text, few_shot, knowledge_pool):
    """Given a user's input, format it with few-shot chat examples and our method."""
    few_shot = [PREPEND_TEXT + "\n" + fs + "\n" + POST_FS_TEXT + POST_FS_SUFFIX for fs in few_shot]
    knowledge_pool = KNOWLEDGE_PREFIX + "\n" + "\n".join(knowledge_pool) + "\n" + KNOWLEDGE_SUFFIX + "\n" if len(knowledge_pool) != 0 else ""
    user_text_formatted = INPUT_PREFIX + user_text + OUTPUT_PREFIX
    model_input = "\n\n".join(few_shot) + "\n" + PREPEND_TEXT + "\n" + knowledge_pool + "\n" + user_text_formatted
    return model_input, user_text_formatted

def get_output(model_output, few_shot):
    """Given the full output from the model, get only the model's response (remove our method).

    This will just be the text we want to display to the user in the chat window.
    """        
    current_output_only = model_output.split(OUTPUT_PREFIX)[len(few_shot) + 1]
    # Cut off when the model thinks it's done (when the model uses our POST_FS_TEXT or KNOWLEDGE_SUFFIX, it should be done).
    current_output_only = current_output_only.split(POST_FS_TEXT)[0].strip()
    # print("real current output_only: ", current_output_only)
    current_output_only_k = current_output_only.split(KNOWLEDGE_SUFFIX)[0].strip()  
    current_output_only_suff = current_output_only.split(POST_FS_SUFFIX)[0].strip()
    current_output_only_input = current_output_only.split(INPUT_PREFIX)[0].strip()
    current_output_only_kinput = current_output_only.split(KNOWLEDGE_INPUT_PREFIX)[0].strip()
    # current_output_newlines = current_output_only.split("\n\n").strip()
    # print(current_output_only.split("\n\n"))
    # current_output_newlines = "\n\n".join(current_output_only.split("\n\n")[:-1]).strip()


    # return shortest (argmin) of the three possibilities for cut-offs.
    shortest_output = min([current_output_only, current_output_only_k, current_output_only_suff, current_output_only_input, current_output_only_kinput], key=len)
    return shortest_output #current_output_only if len(current_output_only) < len(current_output_only_k) else current_output_only_k

def cycle_examples(current_output_only, few_shot):
    """Cycle through the few-shot examples. Always keep 5 examples, 
    but start using the user responses instead of the pre-generated few-shot.
    """
    # current_output_only = INPUT_PREFIX + current_output_only + OUTPUT_PREFIX
    few_shot = few_shot[1:] + [current_output_only]    
    # few_shot = [current_output_only]      
    # few_shot = few_shot + [current_output_only]     
    return few_shot

def change_knowledge(user_text, current_output_only, knowledge_pool):
    """Given current knowledge (what user has typed before) will update the knowledge pool.

    The knowledge pool is similar to the few-shot examples, but has a different structure.
    This should ensure that the model will not copy its previous responses.

    Change input/output prefixes. Will be after fs examples and before user input.
    """
    # Reverse sequence of words
    current_output_reversed = INPUT_PREFIX + user_text + KNOWLEDGE_OUTPUT_PREFIX + current_output_only #" ".join(current_output_only.split()[::-1])
    knowledge_pool.append(current_output_reversed) # Maybe don't need to add input question, just output?
    # Could also just jumble it up; keep basic stuff, but change order, make nonsensical? BACKWARDS???
    return knowledge_pool    

def one_chat(model, user_text, few_shot, knowledge_pool):
    """Goes through one full chat cycle.
    
    Given a model, user's input, and current few-shot example will
    format it according to our method and return the text to display to the user.

    Expects the model to take in a string and return a string.
    """
    model_input, user_text_formatted = format_input(user_text, few_shot, knowledge_pool)    
    # print("model_input:\n", model_input)
    output = model_input + model(model_input) + "\n" + POST_FS_TEXT
    # output = model(model_input) + "\n" + POST_FS_TEXT
    # print("full output:\n", output)

    current_output_only = get_output(output, few_shot)
    # print("\nChatbot:", current_output_only + "\n")
    formatted_output = user_text_formatted + current_output_only
    # few_shot = cycle_examples(formatted_output, few_shot)
    knowledge_pool = change_knowledge(user_text, current_output_only, knowledge_pool)
    return current_output_only, knowledge_pool

def chat(model, few_shot=get_few_shot1(), alpaca_fs=False, most_similar=True, k=5, min_length=0):
    """Chat with the model."""
    instructions, outputs = load_alpaca_data() # load_dolly_data()
    file_name = "alpaca_instruction_ngrams_1.pkl"
    with open(file_name, "rb") as f:
        instruction_ngrams = pickle.load(f)
    file_name = "alpaca_ngram_outputs_1.pkl"
    with open(file_name, "rb") as f:
        ngram_outputs = pickle.load(f)
    knowledge_pool = []
    while True:
        # Don't need knowledge pool here.
        print("knowledge_pool:", knowledge_pool)
        user_text = input("User: ")
        if alpaca_fs:
            few_shot = get_few_shot1() + find_few_shot(user_text, instruction_ngrams, outputs, n=1, k=k, most_similar=most_similar, min_length=min_length, instructions_full=instructions, ngram_outputs=ngram_outputs)
        print("few_shot:", few_shot)
        chat_output, knowledge_pool = one_chat(model, user_text, few_shot, knowledge_pool)

def our_fast_chat(generate_funct, user_text, instructions, outputs, alpaca_fs=True, most_similar=True, k=5, min_length=0, knowledge_pool=[], instructions_full=[], dialogue_chat=False, ngram_outputs=[]):
    """Chat with the model."""    
    if alpaca_fs:
        # Could change n or k?
        few_shot = find_few_shot(user_text, instructions, outputs, n=1, k=k, most_similar=most_similar, min_length=min_length, instructions_full=instructions_full, ngram_outputs=ngram_outputs)    
    if dialogue_chat:
        few_shot = get_few_shot1() + few_shot
    chat_output, knowledge_pool = one_chat(generate_funct, user_text, few_shot, knowledge_pool)
    return chat_output, knowledge_pool

if __name__ == "__main__":
    model = lambda x: "Heyyyy how are you doing?"
    chat(model, get_few_shot1(), alpaca_fs=True, most_similar=True, k=2, min_length=0)
    quit()
    few_shot = get_few_shot1()
    user_text = "What is your name?"
    model_input, user_text_formatted = format_input(user_text, few_shot)
    print(model_input)

    output = model_input + "I am Mr.John David Fiber" + POST_FS_TEXT #model(model_input)
    current_output_only = get_output(output, few_shot)
    print("Chatbot: ", current_output_only)
    formatted_output = user_text_formatted + current_output_only
    few_shot = cycle_examples(formatted_output, few_shot)
    # few_shot = cycle_examples(user_text_formatted + OUTPUT_PREFIX + current_output_only, few_shot)

    new_user_text = "How old are you?"
    new_model_input = format_input(new_user_text, few_shot)
    print(new_model_input)

    possible_prompt = [
        "As an AI language model, I am here to chat with you about any topic you'd like to discuss. Feel free to ask me questions, share your thoughts, or tell me about your interests. I'm here to help and engage in a meaningful conversation with you."
    ]

