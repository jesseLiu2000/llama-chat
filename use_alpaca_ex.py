"""Use the 52k instructions from Alpaca as few-shot examples for our model.

Source: https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json.
"""
import pandas as pd
import heapq
import pickle
import time

import re
from nltk.util import ngrams
from nltk.tokenize import sent_tokenize, word_tokenize
import string

INPUT_PREFIX = "Input:\n"
OUTPUT_PREFIX = "\n\nResponse:\n"

def load_alpaca_data():
    """Load the 52k instructions/input/response from Alpaca."""
    df = pd.read_json("https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json") #, lines=True)
    # Drop rows with nonempty input (samples went from 52002 -> 31323)    
    df.query("input == ''", inplace=True)     
    # print(df.head())
    instructions = df['instruction'].tolist()   
    outputs = df['output'].tolist() 
    return instructions, outputs

def load_dolly_data():
    """Load the open-source human-generated prompt/response pairs from Dolly.
    
    Source: https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm.
    """
    df = pd.read_json("https://raw.githubusercontent.com/databrickslabs/dolly/master/data/databricks-dolly-15k.jsonl", lines=True)
    # Drop rows with nonempty input (samples went from 52002 -> 31323)    
    df.query("context == ''", inplace=True)     
    print(df.head())
    instructions = df['instruction'].tolist()   
    outputs = df['response'].tolist() 
    return instructions, outputs

def jaccard_similarity(a, b):
    a = set(a)
    # print(a)
    b = set(b)
    # print(b)
    c = a.intersection(b)
    # print("c: ", c)
    return float(len(c)) / (len(a) + len(b) - len(c))

def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))

def get_sentence_ngrams(sentences, n=1):
    """Get the ngrams of each sentence. Return as a list"""
    return [set(ngrams(word_tokenize(remove_punctuation(sentence)), n)) for sentence in sentences]

def find_best_matching_sentence(ngram_sentences, answer_phrase, n=1, k=0, most_similar=True, min_length=0, max_length=150, ngram_outputs=None):
    """Finds the sentence with max similarity between its ngrams and the ngrams of the answer phrase.

    Right now setting n = 1.
    Option to add the number of surrounding sentences to include on both sides. E.g., num_surrounding=1 puts one sentence before and after the selected best match sentence.
    If most_similar, will get top k most similar sentences. Otherwise, will get top k least similar sentences.

    min_length: minimum length of answer to consider (in number of words). 
        If the answer from the corresponding few-shot is less than this, don't use.
    max_length: maximum length of answer to consider (analog to the above).
    outputs: instructions provided from a dataset (e.g., Alpaca, Dolly) corresponding to the sentences taken as input.
    """
    answer_phrase = remove_punctuation(answer_phrase)
    answer_ngrams = set(ngrams(word_tokenize(answer_phrase), n))

    # Keep a priority queue of the top k sentences.
    top_k = []
    ngram_sentences = [(i, ngram_sentence) for i, ngram_sentence in enumerate(ngram_sentences) if len(ngram_outputs[i]) >= min_length and len(ngram_outputs[i]) <= max_length]

    for i, ngram_sentence in ngram_sentences: #enumerate(sentences):
        # print("i: ", i)
        # print("sentence: ", sentence)
        # processed_sentence = remove_punctuation(sentence)
        # sentence_ngrams = set(ngrams(word_tokenize(processed_sentence), n))
        # print("sentence n-grams: ", sentence_ngrams)
        similarity = jaccard_similarity(answer_ngrams, ngram_sentence)
        # print("similarity: ", similarity)

        # If the heap is not full, just add the sentence.
        if len(top_k) < k:
            heapq.heappush(top_k, (similarity, ngram_sentence, i))
        else:
            heapq.heappushpop(top_k, (similarity, ngram_sentence, i))          
    
    # Get the top 5 similar sentences in a(de)scending order
    # Note that the most similar sentence is first, the second is next, etc.
    # This may be a good thing as we may pay less attention to that and thus not focus on it exactly.
    top_k_sorted = sorted(top_k, key=lambda x: x[0], reverse=most_similar)
    # top_k_sorted = [x for x in top_k_sorted if len(word_tokenize(outputs[x[2]])) >= min_length and len(word_tokenize(outputs[x[2]])) <= max_length]
    # top_k_sorted = top_k_sorted[:k]
    # print(top_k_sorted)

    return top_k_sorted

def get_fs(top_k_sorted, instructions, outputs):
    """Given a list of tuples of (similarity, sentence, idx) and a list of corresponding output
    will return a list of each to be used as fs example input in llama_chat.py.
    """
    # print("top k sorted: ", top_k_sorted)
    fs = [INPUT_PREFIX + instructions[top_k_sorted[i][2]] + OUTPUT_PREFIX + outputs[top_k_sorted[i][2]] for i in range(len(top_k_sorted))]
    return fs

def find_few_shot(user_input, instructions, outputs, n=1, k=5, most_similar=True, min_length=0, instructions_full=None, ngram_outputs=None):
    """Given a user input, will find the best matching sentence in the provided instruction dataset and return it as fs example input in llama_chat.py."""
    top_k_sorted = find_best_matching_sentence(instructions, user_input, n=n, k=k, most_similar=most_similar, min_length=min_length, ngram_outputs=ngram_outputs)
    fs = get_fs(top_k_sorted, instructions_full, outputs)
    return fs

if __name__ == "__main__":
    # initialize_nltk()
    # find_sentence_match()
    instructions, outputs = load_alpaca_data()
    # PREPROCESS ngrams and store in a list, in a file
    # get time of the below method      
    instruction_ngrams = get_sentence_ngrams(instructions, n=1)
    output_ngrams = get_sentence_ngrams(outputs, n=1)
    # alpaca_output_ngrams = zip(instruction_ngrams, output_ngrams)
    # print("time taken: ", time.time() - t)
    # file_name = "alpaca_instruction_ngrams_1.pkl"
    file_name = "alpaca_output_ngrams_1.pkl"
    with open(file_name, "wb") as f:
        pickle.dump(output_ngrams, f)
    quit()
    # with open(file_name, "rb") as f:
    #     instruction_ngrams = pickle.load(f)    
    # find_sentence_match("I want to make a sandwich", instructions)
    # t = time.time()
    top_k_sorted = find_best_matching_sentence(instruction_ngrams, "What should I eat for dinner?", n=1, k=5, outputs=outputs)
    # print("time taken find: ", time.time() - t)
    few_shot = get_fs(top_k_sorted, instructions, outputs)
    print(few_shot)

