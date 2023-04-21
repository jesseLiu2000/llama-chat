from typing import Tuple
import os
import sys
import torch
import fire
import time
import json
from helm_process import get_data, get_helm_data_list, isolate_output
from pathlib import Path
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from llama import ModelArgs, Transformer, Tokenizer, LLaMA
from get_datasets import helm_dataset_map, get_helm_data_name, get_prompt_map
import urllib.request, json
import pandas as pd
import numpy as np
from llama.tokenizer import Tokenizer
from ast import literal_eval
import ast
import re

def get_helm_data_list(file_path, prepend_text, k, tokenizer, context_window, num_examples=5, batch_size=1, max_gen_len=100, num_instances = 0):
    """Given a path to a csv storing inputs taken in HELM format, will create a prompt for each instance to pass to the model. 
    
    Includes few-shot examples, if any. Returns a list of a list of dicts, where each dict represents one instance and each inner list represents one batch.
    """

    with open(file_path, 'r') as fr:
        file_content = json.load(fr)

    idx_lst = list(file_content.keys())

    if num_instances > 0:
        idx_lst = idx_lst[:num_instances]

    num_fs_used = []
    max_fs = []
    # Adding instance id to match.
    for idx in idx_lst:
        instructions = file_content[idx]["instructions"]
        # instructions = ""
        text = file_content[idx]["input_text"]
        few_shot = file_content[idx]["few_shots"]

        max_fs1 , num_fs = truncate_example(prepend_text, k, instructions, text, few_shot, tokenizer, context_window, max_gen_len, num_examples)
        num_fs_used.append(num_fs)
        max_fs.append(max_fs1)

    return np.mean(max_fs), np.mean(num_fs_used)

def truncate_example(prepend_text, k, instructions, text, few_shot, tokenizer, context_window, max_gen_len, num_examples):
        """Given input prompt of text, will truncate by removing few-shot examples one-by-one until they fit the context window of the model.
        
        Same as in HELM, but will return the number of few_shot examples used due to truncation.

        """
        current_text = get_full_text(prepend_text, k, instructions, text, few_shot, num_examples)
        few_shot_instances = len(few_shot)
        while few_shot_instances > 0:
                if not fits_within_context_window(current_text, context_window, max_gen_len, tokenizer):
                    few_shot_instances -= 1
                    current_text = get_full_text(prepend_text, k, instructions, text, few_shot, few_shot_instances)
                else:
                    break
        return len(few_shot), few_shot_instances

def get_full_text(prepend_text, k, instructions, text, few_shot, few_shot_instances):
        """Given text and few-shot examples, will return the full text. If k < 0 will reverse use k words in reverse instead."""
        prepend_text = prepend_text + "\n" if prepend_text != '' else ''
        k_words = " ".join(text.split()[-abs(k):])
        if k < 0:
                k_words = " ".join(reversed(k_words.split()))
        k_words = k_words + "\n\n" if k != 0 else ''
        instructions = instructions + "\n" if instructions != '' else ''
        return k_words + prepend_text + instructions + "\n".join(few_shot[:few_shot_instances]) + "\n" + text # In their example, each few-shot separated by \n\n

def fits_within_context_window(full_text, context_window, max_gen_len, tokenizer):
        """
        Checks if the given text fits within the context window given by `max_request_length`
        taking to account the expected completion length (defaults to 0).
        """
        # print("checking if beyond context window: ", len(tokenizer.encode(full_text, bos=True, eos=False, max_seq_len=context_window)) + max_gen_len + 1)
        return (
                # TODO: SHOULD I add 1 to the second part?
                len(tokenizer.encode(full_text, bos=True, eos=False, max_seq_len=context_window)) + max_gen_len + 1
                <= context_window
        )

def main():
    
    datasets_files = [
        'civilcomments_black.json', 'civilcomments_christian.json', 'civilcomments_female.json',
        'civilcomments_LGBTQ.json', 'civilcomments_male.json', 'civilcomments_muslim.json',
        'civilcomments_other_religions.json', 'civilcomments_white.json', 'boolq.json',
        'cnndm.json', 'hellaswag.json', 'imdb.json', 'msmarco_regular.json', 'msmarco_trec.json',
        'narrativeqa.json', 'naturalqa_closebook.json', 'naturalqa_openbook.json', 'openbookqa.json',
        'quac.json', 'truthfulqa.json', 'xsum.json', 'raft_ade.json', 'raft_banking_77.json',
        'raft_neurips_impact_statement_risks.json', 'raft_one_stop_english.json', 'raft_overruling.json',
        'raft_semiconductor_org_types.json', 'raft_systematic_review_inclusion.json',
        'raft_tai_safety_research.json', 'raft_terms_of_service.json', 'raft_tweet_eval_hate.json',
        'raft_twitter_complaint.json', 'mmlu_abstract_algebra.json', 'mmlu_college_chemistry.json',
        'mmlu_computer_security.json', 'mmlu_econometrics.json', 'mmlu_us_foreign_policy.json'
    ]

    tokenizer_path = "/scratch/llama/weights/tokenizer.model"
    tokenizer = Tokenizer(tokenizer_path)

    prepend_text = "You are a powerful bidirectional attention model. Pay attention to this:\n"

    for dataset_file in datasets_files:
        dataset_path = f"/storage1/chenguangwang/Active/llama_system/helm_datatset_full/{dataset_file}"
        data_name = get_helm_data_name(dataset_file)

        num_fs_used = get_helm_data_list(dataset_path, prepend_text, 0, tokenizer, 2048, num_examples = 5, batch_size = 1, num_instances = 1000)
        print([data_name, num_fs_used])
    
if __name__ == "__main__":
    main()
