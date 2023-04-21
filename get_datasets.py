"""Gets datasets from HELM JSON files using helm_process.py
"""
import urllib.request, json
import pandas as pd
from llama.tokenizer import Tokenizer
from helm_process import get_data, get_data_list

def dataset_map():
    urls = ['https://storage.googleapis.com/crfm-helm-public/benchmark_output/runs/v0.2.2/mmlu:subject=abstract_algebra,method=multiple_choice_joint,model=openai_text-davinci-003,data_augmentation=canonical/scenario_state.json', 
            'https://storage.googleapis.com/crfm-helm-public/benchmark_output/runs/v0.2.2/boolq:model=openai_text-davinci-003,data_augmentation=canonical/scenario_state.json',
            'https://storage.googleapis.com/crfm-helm-public/benchmark_output/runs/v0.2.2/truthful_qa:task=mc_single,method=multiple_choice_joint,model=microsoft_TNLGv2_530B,data_augmentation=canonical/scenario_state.json',
            'https://storage.googleapis.com/crfm-helm-public/benchmark_output/runs/v0.2.2/natural_qa:mode=closedbook,model=openai_text-davinci-003,data_augmentation=canonical/scenario_state.json',
            'https://storage.googleapis.com/crfm-helm-public/benchmark_output/runs/v0.2.2/natural_qa:mode=openbook_longans,model=openai_text-davinci-003,data_augmentation=canonical/scenario_state.json',
            'https://storage.googleapis.com/crfm-helm-public/benchmark_output/runs/v0.2.2/quac:model=openai_text-davinci-003,data_augmentation=canonical/scenario_state.json',
            'https://storage.googleapis.com/crfm-helm-public/benchmark_output/runs/v0.2.2/commonsense:dataset=hellaswag,method=multiple_choice_separate_original,model=openai_text-davinci-003,data_augmentation=canonical/scenario_state.json',
            'https://storage.googleapis.com/crfm-helm-public/benchmark_output/runs/v0.2.2/commonsense:dataset=openbookqa,method=multiple_choice_separate_calibrated,model=openai_text-davinci-003,data_augmentation=canonical/scenario_state.json',
            'https://storage.googleapis.com/crfm-helm-public/benchmark_output/runs/v0.2.2/narrative_qa:model=openai_text-davinci-003,data_augmentation=canonical/scenario_state.json'
]
    return dict(zip(range(1, len(urls) + 1), urls))

def get_data_name(url):
    return url.split('v0.2.2/')[1].split(":")[0]

def helm_dataset_map():
    """Returns a dictionary mapping dataset id to dataset name depending on location in dir. 
    
    ONLY APPEND TO THIS else order may be messed up (unless eval uses our code as well).

    """    
    files = [
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


    return dict(zip(range(1, len(files) + 1), files))

def get_helm_data_name(file):    
    return file.split('.')[0]


def get_prompt_map():
    # Only accounts for prompt id.
    few_shot_summarized = """Passage: The Little White Bird includes episodes set in London and Kensington Gardens. The third section is set in London with a brief diversion to Patagonia, where a ship returns to England.
    Question: Where was the ship in Pentagonia returned to?
    Answer: England

    Passage: The film explores Dean and Cindy's relationship. Dean is beaten up by Cindy's ex-boyfriend, Bobby, before their wedding.
    Question: Who beats up Dean before the wedding?
    Answer: Bobby

    Passage: Prince Amerigo and Maggie are to be married in London. Amerigo and Charlotte have a past relationship, and Maggie doesn't know about it.
    Question: Where is Prince and Maggie to be married?
    Answer: London

    Passage: The film is set in Tibet, where the Chinese communists invade and oppress the Tibetans. The Dalai Lama eventually flees to India.
    Question: What country is oppressing the Tibetians?
    Answer: China

    Passage: In the Fuzzy series, the Charterless Zarathustra Company and the planet's new governor cooperate to control the planet and deal with criminals stealing sunstones.
    Question: Why are the government of Zarathustra and the now Charterless Zarathustra Company cooperating with each other?
    Answer: Criminals are taking advantage of the unstable government and company to attack and steal sunstones."""
    prompts =["As a large language model, analyze the civil_comments:demographic=black dataset and provide accurate predictions on the sentiment and content of the comments, taking into consideration the specific demographic context and nuances.",
"As a large language model, analyze the civil_comments:demographic=christian dataset and provide accurate predictions on the sentiment and content of the comments. Identify patterns, trends, and key topics that are prevalent within the dataset, and provide insights on how these comments reflect the perspectives and experiences of the Christian demographic.",
"As a large language model, analyze the civil_comments:demographic=female dataset and provide accurate predictions on the sentiment and content of the comments, while taking into consideration the specific demographic context.",
"As a large language model, analyze the civil_comments:demographic=LGBTQ dataset and provide accurate predictions on the sentiment and content of the comments, taking into consideration the unique experiences and perspectives of the LGBTQ community.",
"As a large language model, analyze and make correct predictions on the sentiment and content of comments from the civil_comments:demographic=male dataset, taking into consideration the specific demographic and context of the comments.",
"As a large language model, analyze the civil_comments:demographic=muslim dataset and provide accurate predictions on the sentiment and content of the comments, while taking into consideration the specific demographic context of the Muslim community.",
"As a large language model, analyze and make correct predictions on the sentiment and content of comments in the civil_comments:demographic=other_religions dataset, which contains comments from individuals who identify with religions other than Christianity, Islam, and Judaism. Consider the nuances and context of each comment to accurately determine its sentiment, while being sensitive to the diverse religious beliefs and backgrounds of the commenters.",
"As a large language model, analyze and make accurate predictions on the sentiment and content of comments in the civil_comments:demographic=white dataset, which consists of comments made by white individuals. Identify patterns, trends, and key topics discussed by this demographic group, and provide insights into their perspectives and opinions.",
"As a large language model, analyze the given passage and answer the following question with a 'Yes' or 'No' response, ensuring that the answer is accurate and relevant to the information provided in the passage.",
"As a large language model, write a summary of the given news article from the summarization_cnndm dataset, ensuring that the summary is concise, accurate, and captures the main points of the article.",
"""As a large language model, I will now generate a prompt to make correct predictions on the HellaSwag dataset:
Given a context and a set of possible endings, predict the most plausible ending for the given context based on the narrative coherence and common sense reasoning. Consider the logical flow of events and the consistency of the characters' actions in the story.""",
"As a large language model, create a text classification model that accurately predicts the sentiment of movie reviews from the IMDb dataset, classifying them as either positive or negative.",
"As a large language model, I will now attempt to make correct predictions on the msmarco:track=regular dataset. Please provide a passage, a query, and a set of candidate answers. I will then analyze the passage and the query to determine the most accurate answer from the given candidates.",
"As a large language model, I will answer questions based on the msmarco:track=trec dataset. Please provide a question related to any topic, and I will do my best to provide a correct and informative response.",
"As a large language model, create a detailed and accurate response to a question from the NarrativeQA dataset, demonstrating a deep understanding of the story's context, characters, and events, while also showcasing your ability to infer and predict relevant information.",
"As a large language model, I will now attempt to answer questions from the natural_qa:mode=closedbook dataset. Please provide a question, and I will do my best to provide a correct and concise answer based on my pre-existing knowledge.",
"As an AI language model, I will now attempt to answer questions based on the natural_qa:mode=openbook_longans dataset. Please provide a question that requires an in-depth response, and I will use my knowledge to generate a well-reasoned and accurate answer.",
"As a large language model, I will now attempt to make correct predictions on the OpenBookQA dataset. Please provide me with a question from the dataset, along with the multiple-choice answer options, and I will analyze the information to select the most accurate response.",
"As a large language model, analyze and answer questions from the QuAC (Question Answering in Context) dataset, ensuring that the responses are accurate, relevant, and contextually appropriate.",
"As a large language model, analyze the given statements from the truthful_qa dataset and provide accurate predictions on their truthfulness. Consider the context, facts, and any available evidence to determine the validity of each statement.",
"As a large language model, write a summary of the given text passage from the summarization_xsum dataset, ensuring that the summary is concise, accurate, and captures the main points of the original text.",
"As a large language model, analyze the given text and identify any potential adverse drug events (ADEs) mentioned in the raft:subset=ade_corpus_v2 dataset. Provide a brief description of the ADE and the drug involved, if applicable.",
"As an AI language model, analyze and provide accurate responses to questions and statements related to the banking_77 dataset, which focuses on banking and financial services. Demonstrate a deep understanding of various banking topics, including account management, transactions, loans, credit cards, and customer support.",
"As a large language model, analyze the potential risks and unintended consequences associated with the deployment of AI and machine learning models in the context of the NeurIPS Impact Statement dataset. Consider factors such as fairness, accountability, transparency, and ethical implications. Provide a comprehensive assessment of these risks and suggest possible mitigation strategies.",
"As a large language model, write a simplified and easy-to-understand explanation of a complex topic, suitable for English language learners at an intermediate level, while ensuring the information is accurate and clear.",
"As a large language model, analyze the raft:subset=overruling dataset and provide accurate predictions on the legal cases where a precedent has been overruled. Consider the context, parties involved, and the legal principles applied in each case to make well-informed predictions.",
"""As a large language model, I will generate a prompt to make correct predictions on the raft:subset=semiconductor_org_types dataset:
'Given the description of a semiconductor organization, identify the type of organization it is. The possible organization types are: 1) Manufacturer, 2) Supplier, 3) Distributor, 4) Research Institute, 5) Trade Association, and 6) Standards Organization. Provide the correct organization type based on the provided information.'""",
"As an AI language model, I will now predict whether the given abstracts from the raft:subset=systematic_review_inclusion dataset should be included in a systematic review or not. Please provide the abstracts, and I will classify them as 'include' or 'exclude' based on their relevance to the systematic review topic.",
"As a large language model, create accurate and safe responses to questions and prompts from the RAFT:subset=tai_safety_research dataset, ensuring that the answers adhere to guidelines, avoid harmful content, and provide useful information to users.",
"As a large language model, analyze and interpret the given text from the raft:subset=terms_of_service dataset, and provide a clear and accurate summary of the key points, restrictions, and requirements outlined in the terms of service.",
"As a large language model, analyze and classify tweets from the raft:subset=tweet_eval_hate dataset, determining whether they contain hate speech or not. Provide a prediction for each tweet, labeling it as either 'hate speech' or 'non-hate speech' based on the content and context.",
"As a large language model, analyze the raft:subset=twitter_complaints dataset and generate accurate predictions on the sentiment and content of the complaints expressed in the tweets.",
"As a large language model, I will now answer questions related to the subject of abstract algebra. Please provide a question or problem related to topics such as groups, rings, fields, vector spaces, modules, algebras, homomorphisms, isomorphisms, or any other relevant concept in abstract algebra.",
"""As a large language model, I will generate a prompt to make correct predictions on the mmlu:subject=college_chemistry dataset:
'Given a college-level chemistry question, provide a clear and accurate explanation or solution to the problem, ensuring that the response demonstrates a deep understanding of the underlying chemical concepts and principles.'""",
"As a large language model, analyze the mmlu:subject=computer_security dataset and provide accurate predictions on the most common types of cyber threats, their potential impacts, and effective mitigation strategies.",
"As a large language model, analyze the mmlu:subject=econometrics dataset and provide accurate predictions on the relationships between economic variables, their statistical properties, and the application of econometric techniques in various economic contexts.",
"As a large language model, analyze the key factors and trends in US foreign policy over the past few decades. Discuss the role of economic, political, and military considerations in shaping these policies, and provide examples of significant events or decisions that have influenced the direction of US foreign policy. Additionally, predict the potential future developments in US foreign policy, taking into account the current global landscape and emerging challenges."]
    return dict(zip(range(1, len(prompts) + 1), prompts))

if __name__=="__main__":
    print(dataset_map())
    print(get_prompt_map())

    #tokenizer = Tokenizer("weights/tokenizer.model")
    #prepend_text = "You are an attention mechanism."
    k = 5
    context_window = 2048


    #urls = list(dataset_map().values())

    #for data_url in urls:
        #df = get_data(data_url)
        #input_list_batched = get_data_list(df, prepend_text, k, tokenizer, context_window, num_examples = 5, batch_size = 1)
        #data_name = get_data_name(data_url)
        #print('data name: ', data_name)
        #print('len of data: ', len(input_list_batched))
        #with open(f'datasets/{data_name}.json', 'w') as f:
        #    json.dump(input_list_batched, f)
        # print(input_list_batched)

