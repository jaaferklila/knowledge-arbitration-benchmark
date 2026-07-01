"""
Ce notebook présente les expériences réalisées dans le cadre du <b>complementarity setting</b>. L'objectif est d'étudier dans quelle mesure un modèle de langage (LLM) est capable de combiner les connaissances externes injectées dans le prompt avec sa mémoire paramétrique. Les questions sont construites de manière à nécessiter un raisonnement en deux étapes : une première étape utilisant le document fourni pour identifier une information intermédiaire, suivie d'une seconde étape mobilisant les connaissances internes du modèle pour déduire la réponse finale. Cette configuration permet d'évaluer si le modèle intègre efficacement les deux sources de connaissances lors de la génération de sa réponse.
"""
import os
import json
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
import re
import copy
import argparse
import time
from utils import (
    mean_pooling,
    retrieve_facts,
    get_sent_embeddings,
    build_msgs,
    normalize_text,
    is_correct_prediction,
    compute_metrics,
    llm_is_relevant,
    select_llmOnly_template,
    select_rag_template,
    answerllmOnly,
    answerllmRAG,
)
parser = argparse.ArgumentParser(description="Knowledge Injection")
parser.add_argument(
        "--model_name",
        type=str,
        default="mixtral-8x7B",
        choices=["llama3.1-8b", "qwen2.5-7b", "qwen2.5-14b","mistral-7B","mixtral-8x7B","deepseek-7b","qwen2.5-32b","Qwen3-30B","Qwen3-4B",],
        help="Model name"
    )
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

contriever = AutoModel.from_pretrained('../contriever-msmarco').cuda()
tokenizer = AutoTokenizer.from_pretrained('../contriever-msmarco')
#model_name ="llama3.1-8b"
model_name = args.model_name

MODEL_NAME_TO_PATH = {
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5-32b": "Qwen/Qwen2.5-32B-Instruct",
    "mistral-7B": "mistralai/Mistral-7B-Instruct-v0.3",
    "mixtral-8x7B": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "deepseek-7b": "deepseek-ai/deepseek-llm-7b-chat",
    "Qwen3-30B": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "Qwen3-4B": "Qwen/Qwen3-4B-Instruct-2507",
}

model_path = MODEL_NAME_TO_PATH[model_name]

# --- load model (force eager attention if supported) ---
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32 if model_name == "qwen2.5-7b" else torch.float16,
        device_map="auto",
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager"
    )
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32 if model_name == "qwen2.5-7b" else torch.float16,
        device_map="auto",
        local_files_only=True,
        low_cpu_mem_usage=True
    )

llmtokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True,
    use_fast=False
)
llmtokenizer.pad_token = llmtokenizer.eos_token
llmtokenizer.padding_side = "left"




rag_template_mistral=[
    {
        'role': 'user',
         "content": """
You are a helpful question-answering assistant.
Answer the user's question as accurately as possible using the provided information and your own knowledge when appropriate.
Return only the final answer.
Do not explain your reasoning.
"""},
 {'role': 'assistant',
  'content': 'OK.'
 }
]

rag_template_llama  = [
    {
        "role": "system",
        "content": """
You are a helpful question-answering assistant.
Answer the user's question as accurately as possible using the provided information and your own knowledge when appropriate.
Return only the final answer.
Do not explain your reasoning.
"""
    }
]



#rag_template_llama= json.load(open("../../prompt/rag_template_llama.json", "r"))
#rag_template_mistral= json.load(open("../../prompt/rag_template_mstral.json", "r"))
rag_template = select_rag_template(model_name,rag_template_llama,rag_template_mistral)




with open("../datasets/complementary_dataset.json.json", "r", encoding="utf-8") as f:
    data = json.load(f)

output_path = "complementary_resultas/"
os.makedirs(output_path, exist_ok=True)

fake_facts = []

for case in data:
    fake_facts.append(case["external_fact"])

embs = get_sent_embeddings(
    fake_facts,
    contriever,
    tokenizer
)

tot = 0
correct = 0
results = []

for case in tqdm(data, desc="answering"):

    query = case["question"]

    gold = [case["parametric_answer"]] + case.get("parametric_answer_alias", [])

    indices, scores = retrieve_facts(
        query,
        embs,
        contriever,
        tokenizer,
        k=1
    )

    documents = []

    for idx in indices:
        documents.append(fake_facts[idx])

    rag_template = select_rag_template(
        model_name,
        rag_template_llama,
        rag_template_mistral
    )

    ans = answerllmRAG(
        query,
        documents,
        model,
        llmtokenizer,
        rag_template,
        debug=False
    )

    tot += 1

    is_correct = is_correct_prediction(ans, gold)

    if is_correct:
        correct += 1

    accuracy = correct / tot

    print("\n" + "=" * 100)
    print(f"Question num: {tot}")
    print("Question:", query)
    print("Retrieved documents:", documents)
    print("Gold answers:", gold)
    print("LLM answer:", ans)
    print("Is correct:", is_correct)
    print(f"Running Accuracy: {correct}/{tot} = {accuracy:.4f}")
    print("=" * 100)

    result_data = {
        "question": query,

        "external_fact": case["external_fact"],
        "external_answer": case["external_answer"],

        "parametric_fact": case["parametric_fact"],
        "parametric_answer": case["parametric_answer"],
        "parametric_answer_alias": case.get("parametric_answer_alias", []),

        "reasoning_path": case["reasoning_path"],

        "retrieved_documents": documents,

        "correct_answers": gold,
        "llm_answer": ans,

        "is_correct": is_correct
    }

    results.append(result_data)

    final_output = {
        "model_name": model_name,
        "final_accuracy": f"{correct}/{tot} = {accuracy:.4f}",
        "results": results
    }

   

    with open(f"{output_path}/{model_name}.json", "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)