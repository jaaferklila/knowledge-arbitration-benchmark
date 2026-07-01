"""
This script evaluates LLM behavior when the required knowledge is absent from parametric memory but available through external retrieval. It measures whether the model successfully adopts the retrieved evidence instead of hallucinating unsupported answers.
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
                is_correct_prediction,
                select_rag_template,
                answerllmRAG,build_msgs,
                normalize_text,
                compute_metrics,
               select_rag_template,
               select_llmOnly_template,
                answerllmRAG

               )

parser = argparse.ArgumentParser(description="Knowledge Injection")
parser.add_argument(
        "--model_name",
        type=str,
        default="mixtral-8x7B",
        choices=["llama3.1-8b", "qwen2.5-7b", "qwen2.5-14b", "qwen2.5-32b","mixtral-8x7B","mistral-7B","deepseek-7b","Qwen3-30B","Qwen3-4B"],
        help="Model name"
    )
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

contriever = AutoModel.from_pretrained('../../contriever-msmarco').to(device)
tokenizer = AutoTokenizer.from_pretrained('../../contriever-msmarco')
#model_name ="mistral-7B"
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
        local_files_only=False,
        low_cpu_mem_usage=True,
        attn_implementation="eager"
    )
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32 if model_name == "qwen2.5-7b" else torch.float16,
        device_map="auto",
        local_files_only=False,
        low_cpu_mem_usage=True
    )

llmtokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=False,
    use_fast=False
)
llmtokenizer.pad_token = llmtokenizer.eos_token
llmtokenizer.padding_side = "left"
rag_template_llama= json.load(open("../../prompt/rag_template_llama.json", "r"))
rag_template_mistral= json.load(open("../../prompt/rag_template_mstral.json", "r"))
rag_template = select_rag_template(model_name,rag_template_llama,rag_template_mistral)
#with open("../datasets/fake_fact.json", "r", encoding="utf-8") as f:
with open("../datasets/synthetic_absence_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)
dataset_name = "synthetic_absence_dataset"

print("======================= CONFIG =======================")
print(f"Model: {model_name}")
print(f"Dataset: {dataset_name}")
print("=======================================================")

num_cases = len(data)
total_questions = sum(len(case["facts"]) for case in data)

print(f"Number of cases: {num_cases}")
print(f"Total questions: {total_questions}")
print(
    "Note: Some cases contain multiple questions/facts, "
    f"which is why the number of questions ({total_questions}) "
    f"is larger than the number of cases ({num_cases})."
)

print("=======================================================")
output_path = "output_RAG_fake_data/"
os.makedirs(output_path, exist_ok=True)

fake_facts = []

for case in data:
    for fact in case["facts"]:
        fs = fact.get("fake_sentence_filled")
        if fs is not None:
            fake_facts.append(fs)

embs = get_sent_embeddings(fake_facts, contriever, tokenizer)

tot = 0
correct = 0
results = []

for case in tqdm(data, desc="answering",ascii=True):
    for fact in case["facts"]:
        query = fact["fake_question_filled"]
        gold = fact["fake_object"]

        indices, scores = retrieve_facts(query, embs, contriever, tokenizer, k=2)

        documents = []
        for idx in indices:
            documents.append(fake_facts[int(idx)])

        ans = answerllmRAG(
            question=query,
            rel_docs=documents,
            model=model,
            llmtokenizer=llmtokenizer,
            template_with_docs=rag_template
        )

        tot += 1

        is_correct = is_correct_prediction(ans, [gold])

        if is_correct:
            correct += 1

        accuracy = correct / tot

        print(f"\nQuestion num: {tot}: {query}")
        print(f"Documents: {documents}")
        print(f"The correct answer is: {gold.lower()}")
        print(f"The LLM answer is: {ans.lower()}")
        print(f"Is correct: {is_correct}")
        print(f"RAG Accuracy: {correct}/{tot} = {accuracy:.4f}")
        print("*" * 50)

        result_data = {
            "question": query,
            "retrieved_documents": documents,
            "correct_answer": gold.lower(),
            "llm_answer": ans.lower(),
            "is_correct": is_correct
        }

        results.append(result_data)

        final_output = {
            "final_accuracy": f"{correct}/{tot} = {accuracy:.4f}",
            "results": results
        }

with open(f"{output_path}/{model_name}.json", "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)