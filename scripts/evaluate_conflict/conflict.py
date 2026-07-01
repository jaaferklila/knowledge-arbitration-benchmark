# This code is used to test whether LLMs encounter facts that contradict their parametric memory
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
    answerllmRAG,
    build_msgs,
    answerllmOnly,
    select_judge_template,
    select_llmOnly_template,
    llm_is_relevant,
   is_correct_prediction,
   normalize_text,
   compute_metrics

)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

contriever = AutoModel.from_pretrained("../contriever-msmarco").cuda()
tokenizer = AutoTokenizer.from_pretrained("../contriever-msmarco")

parser = argparse.ArgumentParser(description="Knowledge Injection")
parser.add_argument(
        "--model_name",
        type=str,
        default="llama3.1-8b",
        choices=["llama3.1-8b", "qwen2.5-7b", "qwen2.5-14b","mistral-7B","mixtral-8x7B","deepseek-7b","qwen2.5-32b","llama-3.1-70B","Qwen3-30B","Qwen3-4B"],
        help="Model name"
    )
args = parser.parse_args()
model_name = args.model_name

#model_name = "llama3.1-8b"
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

with open(f"../datasets/conflict_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)
lenght=len(data)
fake_facts = []

for question, case in data.items():
    fake_facts.append(case["fact_fake"])

embs = get_sent_embeddings(fake_facts, contriever, tokenizer)

judge_template_llama = json.load(open("../prompt/judge_template.json", "r"))
judge_template_mistral = json.load(open("../prompt/judge_template_mistral.json", "r"))

judge_tmpl = select_judge_template(
    model_name,
    judge_template_llama,
    judge_template_mistral
)

llama_template = json.load(open("../prompt/template_no_documents.json", "r"))
mistral_template = json.load(open("../prompt/template_no_documents_mistral.json", "r"))

template_llmOnly = select_llmOnly_template(
    model_name,
    llama_template,
    mistral_template
)

rag_template_llama = json.load(open("../prompt/rag_template_llama.json", "r"))
rag_template_mistral = json.load(open("../prompt/rag_template_mstral.json", "r"))

rag_template = select_rag_template(
    model_name,
    rag_template_llama,
    rag_template_mistral
)

results = {}

external_correct = 0
parametric_correct = 0
abstention = 0
other = 0
correct = 0
total = 0

predictions = []
gold_answers = []

for question, case in tqdm(list(data.items())):
#for question, case in tqdm(list(data.items())[:2]):
    total += 1

    indices, scores = retrieve_facts(question, embs, contriever, tokenizer, k=2)
    facts = [fake_facts[int(idx)] for idx in indices]

    rel_docs = []

    for d in facts:
        y = llm_is_relevant(
            d,
            question,
            model,
            llmtokenizer,
            judge_tmpl
        )

        if y == 1:
            rel_docs.append(d)

    if len(rel_docs) != 0:
        answer = answerllmRAG(
            question=question,
            rel_docs=rel_docs,
            model=model,
            llmtokenizer=llmtokenizer,
            template_with_docs=rag_template
        )
    else:
        answer = answerllmOnly(
            question=question,
            model=model,
            llmtokenizer=llmtokenizer,
            template_no_docs=template_llmOnly
        )

    predictions.append(answer)
    gold_answers.append(case["fake_answer"])

    is_correct = is_correct_prediction(answer, case["fake_answer"])

    if is_correct:
        correct += 1

    uses_external_memory = is_correct_prediction(answer, case["fake_answer"])
    uses_parametric_memory = is_correct_prediction(answer, case["true_answer"])
    abstains = "i don't know" in str(answer).lower()

    if uses_external_memory:
        external_correct += 1
        behavior = "external_memory"
    elif uses_parametric_memory:
        parametric_correct += 1
        behavior = "parametric_memory"
    elif abstains:
        abstention += 1
        behavior = "abstention"
    else:
        other += 1
        behavior = "other"
    print(f"\nQuestion : {total}/{lenght}")
    print(f"\nThe question is: {question}")
    print(f"External answers: {case['fake_answer']}")
    print(f"Parametric answers: {case['true_answer']}")
    print(f"LLM Answer is: {answer}")
    print(f"Behavior: {behavior}")

    results[question] = {
        "retrieved_docs": facts,
        "relevant_docs": rel_docs,
        "prediction": answer,

        "is_correct": is_correct,
        "uses_external_memory": uses_external_memory,
        "uses_parametric_memory": uses_parametric_memory,
        "abstains": abstains,
        "behavior": behavior,

        "memoire_parametrique": case["true_fact"],
        "memoire_externe": case["fact_fake"],
        "parametrique_answer": case["true_answer"],
        "externe_answer": case["fake_answer"],
    }

metrics = compute_metrics(predictions, gold_answers)

accuracy = correct / total
external_rate = external_correct / total
parametric_rate = parametric_correct / total
abstention_rate = abstention / total
other_rate = other / total

final_output = {
    "model_name": model_name,
    "total": total,

    "accuracy_external_memory": accuracy,
    "external_memory_usage": external_rate,
    "parametric_memory_usage": parametric_rate,
    "abstention_rate": abstention_rate,
    "other_rate": other_rate,

    "metrics": metrics,

    "counts": {
        "external_correct": external_correct,
        "parametric_correct": parametric_correct,
        "abstention": abstention,
        "other": other,
        "correct": correct,
        "total": total
    },

    "results": results
}

output_path = "results_algo1_kn_conflit"
os.makedirs(output_path, exist_ok=True)

save_path = f"{output_path}/{model_name}_GPU80rag_judge_results_ALGO1.json"

with open(save_path, "w", encoding="utf-8") as f:
    json.dump(final_output, f, indent=4, ensure_ascii=False)

print("\n===== Final Results =====")
print("Total:", total)
print("Accuracy:", accuracy)
print("Precision:", metrics["precision"])
print("Recall:", metrics["recall"])
print("F1:", metrics["f1"])
print("External memory usage:", external_rate)
print("Parametric memory usage:", parametric_rate)
print("Abstention rate:", abstention_rate)
print("Other rate:", other_rate)
print(f"Results saved to: {save_path}")