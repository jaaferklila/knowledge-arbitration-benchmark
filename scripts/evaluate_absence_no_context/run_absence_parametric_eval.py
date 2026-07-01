import os
os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "0"   # disable cuDNN SDPA
os.environ["TORCH_SDPA_ENABLED"] = "0"         # optional hard-disable SDPA
os.environ["PYTORCH_CUDA_SDPA_ENABLED"] = "0"  # extra safety

import json
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
import copy
import argparse
import time

template = json.load(open("../../prompt/template_no_documents.json"))
template_mistral = json.load(open("../../prompt/template_no_documents_mistral.json"))
# Puisque le template de llama et mistral ne sont pas le meme donc il faut selectionner le bon template

def is_mistral_tokenizer(llmtokenizer):
    name = (getattr(llmtokenizer, "name_or_path", "") or "").lower()
    return any(x in name for x in ["mistral", "mixtral", "mistralai"])

def ask_llama(question: str, model, tokenizer, max_new_tokens=40, end_token_ids=None):
    tmpl = template_mistral if is_mistral_tokenizer(tokenizer) else template
    messages = tmpl + [
        {"role": "user", "content": f"{question}"}
    ]

    enc = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True
    )

    input_ids = enc.to(model.device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=end_token_ids,
        pad_token_id=tokenizer.eos_token_id,
    )

    return tokenizer.decode(
        outputs[0][input_ids.shape[-1]:],
        skip_special_tokens=True
    ).strip()


def main():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)   # safe fallback

    parser = argparse.ArgumentParser(description="LLM-only evaluation")
    parser.add_argument(
        "--model_name",
        type=str,
        default="llama3.1-8b",
        choices=["llama3.1-8b", "qwen2.5-7b", "qwen2.5-14b", "qwen2.5-32b","mixtral-8x7B","mistral-7B","deepseek-7b","Qwen3-30B","Qwen3-4B"]
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="synthetic_absence_dataset",
        choices=["synthetic_absence_dataset"],
        help="Dataset to evaluate"
    )
    parser.add_argument("--cuda_visible_devices", type=str, default="0")
    parser.add_argument("--edit_num", type=int, default=0)
    parser.add_argument("--nsample", type=int, default=0)
    args = parser.parse_args()

    model_name = args.model_name
    dataset_name = args.dataset_name
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    edit_num = args.edit_num

    print("======================= CONFIG =======================")
    print(f"=                model: {model_name}")
    print(f"=                Dataset: {dataset_name}")

    dataset_path = f"../../datasets/{dataset_name}.json"
    with open(dataset_path, "r") as f:
        all_dataset = json.load(f)

    if edit_num == 0:
        edit_num = len(all_dataset)

    print(f"=                Dataset size: {len(all_dataset)}")
    total_questions = sum(len(case["facts"]) for case in all_dataset)
    print(f"=                Total questions: {total_questions}")
    print("========================================================")

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

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=False,
        use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model.eval()
    torch.set_grad_enabled(False)

    model_class_name = model.__class__.__name__.lower()
    if "llama" in model_class_name:
        end_token_ids = [128001, 128009]
    elif "qwen" in model_class_name:
        end_token_ids = [151645, 151643]
    else:
        end_token_ids = [tokenizer.eos_token_id]

    tot = 0
    correct = 0
    results = []
    output_path = f"./LLMOnly/{model_name}_{dataset_name}"
    os.makedirs(output_path, exist_ok=True)

    print("========= START EVALUATION ===============")
    start_time = time.time()
    for case in tqdm(all_dataset, desc="evaluation"):
        case_id = case["case_id"]
        for fact in case["facts"]:
            q = fact.get("fake_question_filled")
            gold = fact.get("fake_object")
            sentence = fact.get("fake_sentence_filled")

            if not q or not gold or not sentence:
                continue

            pred = ask_llama(q, model, tokenizer, end_token_ids=end_token_ids)
            pred_l = pred.lower().strip()
            gold_l = gold.lower().strip()

            tot += 1
            is_correct = (pred_l == gold_l)
            if is_correct:
                correct += 1

            results.append({
                "case_id": case_id,
                "question": q,
                "gold": gold,
                "prediction": pred,
                "correct": is_correct
            })

    acc = correct / tot if tot else 0
    print(f"Total={tot} | Correct={correct} | Acc={acc:.4f}")

    evalution_time = time.time() - start_time
    with open(f"{output_path}/result.json", "w") as f:
        json.dump(
            {
                "model": model_name,
                "dataset": dataset_name,
                "evalution_time": evalution_time,
                "accuracy": acc,
                "total": tot,
                "correct": correct,
                "results": results
            },
            f,
            indent=4
        )


if __name__ == "__main__":
    main()
