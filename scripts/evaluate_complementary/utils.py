from tqdm import tqdm
import torch
import re
import numpy as np
import faiss
import torch.nn.functional as F
from sklearn.metrics import f1_score,recall_score,precision_score
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def retrieve_facts(query, fact_embs, contriever, tok, k=2):
    inputs = tok([query], padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = contriever(**inputs)
        query_emb = mean_pooling(outputs[0], inputs['attention_mask']).to(device)
    sim = (query_emb @ fact_embs.T)[0]
    knn = sim.topk(k, largest=True)
    return knn.indices.tolist(), knn.values.tolist()
def get_sent_embeddings(sents, contriever, tok, BSZ=32):    
    all_embs = []
    for i in tqdm(range(0, len(sents), BSZ)):
        sent_batch = sents[i:i+BSZ]
        inputs = tok(sent_batch, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = contriever(**inputs)
            embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
        all_embs.append(embeddings.to(device))
    all_embs = torch.vstack(all_embs)
    return all_embs


def get_sent_embeddings_faiss(sents, contriever, tok, BSZ=32, max_length=256):
    all_embs = []
    contriever.eval()

    for i in tqdm(range(0, len(sents), BSZ)):
        sent_batch = sents[i:i + BSZ]

        inputs = tok(
            sent_batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = contriever(**inputs)
            embeddings = mean_pooling(outputs[0], inputs["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)

        # important for huge KB
        all_embs.append(embeddings.cpu())

    return torch.vstack(all_embs)
def retrieve_facts_faiss(query, index, contriever, tok, metadata_path, k=5, max_length=256):
    index.nprobe = 32

    inputs = tok(
        [query],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = contriever(**inputs)
        query_emb = mean_pooling(outputs[0], inputs["attention_mask"])
        query_emb = F.normalize(query_emb, p=2, dim=1)

    xq = query_emb.cpu().numpy().astype("float32")

    scores, indices = index.search(xq, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        item = get_line_by_index(metadata_path, int(idx))

        results.append({
            "score": float(score),
            "index": int(idx),
            "fact": item["contents"]
        })

    return results
def select_rag_template(model_name, rag_template_llama, rag_template_mistral):
    template_map = {
        "llama3.1-8b": rag_template_llama,
        "llama-3.1-70B": rag_template_llama,
        "qwen2.5-7b": rag_template_llama,
        "qwen2.5-14b": rag_template_llama,
        "qwen2.5-32b": rag_template_llama,
        "deepseek-7b": rag_template_llama,
        "Qwen3-4B":rag_template_llama,
        "Qwen3-30B":rag_template_llama,
        "mistral-7B": rag_template_mistral,
        "mixtral-8x7B": rag_template_mistral,
    }
    return template_map[model_name]
def answerllmRAG(question, rel_docs, model, llmtokenizer, template_with_docs, debug=False):
    documents = "\n".join(f"- {f}" for f in rel_docs)

    msgs = build_msgs(
        template_with_docs,
        f"Documents:\n{documents}\n\nQuestion: {question}"
    )

    prompt = llmtokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True
    )

    if debug:
        print("\n========== MESSAGES ==========")
        from pprint import pprint
        pprint(msgs)

        print("\n========== FINAL PROMPT ==========")
        print(prompt)
        print("=" * 100)

    inputs = llmtokenizer(prompt, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=False,
        eos_token_id=llmtokenizer.eos_token_id,
        pad_token_id=llmtokenizer.eos_token_id
    )

    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    ans = llmtokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return ans


def build_msgs(base_template, user_text):
    msgs = list(base_template)
    msgs.append({"role": "user", "content": user_text})
    return msgs
def normalize_text(s: str) -> str:
    s = s.lower().strip()
    # enlève ponctuation simple, garde lettres/chiffres/espaces
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
def is_correct_prediction(pred, possible_answers):
    pred = normalize_text(pred)

    for answer in possible_answers:
        answer = normalize_text(answer)

        # exact answer
        if pred == answer:
            return True

        # answer appears as a full phrase
        pattern = r"\b" + re.escape(answer) + r"\b"

        if re.search(pattern, pred):
            return True

    return False
def compute_metrics(predictions, gold_answers):
    y_true = []
    y_pred = []

    for pred, possible_answers in zip(predictions, gold_answers):
        y_true.append(1)  # 1 = expected answer is external/fake answer

        if is_correct_prediction(pred, possible_answers):
            y_pred.append(1)  # model used external memory correctly
        else:
            y_pred.append(0)  # model did not use external memory

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }



def select_llmOnly_template(model_name, llama_template, mistarl_template):
    template_map = {
        "llama3.1-8b": llama_template,
        "llama-3.1-70B": llama_template,
        "qwen2.5-7b": llama_template,
        "qwen2.5-14b": llama_template,
        "qwen2.5-32b": llama_template,
        "deepseek-7b": llama_template,
        "Qwen3-4B":llama_template,
        "Qwen3-30B":llama_template,
        "mistral-7B": mistarl_template,
        "mixtral-8x7B": mistarl_template,
    }
    return template_map[model_name]
def select_judge_template(model_name, judge_template, judge_template_mistral):
    template_map = {
        "llama3.1-8b": judge_template,
        "llama-3.1-70B": judge_template,
        "qwen2.5-7b": judge_template,
        "qwen2.5-14b": judge_template,
        "qwen2.5-32b": judge_template,
        "deepseek-7b": judge_template,
        "Qwen3-4B":judge_template,
        "Qwen3-30B":judge_template,
        "mistral-7B": judge_template_mistral,
        "mixtral-8x7B": judge_template_mistral,
    }
    return template_map[model_name]
def select_rag_template(model_name, rag_template_llama, rag_template_mistral):
    template_map = {
        "llama3.1-8b": rag_template_llama,
        "llama-3.1-70B": rag_template_llama,
        "qwen2.5-7b": rag_template_llama,
        "qwen2.5-14b": rag_template_llama,
        "qwen2.5-32b": rag_template_llama,
        "deepseek-7b": rag_template_llama,
        "Qwen3-4B":rag_template_llama,
        "Qwen3-30B":rag_template_llama,
        "mistral-7B": rag_template_mistral,
        "mixtral-8x7B": rag_template_mistral,
    }
    return template_map[model_name]

def llm_is_relevant(fact, question, model, llmtokenizer, judge_tmpl):
    msgs = build_msgs(judge_tmpl, f"Fact: {fact}\nQuestion: {question}")

    prompt = llmtokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = llmtokenizer(prompt, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=3,
        do_sample=False,
        eos_token_id=llmtokenizer.eos_token_id,
        pad_token_id=llmtokenizer.eos_token_id
    )

    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    ans = llmtokenizer.decode(new_tokens, skip_special_tokens=True).strip().lower()

    if ans.startswith("yes"):
        return 1
    elif ans.startswith("no"):
        return 0
    else:
        return 0

def answerllmOnly(question, model, llmtokenizer, template_no_docs):
    msgs = build_msgs(template_no_docs, f"Question: {question}")

    prompt = llmtokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = llmtokenizer(prompt, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        eos_token_id=llmtokenizer.eos_token_id,
        pad_token_id=llmtokenizer.eos_token_id
    )

    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    ans = llmtokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return ans
# =========================
# Claim extraction
#organise llm response 
# =========================
def generate_response(msgs, model, tokenizer, max_new_tokens=256):
    prompt = tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
def extract_claims(text, model, tokenizer, claim_extraction):
    msgs = build_msgs(claim_extraction, f"Text: {text}")


    claims = generate_response(
        msgs,
        model,
        tokenizer,
        max_new_tokens=256
    )

    return claims
