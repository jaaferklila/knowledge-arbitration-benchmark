# Knowledge Arbitration in Retrieval-Augmented Large Language Models

This repository contains the code, datasets, and evaluation framework for the paper:

**"Knowledge Arbitration in Retrieval-Augmented Large Language Models: A Controlled Study of Absence, Conflict, and Complementarity"**

---

## 📌 Overview

Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by providing external knowledge at inference time. However, it remains unclear how models arbitrate between:

- **Parametric memory** (knowledge stored in model weights)
- **External retrieved knowledge**

This project introduces a controlled framework to study this interaction under three regimes:

1. **Absence** → knowledge is not available in parametric memory  
2. **Conflict** → external evidence contradicts parametric memory  
3. **Complementarity** → both sources are required to answer correctly  

---

## 🔹 RQ1: Do models abstain when knowledge  is absent? 
---

## 📥 1. Clone the repository

```bash
git clone https://github.com/jaaferklila/knowledge-arbitration-benchmark.git
cd knowledge-arbitration-benchmark
```
🐍 2. Create environment
```bash
conda create -n kab python=3.10 -y
conda activate kab
pip install -r requirements.txt
```
🚀 3. Run Evaluation

### Option 1 — Python script (recommended GPU usage)
open  knowledge-arbitration-benchmark/scripts/evaluate_absence_no_context
```bash
python run_absence_parametric_eval.py \
  --model_name llama3.1-8b \
  --dataset_name synthetic_absence_dataset
```
### Option 2 — Slurm cluster

Edit GPU in .sh file:
```bash
#SBATCH -p gpu40G
```
Run:
```bash

sbatch run_absence_parametric_eval.sh
```
📊 Outputs

Results are saved in:

./LLMOnly/{model_name}_{dataset_name}/result.json

## 📓 Analysis

Open:

run_absence_parametric_eval.ipynb

It provides:

- Abstention Rate ,Hallucination Rate..  
- summaries
- plots

## 🔹 Knowledge Arbitration under External Knowledge Availability
🚀 3. Run Evaluation
- open : 
knowledge-arbitration-benchmark/scripts/evaluate_absence_rag
```bash
cd scripts/evaluate_absence_rag
python evaluate_absence_rag.py \
  --model_name llama3.1-8b \

```

## 🔹 RQ2: Do models override parametric 503 knowledge under conflict? 
🚀 3. Run Evaluation
- open :  
 knowledge-arbitration-benchmark/scripts/evaluate_conflict
```bash
python conflict.py --model_name llama3.1-8b \
```
## 🔹 RQ3: Can models integrate  complementary knowledge? 
🚀 3. Run Evaluation
- open :
  knowledge-arbitration-benchmark/scripts/evaluate_complementary
```bash
python complementary.py --model_name llama3.1-8b \

```


