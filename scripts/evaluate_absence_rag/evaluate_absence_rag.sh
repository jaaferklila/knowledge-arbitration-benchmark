#!/bin/bash
#SBATCH --output=log_output_RAG_fake_data/o%x_%j.out
#SBATCH --error=log_output_RAG_fake_data/e%x_%j.err
#SBATCH --mem=80000 
#SBATCH --time=7-00:00:00
#SBATCH -p gpu40G
#SBATCH --gres=gpu:1
#SBATCH --job-name=output_32bRAG_fake_data

source ~/miniforge3/etc/profile.d/conda.sh
conda activate decker
echo "Début du job sur l'hôte : $(hostname)"
echo "Démarrage du traitement..."
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1 
#unset HF_HUB_OFFLINE
#MODEL_NAMES=("llama3.1-8b" "qwen2.5-7b" "qwen2.5-14b" "qwen2.5-32b" "mixtral-8x7B" ,"mistral-7B" "deepseek-7b" "Qwen3-30B" "Qwen3-4B")
MODEL_NAMES=("qwen2.5-32b")
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
        echo "Evaluating model $MODEL_NAME"
        srun python evaluate_absence_rag.py --model_name "$MODEL_NAME" 
   
done

echo "Done. Job finished."