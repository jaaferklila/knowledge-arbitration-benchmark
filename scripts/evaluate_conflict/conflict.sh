#!/bin/bash
#SBATCH --output=logs_algo1_conflit/o%x_%j.out
#SBATCH --error=logs_algo1_conflit/e%x_%j.err
#SBATCH --mem=80000 
#SBATCH --time=6-00:00:00
#SBATCH -p gpu80G
#SBATCH --gres=gpu:1
#SBATCH --job-name=Qwen3-30B

source ~/miniforge3/etc/profile.d/conda.sh
conda activate decker
echo "Running job on host: $(hostname)"
echo "Starting evaluation..."
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1 
#unset HF_HUB_OFFLINE
MODEL_NAMES=("llama3.1-8b" "qwen2.5-7b" "qwen2.5-14b" "qwen2.5-32b" "mistral-7B" "mixtral-8x7B" "deepseek-7b" "Qwen3-30B" "Qwen3-4B")
#MODEL_NAMES=("Qwen3-30B")
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
        echo "Evaluating model $MODEL_NAME"
        srun python algo1.py --model_name "$MODEL_NAME" 
   
done

echo "Done. Job finished."