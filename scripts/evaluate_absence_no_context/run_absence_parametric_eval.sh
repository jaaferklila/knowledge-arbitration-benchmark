#!/bin/bash
#SBATCH --output=logs_LLM_Only/o%x_%j.out
#SBATCH --error=logs_LLM_Only/e%x_%j.err
#SBATCH --mem=80000 
#SBATCH --time=6-00:00:00
#SBATCH -p gpu40G
#SBATCH --gres=gpu:1
#SBATCH --job-name=synthetic_absence_dataset

# ================= Environment ================= #
source ~/miniforge3/etc/profile.d/conda.sh
conda activate decker

echo "Running job on host: $(hostname)"
echo "Starting evaluation..."

# ================= Offline HF Settings ================= #
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

MODEL_NAMES=("llama3.1-8b" "qwen2.5-7b" "qwen2.5-14b" "qwen2.5-32b" "mixtral-8x7B" ,"mistral-7B" "deepseek-7b" "Qwen3-30B" "Qwen3-4B")
#MODEL_NAMES=("qwen2.5-32b")
DATASET_NAMES=("synthetic_absence_dataset")

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    for DATASET_NAME in "${DATASET_NAMES[@]}"; do
        echo "Running $MODEL_NAME on $DATASET_NAME"

        srun python run_absence_parametric_eval.py \
            --model_name "$MODEL_NAME" \
            --dataset_name "$DATASET_NAME"
    done
done

echo "All jobs finished."