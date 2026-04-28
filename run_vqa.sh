#!/bin/bash
#SBATCH --job-name=biovqa_qwen7b
#SBATCH -A bio240304-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:59:00
#SBATCH --output=/anvil/scratch/x-vobute1/vqa_logs/qwen7b_PMC13018909%j.out
#SBATCH --error=/anvil/scratch/x-vobute1/vqa_logs/qwen7b_PMC13018909%j.err

module load anaconda
conda activate biovqa

export HF_HOME=$SCRATCH/.hf_cache
export HF_TOKEN=

cd ~/bio_vqa_draft

python run.py \
    --archive /home/x-vobute1/bio_vqa_draft/PMC13018909.zip \
    --models qwen2.5-vl-7b \
    --output-dir $SCRATCH/vqa_results/qwen7b_pmc1_PMC13018909 \

