#!/bin/zsh
#
#SBATCH -J is_lora8
#SBATCH --gpus 4
#SBATCH -C "thin"
#SBATCH -t 0-12:00:00
#


conda activate adapters

python run_mlm.py \
    --model_name_or_path microsoft/mdeberta-v3-base \
    --dataset_name cc100 \
    --dataset_config_name is \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir ./results/mdberta/lora8_is \
    --train_adapter \
    --adapter_config lora_config.json \
    --overwrite_output_dir \
    --max_seq_length 1024 \
    --max_train_samples 250000 \
    --max_eval_samples 25000 \




