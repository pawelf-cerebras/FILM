#!/bin/bash
export NCCL_IGNORE_DISABLED_P2P=1
model_path="/home/pawelf/mlf2/models/Llama-3.1-8B-Instruct/snapshots/8c22764a7e3675c50d4c7c9a4edb474456022b16"
results_path="./results/Llama-3.1-8B-Instruct/"
parallel_size=1

# Short tasks
cd short_tasks

python ../vllm_inference/vllm_inference.py \
    --model_path $model_path \
    --testdata_file gsm8k_8shot.jsonl \
    --testdata_folder ./prompts/ \
    --output_folder $results_path \
    --max_length 2048 \
    --tensor_parallel_size $parallel_size

python ../vllm_inference/vllm_inference.py \
    --model_path $model_path \
    --testdata_file math_4shot.jsonl \
    --testdata_folder ./prompts/ \
    --output_folder $results_path \
    --max_length 2048 \
    --tensor_parallel_size $parallel_size

python ../vllm_inference/vllm_inference.py \
    --model_path $model_path \
    --testdata_file csqa_0shot.jsonl \
    --testdata_folder ./prompts/ \
    --output_folder $results_path \
    --max_length 128 \
    --tensor_parallel_size $parallel_size

python evaluation.py

cd ..