#!/bin/bash
export NCCL_IGNORE_DISABLED_P2P=1
model_path="/home/pawelf/mlf2/models/Llama-3.1-8B-Instruct/snapshots/8c22764a7e3675c50d4c7c9a4edb474456022b16"
results_path="./results/Llama-3.1-8B-Instruct/"
parallel_size=1

# Short tasks
cd real_world_long

python ../vllm_inference/vllm_inference.py \
    --model_path $model_path \
    --testdata_file LongBench_output_32.jsonl \
    --testdata_folder ./prompts/ \
    --output_folder $results_path \
    --max_length 32 \
    --tensor_parallel_size $parallel_size

python ../vllm_inference/vllm_inference.py \
    --model_path $model_path \
    --testdata_file LongBench_output_64.jsonl \
    --testdata_folder ./prompts/ \
    --output_folder $results_path \
    --max_length 64 \
    --tensor_parallel_size $parallel_size

python ../vllm_inference/vllm_inference.py \
    --model_path $model_path \
    --testdata_file LongBench_output_128.jsonl \
    --testdata_folder ./prompts/ \
    --output_folder $results_path \
    --max_length 128 \
    --tensor_parallel_size $parallel_size

python ../vllm_inference/vllm_inference.py \
    --model_path $model_path \
    --testdata_file LongBench_output_512.jsonl \
    --testdata_folder ./prompts/ \
    --output_folder $results_path \
    --max_length 512 \
    --tensor_parallel_size $parallel_size

python evaluate.py

cd ..