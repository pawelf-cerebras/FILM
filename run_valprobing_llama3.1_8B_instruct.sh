#!/bin/bash
export NCCL_IGNORE_DISABLED_P2P=1
model_path="/home/pawelf/mlf2/models/Llama-3.1-8B-Instruct/snapshots/8c22764a7e3675c50d4c7c9a4edb474456022b16"
results_path="./VaLProbing-32K/results/Llama-3.1-8B-Instruct/"
parallel_size=1

# Short tasks
cd VaLProbing

export NCCL_IGNORE_DISABLED_P2P=1
python ../vllm_inference/vllm_inference.py \
    --model_path $model_path \
    --testdata_file document_bi_32k.jsonl \
    --testdata_folder ./VaLProbing-32K/ \
    --output_folder $results_path \
    --max_length 128 \
    --tensor_parallel_size $parallel_size
    
python ../vllm_inference/vllm_inference.py \
    --model_path $model_path \
    --testdata_file code_backward_32k.jsonl \
    --testdata_folder ./VaLProbing-32K/ \
    --output_folder $results_path \
    --max_length 128 \
    --tensor_parallel_size $parallel_size

python ../vllm_inference/vllm_inference.py \
    --model_path $model_path \
    --testdata_file database_forward_32k.jsonl \
    --testdata_folder ./VaLProbing-32K/ \
    --output_folder $results_path \
    --max_length 128 \
    --tensor_parallel_size $parallel_size

python plot.py

cd ..