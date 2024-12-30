#!/bin/bash
# 实验目的：探寻RGCN的最优参数
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 创建 logs 目录
if [ ! -d "./logs_exp" ]; then
    mkdir ./logs_exp
fi

# 创建 logs/AnomalyDetection 目录
if [ ! -d "./logs_exp/Test" ]; then
    mkdir ./logs_exp/Test
fi

# 设置模型和数据路径
# model_name=PathFormer
root_path=./dataset/ALFA/dataset3
model_id_name=dataset3
data_name=ALFA
anomaly_ratio=5
d_model=64
seq_len=96
train_epochs=5
patience=3
batch_size=128
k=3

for model_name in MSGNet TimesNet RGCN; do
    for seq_len in 96; do

        # 构建日志文件路径
        log_file="logs_exp/Test/${model_id_name}_${model_name}.log"

        # 确保日志目录存在
        mkdir -p "$(dirname "$log_file")"

        # 运行 Python 脚本并将输出重定向到日志文件
        python -u run.py \
        --root_path $root_path \
        --model_id $model_id_name \
        --model $model_name \
        --data $data_name \
        --seq_len $seq_len \
        --d_model $d_model \
        --train_epochs $train_epochs \
        --patience $patience \
        --anomaly_ratio $anomaly_ratio \
        --batch_size $batch_size > "$log_file"
    done
done
