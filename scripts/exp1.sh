#!/bin/bash
# 实验目的：探寻最优参数
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 创建 logs 目录
if [ ! -d "./logs_exp" ]; then
    mkdir ./logs_exp
fi

# 创建 logs/AnomalyDetection 目录
if [ ! -d "./logs_exp/exp1" ]; then
    mkdir ./logs_exp/exp1
fi

# 设置模型和数据路径
model_name=RGCN
root_path=./dataset/ALFA/dataset1
model_id_name=dataset1
data_name=ALFA
anomaly_ratio=5
d_model=64
seq_len=96
train_epochs=3
patience=3
batch_size=128
k=3

for num_relations in 1 2 3 4 5; do
    for agg_method in sum average max weighted_sum attention mlp conv_fusion; do
        for conv_channel in 32 64; do

            # 构建日志文件路径
            log_file="logs_exp/exp1/${model_id_name}_${model_name}_rel${num_relations}_agg${agg_method}_ \
            conv${conv_channel}_ratio${anomaly_ratio}_d${d_model}_len${seq_len}_ep${train_epochs}_ \
            pat${patience}_bs${batch_size}_k${k}.log"

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
            --batch_size $batch_size \
            --num_relations $num_relations \
            --agg_method $agg_method \
            --conv_channel $conv_channel \
            --skip_channel $conv_channel  > "$log_file"
        done
    done
done
