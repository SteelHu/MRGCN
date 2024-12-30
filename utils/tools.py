import os
import argparse
import json

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.metrics import roc_curve, auc

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')



def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)




def compute_and_save_ROC_AUC(test_energy, test_labels, save_path=None):
    """
    计算ROC曲线和AUC分数，并将ROC曲线（调整前后）保存为PDF文件。

    在计算过程中应用点调整策略：
    如果真实标签中某一个异常区间中有一个点被预测为异常，
    那么这整个异常区间都视为被正确预测。

    参数:
    - test_energy (list or np.ndarray): 异常检测分数，数值越高表示越异常。
    - test_labels (list or np.ndarray): 真实标签，0表示正常，1表示异常。
    - save_path (str): ROC曲线图保存的路径（默认为当前目录）。

    返回:
    - unadjusted_auc (float): 调整前的AUC分数。
    - adjusted_auc (float): 调整后的AUC分数。
    - num_real_anomalies (int): test_labels 中真实的异常区间数量。
    - real_anomaly_intervals (list of tuples): 真实的异常区间。
    - num_detected_anomalies (int): 识别出的异常区间数量。
    - detected_anomaly_intervals (list of tuples): 识别出的异常区间。
    """

    # 确保输入为NumPy数组
    test_energy = np.array(test_energy)
    test_labels = np.array(test_labels)

    # 识别真实标签中的异常区间
    def get_anomaly_intervals(labels):
        """
        返回标签中的所有异常区间的起始和结束索引。

        参数:
        - labels (np.ndarray): 真实标签数组。

        返回:
        - intervals (list of tuples): 每个元组包含一个异常区间的起始和结束索引。
        """
        intervals = []
        in_anomaly = False
        start = 0
        for i, label in enumerate(labels):
            if label == 1 and not in_anomaly:
                in_anomaly = True
                start = i
            elif label == 0 and in_anomaly:
                in_anomaly = False
                end = i - 1
                intervals.append((start, end))
                # 如果最后一个标签是异常，记录下来
        if in_anomaly:
            intervals.append((start, len(labels) - 1))
        return intervals

        # 获取真实的异常区间

    real_anomaly_intervals = get_anomaly_intervals(test_labels)
    num_real_anomalies = len(real_anomaly_intervals)

    # 打印真实的异常区间
    real_intervals_str = ", ".join([f"[{start}, {end}]" for start, end in real_anomaly_intervals])
    print(f"在test_labels中识别到 {num_real_anomalies} 个真实的异常区间：{real_intervals_str}")

    # 计算调整前的ROC曲线和AUC
    unadjusted_fpr, unadjusted_tpr, unadjusted_thresholds = roc_curve(test_labels, test_energy)
    unadjusted_auc = auc(unadjusted_fpr, unadjusted_tpr)
    print(f"调整前AUC分数: {unadjusted_auc:.4f}")

    # 初始化调整后的FPR和TPR列表
    adjusted_fpr = []
    adjusted_tpr = []
    adjusted_thresholds = []

    # 使用与调整前相同的阈值进行调整
    # 这样可以确保比较的一致性
    for thresh in unadjusted_thresholds:
        # 初始预测
        initial_predictions = (test_energy >= thresh).astype(int)

        # 应用点调整策略
        adjusted_predictions = initial_predictions.copy()

        for interval in real_anomaly_intervals:
            start, end = interval
            # 检查该区间内是否有至少一个点被预测为异常
            if np.any(initial_predictions[start:end + 1] == 1):
                # 如果有，则将整个区间的预测设为异常
                adjusted_predictions[start:end + 1] = 1

                # 计算调整后的TP, FP, FN, TN
        TP = np.sum((adjusted_predictions == 1) & (test_labels == 1))
        FP = np.sum((adjusted_predictions == 1) & (test_labels == 0))
        FN = np.sum((adjusted_predictions == 0) & (test_labels == 1))
        TN = np.sum((adjusted_predictions == 0) & (test_labels == 0))

        # 计算FPR和TPR
        current_fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        current_tpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0

        adjusted_fpr.append(current_fpr)
        adjusted_tpr.append(current_tpr)
        adjusted_thresholds.append(thresh)

        # 转换为NumPy数组
    adjusted_fpr = np.array(adjusted_fpr)
    adjusted_tpr = np.array(adjusted_tpr)
    adjusted_thresholds = np.array(adjusted_thresholds)

    # 计算调整后的AUC
    adjusted_auc = auc(adjusted_fpr, adjusted_tpr)
    print(f"调整后AUC分数: {adjusted_auc:.4f}")

    # 选择一个阈值来展示识别出的异常区间
    # 这里选择Youden's J指数最大的阈值
    youdens_j = adjusted_tpr - adjusted_fpr
    optimal_idx = np.argmax(youdens_j)
    optimal_threshold = adjusted_thresholds[optimal_idx]
    print(f"选择的最佳阈值: {optimal_threshold:.4f} (对应于Youden's J指数)")

    # 基于最佳阈值进行最终的预测和调整
    final_predictions = (test_energy >= optimal_threshold).astype(int)
    final_adjusted_predictions = final_predictions.copy()

    for interval in real_anomaly_intervals:
        start, end = interval
        if np.any(final_predictions[start:end + 1] == 1):
            final_adjusted_predictions[start:end + 1] = 1

            # 识别预测结果中的异常区间
    detected_anomaly_intervals = get_anomaly_intervals(final_adjusted_predictions)
    num_detected_anomalies = len(detected_anomaly_intervals)

    # 打印识别出的异常区间
    detected_intervals_str = ", ".join([f"[{start}, {end}]" for start, end in detected_anomaly_intervals])
    print(f"在预测结果中识别到 {num_detected_anomalies} 个异常区间：{detected_intervals_str}")

    # 绘制ROC曲线
    plt.figure(figsize=(10, 8))
    plt.plot(unadjusted_fpr, unadjusted_tpr, color='blue', lw=2, linestyle='--',
             label=f'原始ROC曲线 (AUC = {unadjusted_auc:.4f})')
    plt.plot(adjusted_fpr, adjusted_tpr, color='darkorange', lw=2, label=f'调整后ROC曲线 (AUC = {adjusted_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机猜测')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率 (False Positive Rate)', fontsize=14)
    plt.ylabel('真正率 (True Positive Rate)', fontsize=14)
    plt.title('接收者操作特征 (ROC) 曲线对比', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)

    # 如果没有指定保存路径，则使用当前目录
    if save_path is None:
        save_path = os.getcwd()
        # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)
    # 保存为PDF文件
    plt.savefig(os.path.join(save_path, 'roc_curve_comparison.pdf'), format='pdf')
    plt.close()  # 关闭图形，避免在某些环境中显示

    # 绘制异常区间图
    def plot_anomalies(test_energy, test_labels, real_intervals, detected_intervals, save_path=None):
        plt.figure(figsize=(15, 5))
        plt.plot(test_energy, label='异常检测分数', color='blue')
        plt.title('异常检测分数与真实/预测异常区间')
        plt.xlabel('时间步')
        plt.ylabel('能量分数')

        # 绘制真实异常区间
        for interval in real_intervals:
            start, end = interval
            plt.axvspan(start, end, color='green', alpha=0.3)

            # 绘制识别出的异常区间
        for interval in detected_intervals:
            start, end = interval
            plt.axvspan(start, end, color='red', alpha=0.3)

            # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', edgecolor='green', alpha=0.3, label='真实异常区间'),
            Patch(facecolor='red', edgecolor='red', alpha=0.3, label='识别异常区间')
        ]
        plt.legend(handles=legend_elements)

        if save_path is None:
            save_path = os.getcwd()
        plt.savefig(os.path.join(save_path, 'anomalies_plot.pdf'), format='pdf')
        plt.close()

        # 绘制并保存异常区间图

    plot_anomalies(test_energy, test_labels, real_anomaly_intervals, detected_anomaly_intervals, save_path)

    return unadjusted_auc, adjusted_auc

# def compute_and_save_ROC_AUC(test_energy, test_labels, save_path=None):
#     """
#     计算ROC曲线和AUC分数，并将ROC曲线（调整前后）保存为PDF文件。
#
#     在计算过程中应用点调整策略：
#     如果真实标签中某一个异常区间中有一个点被预测为异常，
#     那么这整个异常区间都视为被正确预测。
#
#     参数:
#     - test_energy (list or np.ndarray): 异常检测分数，数值越高表示越异常。
#     - test_labels (list or np.ndarray): 真实标签，0表示正常，1表示异常。
#     - save_path (str): ROC曲线图保存的路径（默认为当前目录）。
#
#     返回:
#     - unadjusted_auc (float): 调整前的AUC分数。
#     - adjusted_auc (float): 调整后的AUC分数。
#     - num_real_anomalies (int): test_labels 中真实的异常区间数量。
#     - real_anomaly_intervals (list of tuples): 真实的异常区间。
#     - num_detected_anomalies (int): 识别出的异常区间数量。
#     - detected_anomaly_intervals (list of tuples): 识别出的异常区间。
#     """
#
#     # 确保输入为NumPy数组
#     test_energy = np.array(test_energy)
#     test_labels = np.array(test_labels)
#
#     # 识别真实标签中的异常区间
#     def get_anomaly_intervals(labels):
#         """
#         返回标签中的所有异常区间的起始和结束索引。
#
#         参数:
#         - labels (np.ndarray): 真实标签数组。
#
#         返回:
#         - intervals (list of tuples): 每个元组包含一个异常区间的起始和结束索引。
#         """
#         intervals = []
#         in_anomaly = False
#         start = 0
#         for i, label in enumerate(labels):
#             if label == 1 and not in_anomaly:
#                 in_anomaly = True
#                 start = i
#             elif label == 0 and in_anomaly:
#                 in_anomaly = False
#                 end = i - 1
#                 intervals.append((start, end))
#                 # 如果最后一个标签是异常，记录下来
#         if in_anomaly:
#             intervals.append((start, len(labels) - 1))
#         return intervals
#
#         # 获取真实的异常区间
#
#     real_anomaly_intervals = get_anomaly_intervals(test_labels)
#     num_real_anomalies = len(real_anomaly_intervals)
#
#     # 打印真实的异常区间
#     real_intervals_str = ", ".join([f"[{start}, {end}]" for start, end in real_anomaly_intervals])
#     print(f"在test_labels中识别到 {num_real_anomalies} 个真实的异常区间：{real_intervals_str}")
#
#     # 计算调整前的ROC曲线和AUC
#     unadjusted_fpr, unadjusted_tpr, unadjusted_thresholds = roc_curve(test_labels, test_energy)
#     unadjusted_auc = auc(unadjusted_fpr, unadjusted_tpr)
#     print(f"调整前AUC分数: {unadjusted_auc:.4f}")
#
#     # 初始化调整后的FPR和TPR列表
#     adjusted_fpr = []
#     adjusted_tpr = []
#     adjusted_thresholds = []
#
#     # 使用与调整前相同的阈值进行调整
#     # 这样可以确保比较的一致性
#     for thresh in unadjusted_thresholds:
#         # 初始预测
#         initial_predictions = (test_energy >= thresh).astype(int)
#
#         # 应用点调整策略
#         adjusted_predictions = initial_predictions.copy()
#
#         for interval in real_anomaly_intervals:
#             start, end = interval
#             # 检查该区间内是否有至少一个点被预测为异常
#             if np.any(initial_predictions[start:end + 1] == 1):
#                 # 如果有，则将整个区间的预测设为异常
#                 adjusted_predictions[start:end + 1] = 1
#
#                 # 计算调整后的TP, FP, FN, TN
#         TP = np.sum((adjusted_predictions == 1) & (test_labels == 1))
#         FP = np.sum((adjusted_predictions == 1) & (test_labels == 0))
#         FN = np.sum((adjusted_predictions == 0) & (test_labels == 1))
#         TN = np.sum((adjusted_predictions == 0) & (test_labels == 0))
#
#         # 计算FPR和TPR
#         current_fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
#         current_tpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0
#
#         adjusted_fpr.append(current_fpr)
#         adjusted_tpr.append(current_tpr)
#         adjusted_thresholds.append(thresh)
#
#         # 转换为NumPy数组
#     adjusted_fpr = np.array(adjusted_fpr)
#     adjusted_tpr = np.array(adjusted_tpr)
#     adjusted_thresholds = np.array(adjusted_thresholds)
#
#     # 计算调整后的AUC
#     adjusted_auc = auc(adjusted_fpr, adjusted_tpr)
#     print(f"调整后AUC分数: {adjusted_auc:.4f}")
#
#     # 选择一个阈值来展示识别出的异常区间
#     # 这里选择Youden's J指数最大的阈值
#     youdens_j = adjusted_tpr - adjusted_fpr
#     optimal_idx = np.argmax(youdens_j)
#     optimal_threshold = adjusted_thresholds[optimal_idx]
#     print(f"选择的最佳阈值: {optimal_threshold:.4f} (对应于Youden's J指数)")
#
#     # 基于最佳阈值进行最终的预测和调整
#     final_predictions = (test_energy >= optimal_threshold).astype(int)
#     final_adjusted_predictions = final_predictions.copy()
#
#     for interval in real_anomaly_intervals:
#         start, end = interval
#         if np.any(final_predictions[start:end + 1] == 1):
#             final_adjusted_predictions[start:end + 1] = 1
#
#             # 识别预测结果中的异常区间
#     detected_anomaly_intervals = get_anomaly_intervals(final_adjusted_predictions)
#     num_detected_anomalies = len(detected_anomaly_intervals)
#
#     # 打印识别出的异常区间
#     detected_intervals_str = ", ".join([f"[{start}, {end}]" for start, end in detected_anomaly_intervals])
#     print(f"在预测结果中识别到 {num_detected_anomalies} 个异常区间：{detected_intervals_str}")
#
#     # 绘制ROC曲线
#     plt.figure(figsize=(10, 8))
#     plt.plot(unadjusted_fpr, unadjusted_tpr, color='blue', lw=2, linestyle='--',
#              label=f'原始ROC曲线 (AUC = {unadjusted_auc:.4f})')
#     plt.plot(adjusted_fpr, adjusted_tpr, color='darkorange', lw=2, label=f'调整后ROC曲线 (AUC = {adjusted_auc:.4f})')
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机猜测')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('假正率 (False Positive Rate)', fontsize=14)
#     plt.ylabel('真正率 (True Positive Rate)', fontsize=14)
#     plt.title('接收者操作特征 (ROC) 曲线对比', fontsize=16)
#     plt.legend(loc="lower right", fontsize=12)
#     plt.grid(alpha=0.3)
#
#     # 如果没有指定保存路径，则使用当前目录
#     if save_path is None:
#         save_path = os.getcwd()
#         # 确保保存路径存在
#     os.makedirs(save_path, exist_ok=True)
#     # 保存为PDF文件
#     plt.savefig(os.path.join(save_path, 'roc_curve_comparison.pdf'), format='pdf')
#     plt.close()  # 关闭图形，避免在某些环境中显示
#
#     # 绘制异常区间图
#     def plot_anomalies(test_energy, test_labels, real_intervals, detected_intervals, save_path=None):
#         plt.figure(figsize=(15, 5))
#         plt.plot(test_energy, label='异常检测分数', color='blue')
#         plt.title('异常检测分数与真实/预测异常区间')
#         plt.xlabel('时间步')
#         plt.ylabel('能量分数')
#
#         # 绘制真实异常区间
#         for interval in real_intervals:
#             start, end = interval
#             plt.axvspan(start, end, color='green', alpha=0.3)
#
#             # 绘制识别出的异常区间
#         for interval in detected_intervals:
#             start, end = interval
#             plt.axvspan(start, end, color='red', alpha=0.3)
#
#             # 添加图例
#         from matplotlib.patches import Patch
#         legend_elements = [
#             Patch(facecolor='green', edgecolor='green', alpha=0.3, label='真实异常区间'),
#             Patch(facecolor='red', edgecolor='red', alpha=0.3, label='识别异常区间')
#         ]
#         plt.legend(handles=legend_elements)
#
#         if save_path is None:
#             save_path = os.getcwd()
#         plt.savefig(os.path.join(save_path, 'anomalies_plot.pdf'), format='pdf')
#         plt.close()
#
#         # 绘制并保存异常区间图
#
#     plot_anomalies(test_energy, test_labels, real_anomaly_intervals, detected_anomaly_intervals, save_path)
#
#     return unadjusted_auc, adjusted_auc


def parse_node_types(s):
    """
    将 JSON 字符串解析为字典，并将每个值从字符串转换为整数列表。

    Args:
        s (str): JSON 格式的字符串，例如 '{"A": "0,1,3", "B": "2,4,5,6", "C": "7,8,9"}'

    Returns:
        dict: 解析后的字典，例如 {'A': [0, 1, 3], 'B': [2, 4, 5, 6], 'C': [7, 8, 9]}
    """
    # 解析 JSON 字符串为字典
    try:
        d = json.loads(s)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"无效的 JSON 格式: {e}")

    # 将字符串值转换为整数列表
    parsed_dict = {}
    for key, value in d.items():
        try:
            # 将字符串按逗号分割并转换为整数
            parsed_dict[key] = [int(item.strip()) for item in value.split(',')]
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"键 '{key}' 的值无效: {e}")

    return parsed_dict