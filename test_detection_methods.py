#!/usr/bin/env python3
"""
测试两种错误标签检测方法的脚本
展示 experiment.py 的逻辑与 analysis_error.py 的区别
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append('.')
import config.settings as settings
from experiments.analysis.analysis_error import (
    load_shapley_data, 
    calculate_original_label_contribution,
    detect_mislabeled_samples
)

def analyze_contribution_method(experiment_type='resnet'):
    """分析原始标签贡献检测方法"""
    print(f"🔬 分析原始标签贡献检测方法: {experiment_type.upper()}")
    
    # 加载数据
    try:
        shapley_data = load_shapley_data(experiment_type)
        print("✅ Shapley数据加载成功")
    except FileNotFoundError as e:
        print(f"❌ 找不到Shapley数据: {e}")
        print("请先运行Shapley值计算")
        return
    
    shapley_vectors = shapley_data['shapley_vectors']
    original_labels = shapley_data.get('original_labels', shapley_data['noisy_labels'])
    flipped_indices = shapley_data.get('flipped_indices', [])
    
    print(f"📊 数据概览:")
    print(f"  总样本数: {len(shapley_vectors)}")
    print(f"  翻转样本数: {len(flipped_indices)}")
    print(f"  翻转比例: {len(flipped_indices)/len(shapley_vectors)*100:.2f}%")
    
    # === 原始标签贡献方法分析 ===
    print(f"\n🔍 原始标签贡献法 (experiment.py方法)")
    detection_results = detect_mislabeled_samples(shapley_data, threshold=0.0)
    
    print(f"  可疑样本数: {detection_results['num_suspicious']}")
    if 'f1_score' in detection_results:
        print(f"  精确率: {detection_results['precision']:.4f}")
        print(f"  召回率: {detection_results['recall']:.4f}")
        print(f"  F1分数: {detection_results['f1_score']:.4f}")
    
    # 展示关键统计
    if 'correct_scores_mean' in detection_results:
        print(f"\n📈 原始标签贡献统计:")
        print(f"  正确样本均值: {detection_results['correct_scores_mean']:.6f}")
        print(f"  错误样本均值: {detection_results['flipped_scores_mean']:.6f}")
        print(f"  理论验证: {'✅ 符合预期' if detection_results['flipped_scores_mean'] < 0 and detection_results['correct_scores_mean'] > 0 else '⚠️ 需要分析'}")
    
    # === 可视化分析 ===
    print(f"\n🎨 生成可视化分析...")
    
    # 计算贡献分数
    contribution_scores = calculate_original_label_contribution(shapley_vectors, original_labels)
    
    # 分组数据
    all_indices = np.arange(len(original_labels))
    correct_indices = np.setdiff1d(all_indices, flipped_indices)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 子图1: 原始标签贡献分布
    axes[0,0].hist(contribution_scores[correct_indices], bins=50, alpha=0.7, label='正确标签', color='green', density=True)
    axes[0,0].hist(contribution_scores[flipped_indices], bins=20, alpha=0.8, label='错误标签', color='red', density=True)
    axes[0,0].axvline(0, color='black', linestyle='--', label='阈值 (0)')
    axes[0,0].set_title('原始标签贡献分布')
    axes[0,0].set_xlabel('对原始标签的贡献')
    axes[0,0].set_ylabel('密度')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 子图2: 检测性能指标
    if 'f1_score' in detection_results:
        metrics = ['精确率', '召回率', 'F1分数']
        values = [detection_results['precision'], detection_results['recall'], detection_results['f1_score']]
        bars = axes[0,1].bar(metrics, values, color=['skyblue', 'lightgreen', 'coral'], alpha=0.8)
        axes[0,1].set_ylabel('性能指标')
        axes[0,1].set_title('检测性能')
        axes[0,1].set_ylim(0, 1)
        axes[0,1].grid(True, alpha=0.3)
        
        # 在柱状图上显示数值
        for bar, value in zip(bars, values):
            axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                          f'{value:.3f}', ha='center', va='bottom')
    
    # 子图3: 样本散点图
    axes[1,0].scatter(correct_indices, contribution_scores[correct_indices], 
                     alpha=0.6, s=8, color='green', label='正确标签')
    axes[1,0].scatter(flipped_indices, contribution_scores[flipped_indices], 
                     alpha=0.8, s=12, color='red', label='错误标签')
    axes[1,0].axhline(0, color='black', linestyle='--', label='阈值 (0)')
    axes[1,0].set_xlabel('样本索引')
    axes[1,0].set_ylabel('对原始标签的贡献')
    axes[1,0].set_title('样本贡献散点图')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 子图4: 贡献值排序
    sorted_indices = np.argsort(contribution_scores)
    sorted_scores = contribution_scores[sorted_indices]
    
    axes[1,1].plot(range(len(sorted_scores)), sorted_scores, color='blue', alpha=0.7, linewidth=1)
    
    # 标记可疑样本和真实翻转样本
    suspicious_indices = detection_results['suspicious_indices']
    suspicious_positions = np.where(np.isin(sorted_indices, suspicious_indices))[0]
    axes[1,1].scatter(suspicious_positions, sorted_scores[suspicious_positions], 
                     color='red', s=15, alpha=0.8, label='检测为可疑')
    
    if len(flipped_indices) > 0:
        flipped_positions = np.where(np.isin(sorted_indices, flipped_indices))[0]
        axes[1,1].scatter(flipped_positions, sorted_scores[flipped_positions], 
                         color='orange', s=20, marker='x', alpha=0.8, label='实际翻转')
    
    axes[1,1].axhline(0, color='black', linestyle='--', alpha=0.7, label='阈值 (0)')
    axes[1,1].set_xlabel('样本排序（按贡献值）')
    axes[1,1].set_ylabel('对原始标签的贡献')
    axes[1,1].set_title('贡献值排序图')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    save_dir = settings.RESULTS_DIR
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'contribution_method_analysis_{experiment_type}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 分析图保存至: {save_path}")
    
    # === 总结 ===
    print(f"\n📋 检测方法总结:")
    print(f"{'='*50}")
    print(f"方法: 原始标签贡献法 (experiment.py使用的方法)")
    print(f"原理: 计算每个样本对其原始真实标签的Shapley贡献")
    print(f"阈值: 贡献值 < 0 的样本被标记为可疑")
    print(f"")
    
    print(f"💡 方法优势:")
    print(f"  - 直接针对真实标签，逻辑清晰直观")
    print(f"  - 自然的决策边界（贡献值=0）")
    print(f"  - 理论基础扎实：错误标签样本对正确类别贡献为负")
    print(f"  - 与experiment.py的检测逻辑完全一致")

if __name__ == "__main__":
    analyze_contribution_method('resnet')