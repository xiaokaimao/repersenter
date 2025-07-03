#!/usr/bin/env python3
"""
Shapley值贡献分析和阈值优化脚本
分析翻转样本和正确样本的贡献值分布，找出最佳检测阈值
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score
import pandas as pd

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config.settings as settings

def load_shapley_data(experiment_type='resnet'):
    """加载Shapley值数据"""
    config = settings.CORE_SET_EXPERIMENT_CONFIG[experiment_type]
    
    if experiment_type == 'transformer':
        model_name_safe = config['model_name'].replace('/', '_')
        shapley_file_name = f"shapley_vectors_{config['dataset_name']}_{model_name_safe}_noise_{config.get('label_noise_rate', 0.0)}.pt"
    else:
        shapley_file_name = f"shapley_vectors_{config['dataset_name']}_{config['model_name']}_noise_{config.get('label_noise_rate', 0.0)}.pt"
    
    load_path = os.path.join(settings.RESULTS_DIR, shapley_file_name)
    
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"找不到Shapley文件: {load_path}")
    
    print(f"加载Shapley数据: {load_path}")
    return torch.load(load_path, map_location='cpu', weights_only=False)

def calculate_original_label_contribution(shapley_vectors, original_labels):
    """
    计算每个样本对其原始真实标签的贡献
    
    Args:
        shapley_vectors: Shapley值矩阵 (n_samples, n_classes)
        original_labels: 原始真实标签
    
    Returns:
        numpy.array: 每个样本对其原始标签的贡献分数
    """
    # 提取每个样本对其原始真实标签的Shapley值
    indices = torch.arange(len(original_labels))
    contribution_scores = shapley_vectors[indices, original_labels].numpy()
    return contribution_scores

def analyze_contribution_distribution(shapley_data):
    """分析贡献值分布"""
    shapley_vectors = shapley_data['shapley_vectors']
    original_labels = shapley_data.get('original_labels', shapley_data['noisy_labels'])
    flipped_indices = shapley_data.get('flipped_indices', [])
    
    # 计算对原始标签的贡献
    contribution_scores = calculate_original_label_contribution(shapley_vectors, original_labels)
    
    # 分组索引
    all_indices = np.arange(len(original_labels))
    correct_indices = np.setdiff1d(all_indices, flipped_indices)
    
    # 提取贡献分数
    if len(flipped_indices) > 0:
        flipped_scores = contribution_scores[flipped_indices]
        correct_scores = contribution_scores[correct_indices]
    else:
        flipped_scores = np.array([])
        correct_scores = contribution_scores
    
    return {
        'all_scores': contribution_scores,
        'flipped_scores': flipped_scores,
        'correct_scores': correct_scores,
        'flipped_indices': flipped_indices,
        'correct_indices': correct_indices
    }

def find_optimal_threshold(contribution_data):
    """找出使F1分数最大化的最佳阈值"""
    if len(contribution_data['flipped_scores']) == 0:
        print("警告: 没有翻转样本，无法计算最佳阈值")
        return None, None, None
    
    # 创建真实标签 (1表示翻转样本，0表示正确样本)
    y_true = np.concatenate([
        np.ones(len(contribution_data['flipped_scores'])),  # 翻转样本
        np.zeros(len(contribution_data['correct_scores']))  # 正确样本
    ])
    
    # 贡献分数 (用负值，因为我们希望低贡献值表示翻转样本)
    scores = np.concatenate([
        contribution_data['flipped_scores'],
        contribution_data['correct_scores']
    ])
    
    # 使用负分数进行precision-recall曲线计算
    # 因为我们希望低贡献值对应高"异常"分数
    precision, recall, thresholds = precision_recall_curve(y_true, -scores)
    
    # 计算F1分数（避免除零错误）
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    
    # 找出最佳阈值
    best_idx = np.argmax(f1_scores)
    best_threshold = -thresholds[best_idx]  # 转换回原始阈值
    best_f1 = f1_scores[best_idx]
    best_precision = precision[best_idx]
    best_recall = recall[best_idx]
    
    # 确保数组长度一致
    min_length = min(len(thresholds), len(f1_scores), len(precision), len(recall))
    
    return best_threshold, best_f1, {
        'precision': best_precision,
        'recall': best_recall,
        'f1_score': best_f1,
        'all_thresholds': -thresholds[:min_length],
        'all_f1_scores': f1_scores[:min_length],
        'all_precision': precision[:min_length],
        'all_recall': recall[:min_length]
    }

def evaluate_threshold(contribution_data, threshold):
    """评估特定阈值的性能"""
    if len(contribution_data['flipped_scores']) == 0:
        return None
    
    # 预测 (贡献值 < 阈值 的样本被预测为翻转样本)
    flipped_predictions = contribution_data['flipped_scores'] < threshold
    correct_predictions = contribution_data['correct_scores'] < threshold
    
    # 统计
    tp = np.sum(flipped_predictions)  # 翻转样本中被正确检测的
    fp = np.sum(correct_predictions)  # 正确样本中被错误检测的
    fn = np.sum(~flipped_predictions)  # 翻转样本中被遗漏的
    tn = np.sum(~correct_predictions)  # 正确样本中被正确识别的
    
    # 计算指标
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'threshold': threshold,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'num_detected': tp + fp,
        'detection_rate': (tp + fp) / len(contribution_data['all_scores'])
    }

def plot_analysis_results(contribution_data, optimal_results, save_path=None):
    """绘制分析结果"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 贡献值分布直方图
    ax1 = axes[0, 0]
    if len(contribution_data['flipped_scores']) > 0:
        ax1.hist(contribution_data['correct_scores'], bins=50, density=True, alpha=0.7, 
                label=f'正确样本 (n={len(contribution_data["correct_scores"])})', color='green')
        ax1.hist(contribution_data['flipped_scores'], bins=30, density=True, alpha=0.8, 
                label=f'翻转样本 (n={len(contribution_data["flipped_scores"])})', color='red')
    else:
        ax1.hist(contribution_data['correct_scores'], bins=50, density=True, alpha=0.7, 
                label=f'所有样本 (n={len(contribution_data["correct_scores"])})', color='blue')
    
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=2, label='阈值=0')
    if optimal_results and 'best_threshold' in optimal_results:
        ax1.axvline(x=optimal_results['best_threshold'], color='purple', linestyle='--', 
                   linewidth=2, label=f'最佳阈值={optimal_results["best_threshold"]:.3f}')
    
    ax1.set_title('贡献值分布')
    ax1.set_xlabel('对原始标签的贡献值')
    ax1.set_ylabel('密度')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 贡献值箱型图
    ax2 = axes[0, 1]
    if len(contribution_data['flipped_scores']) > 0:
        data_for_box = [contribution_data['correct_scores'], contribution_data['flipped_scores']]
        labels_for_box = ['正确样本', '翻转样本']
        colors = ['green', 'red']
    else:
        data_for_box = [contribution_data['correct_scores']]
        labels_for_box = ['所有样本']
        colors = ['blue']
    
    box_plot = ax2.boxplot(data_for_box, tick_labels=labels_for_box, patch_artist=True)
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=2, label='阈值=0')
    if optimal_results and 'best_threshold' in optimal_results:
        ax2.axhline(y=optimal_results['best_threshold'], color='purple', linestyle='--', 
                   linewidth=2, label=f'最佳阈值={optimal_results["best_threshold"]:.3f}')
    
    ax2.set_title('贡献值箱型图')
    ax2.set_ylabel('对原始标签的贡献值')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. F1分数随阈值变化
    ax3 = axes[1, 0]
    if optimal_results and 'metrics' in optimal_results:
        metrics = optimal_results['metrics']
        ax3.plot(metrics['all_thresholds'], metrics['all_f1_scores'], 'b-', linewidth=2, label='F1分数')
        ax3.plot(metrics['all_thresholds'], metrics['all_precision'], 'g--', linewidth=2, label='精确率')
        ax3.plot(metrics['all_thresholds'], metrics['all_recall'], 'r--', linewidth=2, label='召回率')
        
        # 标记最佳点
        ax3.scatter([optimal_results['best_threshold']], [optimal_results['best_f1']], 
                   color='purple', s=100, zorder=5, label=f'最佳点 (F1={optimal_results["best_f1"]:.3f})')
    
    ax3.set_title('性能指标随阈值变化')
    ax3.set_xlabel('阈值')
    ax3.set_ylabel('分数')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 检测统计
    ax4 = axes[1, 1]
    if len(contribution_data['flipped_scores']) > 0:
        # 计算不同阈值下的统计
        thresholds_to_test = [0.0]
        if optimal_results and 'best_threshold' in optimal_results:
            thresholds_to_test.append(optimal_results['best_threshold'])
        
        results_data = []
        for thresh in thresholds_to_test:
            eval_result = evaluate_threshold(contribution_data, thresh)
            if eval_result:
                results_data.append(eval_result)
        
        if results_data:
            thresh_labels = [f'阈值={r["threshold"]:.3f}' for r in results_data]
            f1_scores = [r['f1_score'] for r in results_data]
            precision_scores = [r['precision'] for r in results_data]
            recall_scores = [r['recall'] for r in results_data]
            
            x = np.arange(len(thresh_labels))
            width = 0.25
            
            ax4.bar(x - width, f1_scores, width, label='F1分数', alpha=0.8)
            ax4.bar(x, precision_scores, width, label='精确率', alpha=0.8)
            ax4.bar(x + width, recall_scores, width, label='召回率', alpha=0.8)
            
            ax4.set_xlabel('阈值')
            ax4.set_ylabel('分数')
            ax4.set_title('不同阈值下的性能比较')
            ax4.set_xticks(x)
            ax4.set_xticklabels(thresh_labels)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, '没有翻转样本\n无法计算性能指标', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=14)
        ax4.set_title('性能指标')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    plt.show()

def print_detailed_statistics(contribution_data, optimal_results):
    """打印详细统计信息"""
    print("\n" + "="*60)
    print("📊 详细统计分析")
    print("="*60)
    
    # 基本统计
    print("\n1. 基本统计:")
    print(f"   总样本数: {len(contribution_data['all_scores'])}")
    print(f"   翻转样本数: {len(contribution_data['flipped_scores'])}")
    print(f"   正确样本数: {len(contribution_data['correct_scores'])}")
    if len(contribution_data['flipped_scores']) > 0:
        print(f"   翻转比例: {len(contribution_data['flipped_scores']) / len(contribution_data['all_scores']) * 100:.2f}%")
    
    # 贡献值统计
    print("\n2. 贡献值统计:")
    all_scores = contribution_data['all_scores']
    print(f"   整体平均值: {np.mean(all_scores):.6f}")
    print(f"   整体标准差: {np.std(all_scores):.6f}")
    print(f"   整体最小值: {np.min(all_scores):.6f}")
    print(f"   整体最大值: {np.max(all_scores):.6f}")
    
    if len(contribution_data['flipped_scores']) > 0:
        flipped_scores = contribution_data['flipped_scores']
        correct_scores = contribution_data['correct_scores']
        
        print(f"\n   翻转样本贡献值:")
        print(f"     平均值: {np.mean(flipped_scores):.6f}")
        print(f"     标准差: {np.std(flipped_scores):.6f}")
        print(f"     最小值: {np.min(flipped_scores):.6f}")
        print(f"     最大值: {np.max(flipped_scores):.6f}")
        
        print(f"\n   正确样本贡献值:")
        print(f"     平均值: {np.mean(correct_scores):.6f}")
        print(f"     标准差: {np.std(correct_scores):.6f}")
        print(f"     最小值: {np.min(correct_scores):.6f}")
        print(f"     最大值: {np.max(correct_scores):.6f}")
    
    # 意外情况分析
    print("\n3. 意外情况分析:")
    if len(contribution_data['flipped_scores']) > 0:
        flipped_positive = np.sum(contribution_data['flipped_scores'] >= 0)
        correct_negative = np.sum(contribution_data['correct_scores'] < 0)
        
        print(f"   翻转样本中贡献值 >= 0 的数量: {flipped_positive}")
        print(f"   翻转样本中贡献值 >= 0 的比例: {flipped_positive / len(contribution_data['flipped_scores']) * 100:.2f}%")
        print(f"   正确样本中贡献值 < 0 的数量: {correct_negative}")
        print(f"   正确样本中贡献值 < 0 的比例: {correct_negative / len(contribution_data['correct_scores']) * 100:.2f}%")
        
        # 理论验证
        theory_check = np.mean(contribution_data['flipped_scores']) < np.mean(contribution_data['correct_scores'])
        print(f"   理论验证(翻转样本平均值 < 正确样本平均值): {'✅ 通过' if theory_check else '❌ 未通过'}")
    else:
        negative_count = np.sum(contribution_data['correct_scores'] < 0)
        print(f"   所有样本中贡献值 < 0 的数量: {negative_count}")
        print(f"   所有样本中贡献值 < 0 的比例: {negative_count / len(contribution_data['correct_scores']) * 100:.2f}%")
    
    # 最佳阈值分析
    if optimal_results and 'best_threshold' in optimal_results:
        print("\n4. 最佳阈值分析:")
        print(f"   最佳阈值: {optimal_results['best_threshold']:.6f}")
        print(f"   最佳F1分数: {optimal_results['best_f1']:.4f}")
        print(f"   对应精确率: {optimal_results['metrics']['precision']:.4f}")
        print(f"   对应召回率: {optimal_results['metrics']['recall']:.4f}")
        
        # 比较不同阈值
        print("\n5. 阈值比较:")
        for threshold in [0.0, optimal_results['best_threshold']]:
            eval_result = evaluate_threshold(contribution_data, threshold)
            if eval_result:
                print(f"   阈值 {threshold:.6f}:")
                print(f"     检测样本数: {eval_result['num_detected']}")
                print(f"     检测率: {eval_result['detection_rate']*100:.2f}%")
                print(f"     精确率: {eval_result['precision']:.4f}")
                print(f"     召回率: {eval_result['recall']:.4f}")
                print(f"     F1分数: {eval_result['f1_score']:.4f}")
                print(f"     TP: {eval_result['tp']}, FP: {eval_result['fp']}, FN: {eval_result['fn']}, TN: {eval_result['tn']}")
                print()
    
    # 推荐建议
    print("\n6. 推荐建议:")
    if len(contribution_data['flipped_scores']) > 0:
        if optimal_results and optimal_results['best_f1'] > 0.7:
            print("   ✅ 检测性能良好，推荐使用最佳阈值进行检测")
        elif optimal_results and optimal_results['best_f1'] > 0.5:
            print("   ⚠️ 检测性能中等，建议结合其他方法验证")
        else:
            print("   ❌ 检测性能较差，建议采用其他方法或调整数据")
        
        flipped_negative_ratio = np.sum(contribution_data['flipped_scores'] < 0) / len(contribution_data['flipped_scores'])
        if flipped_negative_ratio > 0.8:
            print("   💡 大部分翻转样本贡献值为负，符合理论预期")
        else:
            print("   🤔 部分翻转样本贡献值为正，需要进一步分析")
    else:
        print("   ℹ️ 没有翻转样本，无法进行错误检测分析")

def main():
    """主函数"""
    print("🚀 开始Shapley值贡献分析和阈值优化")
    
    # 加载数据
    try:
        shapley_data = load_shapley_data('resnet')
    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
        print("请先运行相关脚本生成Shapley值数据")
        return
    
    # 分析贡献值分布
    print("\n📈 分析贡献值分布...")
    contribution_data = analyze_contribution_distribution(shapley_data)
    
    # 寻找最佳阈值
    print("\n🎯 寻找最佳阈值...")
    optimal_results = {}
    if len(contribution_data['flipped_scores']) > 0:
        best_threshold, best_f1, metrics = find_optimal_threshold(contribution_data)
        optimal_results = {
            'best_threshold': best_threshold,
            'best_f1': best_f1,
            'metrics': metrics
        }
        print(f"   最佳阈值: {best_threshold:.6f}")
        print(f"   最佳F1分数: {best_f1:.4f}")
    else:
        print("   没有翻转样本，无法计算最佳阈值")
    
    # 打印详细统计
    print_detailed_statistics(contribution_data, optimal_results)
    
    # 生成可视化
    print("\n🎨 生成可视化结果...")
    save_path = os.path.join(os.getcwd(), 'threshold_analysis_results.png')
    plot_analysis_results(contribution_data, optimal_results, save_path)
    
    print("\n✅ 分析完成!")

if __name__ == "__main__":
    main()