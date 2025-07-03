"""
基于Shapley值的错误检测和数据质量分析
用于识别标签噪声、异常样本和低质量数据
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
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
    
    return torch.load(load_path, map_location='cpu')


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


def detect_mislabeled_samples(shapley_data, threshold=0.0):
    """
    检测可能的错误标签样本基于对原始标签的贡献
    
    Args:
        shapley_data: 包含Shapley值和标签信息的数据
        threshold: 贡献阈值，低于此值的样本被认为是可疑的（默认0.0）
    
    Returns:
        dict: 包含检测结果的字典
    """
    shapley_vectors = shapley_data['shapley_vectors']
    noisy_labels = shapley_data['noisy_labels']
    original_labels = shapley_data.get('original_labels', noisy_labels)
    flipped_indices = shapley_data.get('flipped_indices', [])
    
    # 计算每个样本对其原始真实标签的贡献
    contribution_scores = calculate_original_label_contribution(shapley_vectors, original_labels)
    
    # 识别负贡献样本（对原始标签有害的样本）
    suspicious_indices = np.where(contribution_scores < threshold)[0]
    
    # 基本统计
    detection_results = {
        'suspicious_indices': suspicious_indices,
        'contribution_scores': contribution_scores,
        'threshold': threshold,
        'num_suspicious': len(suspicious_indices)
    }
    
    # 分组统计
    all_indices = np.arange(len(original_labels))
    correct_indices = np.setdiff1d(all_indices, flipped_indices)
    
    if len(flipped_indices) > 0:
        flipped_scores = contribution_scores[flipped_indices]
        correct_scores = contribution_scores[correct_indices]
        
        detection_results.update({
            'flipped_scores_mean': np.mean(flipped_scores),
            'flipped_scores_std': np.std(flipped_scores),
            'correct_scores_mean': np.mean(correct_scores),
            'correct_scores_std': np.std(correct_scores),
        })
        
        # 计算检测性能
        # 确保类型一致，转换为整数集合
        true_flipped = set(int(x) for x in flipped_indices)
        detected_suspicious = set(int(x) for x in suspicious_indices)
        
        true_positives = len(true_flipped & detected_suspicious)
        false_positives = len(detected_suspicious - true_flipped)
        false_negatives = len(true_flipped - detected_suspicious)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        detection_results.update({
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'actual_flipped_count': len(flipped_indices)
        })
    
    return detection_results


def visualize_detection_results(shapley_data, save_dir):
    """
    可视化错误检测结果（基于对原始标签的贡献）
    """
    shapley_vectors = shapley_data['shapley_vectors']
    original_labels = shapley_data.get('original_labels', shapley_data['noisy_labels'])
    flipped_indices = shapley_data.get('flipped_indices', [])
    
    # 计算对原始标签的贡献
    contribution_scores = calculate_original_label_contribution(shapley_vectors, original_labels)
    
    # 分组
    all_indices = np.arange(len(original_labels))
    correct_indices = np.setdiff1d(all_indices, flipped_indices)
    
    flipped_scores = contribution_scores[flipped_indices]
    correct_scores = contribution_scores[correct_indices]
    
    plt.figure(figsize=(12, 8))
    
    # 绘制分布直方图
    plt.hist(correct_scores, bins=50, density=True, alpha=0.7, 
             label=f'正确标签样本 (均值: {np.mean(correct_scores):.3f})', color='green')
    plt.hist(flipped_scores, bins=20, density=True, alpha=0.8, 
             label=f'错误标签样本 (均值: {np.mean(flipped_scores):.3f})', color='red')
    
    # 添加决策边界
    plt.axvline(x=0, color='black', linestyle='--', linewidth=2, label='决策边界 (x=0)')
    
    plt.title('对原始真实标签的Shapley贡献分布', fontsize=16, fontweight='bold')
    plt.xlabel('Shapley贡献值', fontsize=12)
    plt.ylabel('密度', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'contribution_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path


def generate_analysis_report(detection_results, experiment_type, save_dir):
    """生成分析报告（基于原始标签贡献方法）"""
    report_lines = []
    report_lines.append("# Shapley值错误检测分析报告\n")
    report_lines.append(f"实验类型: {experiment_type.upper()}\n")
    report_lines.append(f"分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    report_lines.append("## 检测方法\n")
    report_lines.append("- **检测方法**: 原始标签贡献法\n")
    report_lines.append("- **原理**: 计算每个样本对其原始真实标签的Shapley贡献\n")
    report_lines.append("- **阈值**: 贡献值 < 0 的样本被标记为可疑\n\n")
    
    report_lines.append("## 检测结果概述\n")
    total_samples = len(detection_results['contribution_scores'])
    report_lines.append(f"- 总样本数: {total_samples}\n")
    report_lines.append(f"- 可疑样本数: {detection_results['num_suspicious']}\n")
    report_lines.append(f"- 可疑样本比例: {detection_results['num_suspicious']/total_samples*100:.2f}%\n")
    report_lines.append(f"- 检测阈值: {detection_results['threshold']:.6f}\n\n")
    
    if 'precision' in detection_results:
        report_lines.append("## 检测性能指标\n")
        report_lines.append(f"- 精确率 (Precision): {detection_results['precision']:.4f}\n")
        report_lines.append(f"- 召回率 (Recall): {detection_results['recall']:.4f}\n")
        report_lines.append(f"- F1分数: {detection_results['f1_score']:.4f}\n")
        report_lines.append(f"- 真正例 (TP): {detection_results['true_positives']}\n")
        report_lines.append(f"- 假正例 (FP): {detection_results['false_positives']}\n")
        report_lines.append(f"- 假负例 (FN): {detection_results['false_negatives']}\n")
        report_lines.append(f"- 实际翻转样本数: {detection_results['actual_flipped_count']}\n\n")
    
    # 贡献分数统计
    scores = detection_results['contribution_scores']
    report_lines.append("## 原始标签贡献统计\n")
    report_lines.append(f"- 平均值: {np.mean(scores):.6f}\n")
    report_lines.append(f"- 标准差: {np.std(scores):.6f}\n")
    report_lines.append(f"- 最小值: {np.min(scores):.6f}\n")
    report_lines.append(f"- 最大值: {np.max(scores):.6f}\n")
    report_lines.append(f"- 5%分位数: {np.percentile(scores, 5):.6f}\n")
    report_lines.append(f"- 95%分位数: {np.percentile(scores, 95):.6f}\n\n")
    
    # 分组统计
    if 'correct_scores_mean' in detection_results:
        report_lines.append("## 分组统计分析\n")
        report_lines.append(f"- **正确标签样本均值**: {detection_results['correct_scores_mean']:.6f}\n")
        report_lines.append(f"- **正确标签样本标准差**: {detection_results['correct_scores_std']:.6f}\n")
        report_lines.append(f"- **错误标签样本均值**: {detection_results['flipped_scores_mean']:.6f}\n")
        report_lines.append(f"- **错误标签样本标准差**: {detection_results['flipped_scores_std']:.6f}\n\n")
        
        # 效果验证
        if detection_results['flipped_scores_mean'] < 0 and detection_results['correct_scores_mean'] > 0:
            report_lines.append("**✅ 验证结果**: 错误标签样本的平均贡献为负值，正确标签样本为正值，符合理论预期。\n\n")
        else:
            report_lines.append("**⚠️ 注意**: 贡献值分布可能与理论预期不符，需要进一步分析。\n\n")
    
    report_lines.append("## 建议\n")
    if 'f1_score' in detection_results:
        if detection_results['f1_score'] > 0.7:
            report_lines.append("- ✅ 检测性能良好，可以信任检测结果\n")
        elif detection_results['f1_score'] > 0.5:
            report_lines.append("- ⚠️ 检测性能中等，建议结合其他方法验证\n")
        else:
            report_lines.append("- ❌ 检测性能较差，建议调整阈值或方法\n")
    
    report_lines.append("- 建议手动检查贡献分数最低的样本\n")
    report_lines.append("- 可以考虑移除或重新标注检测到的可疑样本\n")
    report_lines.append("- 监控数据质量，定期进行错误检测分析\n")
    
    # 保存报告
    report_path = os.path.join(save_dir, f'analysis_report_{experiment_type}.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report_lines)
    
    return report_path

def generate_comparison_report(detection_results_norm, detection_results_contrib, experiment_type, save_dir):
    """生成两种方法的比较报告"""
    report_lines = []
    report_lines.append("# Shapley值错误检测方法比较报告\\n")
    report_lines.append(f"实验类型: {experiment_type.upper()}\\n")
    report_lines.append(f"分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
    
    # 方法对比
    report_lines.append("## 检测方法对比\\n")
    report_lines.append("### 方法1: L2范数检测法\\n")
    report_lines.append("- **原理**: 使用Shapley向量的L2范数作为样本有用性指标\\n")
    report_lines.append("- **阈值**: 最低5%的样本被标记为可疑\\n")
    report_lines.append(f"- **检测样本数**: {detection_results_norm['num_suspicious']}\\n")
    
    if 'f1_score' in detection_results_norm:
        report_lines.append(f"- **精确率**: {detection_results_norm['precision']:.4f}\\n")
        report_lines.append(f"- **召回率**: {detection_results_norm['recall']:.4f}\\n")
        report_lines.append(f"- **F1分数**: {detection_results_norm['f1_score']:.4f}\\n")
    
    report_lines.append("\\n### 方法2: 原始标签贡献检测法 (experiment.py方法)\\n")
    report_lines.append("- **原理**: 计算每个样本对其原始真实标签的Shapley贡献\\n")
    report_lines.append("- **阈值**: 贡献值 < 0 的样本被标记为可疑\\n")
    report_lines.append(f"- **检测样本数**: {detection_results_contrib['num_suspicious']}\\n")
    
    if 'f1_score' in detection_results_contrib:
        report_lines.append(f"- **精确率**: {detection_results_contrib['precision']:.4f}\\n")
        report_lines.append(f"- **召回率**: {detection_results_contrib['recall']:.4f}\\n")
        report_lines.append(f"- **F1分数**: {detection_results_contrib['f1_score']:.4f}\\n")
    
    # 贡献分数统计
    if 'correct_scores_mean' in detection_results_contrib:
        report_lines.append("\\n## 原始标签贡献统计\\n")
        report_lines.append(f"- **正确标签样本均值**: {detection_results_contrib['correct_scores_mean']:.6f}\\n")
        report_lines.append(f"- **正确标签样本标准差**: {detection_results_contrib['correct_scores_std']:.6f}\\n")
        report_lines.append(f"- **错误标签样本均值**: {detection_results_contrib['flipped_scores_mean']:.6f}\\n")
        report_lines.append(f"- **错误标签样本标准差**: {detection_results_contrib['flipped_scores_std']:.6f}\\n")
        
        # 效果验证
        if detection_results_contrib['flipped_scores_mean'] < 0 and detection_results_contrib['correct_scores_mean'] > 0:
            report_lines.append("\\n**✅ 验证结果**: 错误标签样本的平均贡献为负值，正确标签样本为正值，符合理论预期。\\n")
        else:
            report_lines.append("\\n**⚠️ 注意**: 贡献值分布可能与理论预期不符，需要进一步分析。\\n")
    
    # 方法比较
    report_lines.append("\\n## 方法优缺点比较\\n")
    report_lines.append("### L2范数方法\\n")
    report_lines.append("- **优点**: 考虑样本对所有类别的整体贡献，更全面\\n")
    report_lines.append("- **缺点**: 需要手动设定百分位阈值，可能不够直观\\n")
    
    report_lines.append("\\n### 原始标签贡献方法\\n")
    report_lines.append("- **优点**: 直接针对真实标签，逻辑清晰，阈值自然(0)\\n")
    report_lines.append("- **缺点**: 仅考虑对正确类别的贡献，可能忽略其他信息\\n")
    
    # 建议
    report_lines.append("\\n## 使用建议\\n")
    if 'f1_score' in detection_results_norm and 'f1_score' in detection_results_contrib:
        if detection_results_contrib['f1_score'] > detection_results_norm['f1_score']:
            report_lines.append("- 📊 **推荐**: 原始标签贡献方法在当前数据集上表现更好\\n")
        else:
            report_lines.append("- 📊 **推荐**: L2范数方法在当前数据集上表现更好\\n")
    
    report_lines.append("- 💡 **结合使用**: 可以同时使用两种方法，取交集获得高置信度的错误样本\\n")
    report_lines.append("- 🔍 **人工验证**: 对检测到的可疑样本进行人工审核，提高准确性\\n")
    
    # 保存报告
    report_path = os.path.join(save_dir, f'comparison_report_{experiment_type}.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report_lines)
    
    return report_path

def run_error_analysis(experiment_type='resnet', threshold=0.0):
    """运行完整的错误分析（基于原始标签贡献方法）"""
    print(f"🔍 开始错误分析: {experiment_type.upper()}")
    
    # 创建结果目录
    save_dir = os.path.join(settings.RESULTS_DIR, f'error_analysis_{experiment_type}')
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载数据
    print("📊 加载Shapley值数据...")
    try:
        shapley_data = load_shapley_data(experiment_type)
    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
        print("请先运行Shapley值计算脚本生成数据。")
        return
    
    # 检测错误标签（使用原始标签贡献方法）
    print("🕵️ 检测可疑样本（基于原始标签贡献）...")
    detection_results = detect_mislabeled_samples(shapley_data, threshold)
    
    # 可视化检测结果
    print("🎨 生成可视化结果...")
    detection_plot_path = visualize_detection_results(shapley_data, save_dir)
    
    # 生成报告
    print("📝 生成分析报告...")
    report_path = generate_analysis_report(detection_results, experiment_type, save_dir)
    
    # 打印结果
    print(f"\n=== 🎯 错误分析结果 ===")
    total_samples = len(detection_results['contribution_scores'])
    print(f"总样本数: {total_samples}")
    print(f"可疑样本数: {detection_results['num_suspicious']}")
    print(f"可疑样本比例: {detection_results['num_suspicious']/total_samples*100:.2f}%")
    
    if 'f1_score' in detection_results:
        print(f"检测精确率: {detection_results['precision']:.4f}")
        print(f"检测召回率: {detection_results['recall']:.4f}")
        print(f"F1分数: {detection_results['f1_score']:.4f}")
    
    # 显示贡献统计
    if 'correct_scores_mean' in detection_results:
        print(f"\n=== 📈 贡献统计 ===")
        print(f"正确样本均值: {detection_results['correct_scores_mean']:.6f}")
        print(f"错误样本均值: {detection_results['flipped_scores_mean']:.6f}")
        verification = "✅ 符合预期" if detection_results['flipped_scores_mean'] < 0 and detection_results['correct_scores_mean'] > 0 else "⚠️ 需要分析"
        print(f"理论验证: {verification}")
    
    print(f"\n📁 结果保存在: {save_dir}")
    print(f"🎨 检测结果图: {detection_plot_path}")
    print(f"📝 分析报告: {report_path}")
    
    return {
        'detection_results': detection_results,
        'save_dir': save_dir,
        'plots': {
            'detection': detection_plot_path
        },
        'report': report_path
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="基于Shapley值的错误检测分析（原始标签贡献方法）")
    parser.add_argument("--type", choices=["resnet", "transformer"], 
                       default="resnet", help="实验类型")
    parser.add_argument("--threshold", type=float, default=0.0,
                       help="检测阈值，低于此值的样本被认为是可疑的 (default: 0.0)")
    args = parser.parse_args()
    
    run_error_analysis(args.type, args.threshold)