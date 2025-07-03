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

def calculate_usefulness_scores(shapley_vectors):
    """计算样本有用性分数"""
    # 使用L2范数作为有用性指标
    usefulness_scores = torch.linalg.norm(shapley_vectors, dim=1).numpy()
    return usefulness_scores

def detect_mislabeled_samples(shapley_data, threshold_percentile=5):
    """
    检测可能的错误标签样本
    
    Args:
        shapley_data: 包含Shapley值和标签信息的数据
        threshold_percentile: 低于此百分位的样本被认为是可疑的
    
    Returns:
        dict: 包含检测结果的字典
    """
    shapley_vectors = shapley_data['shapley_vectors']
    noisy_labels = shapley_data['noisy_labels']
    original_labels = shapley_data.get('original_labels', noisy_labels)
    flipped_indices = shapley_data.get('flipped_indices', [])
    
    usefulness_scores = calculate_usefulness_scores(shapley_vectors)
    
    # 计算阈值
    threshold = np.percentile(usefulness_scores, threshold_percentile)
    
    # 识别低有用性样本
    suspicious_indices = np.where(usefulness_scores < threshold)[0]
    
    # 如果有真实的翻转信息，计算检测准确率
    detection_results = {
        'suspicious_indices': suspicious_indices,
        'usefulness_scores': usefulness_scores,
        'threshold': threshold,
        'num_suspicious': len(suspicious_indices)
    }
    
    if len(flipped_indices) > 0:
        # 计算检测性能
        true_flipped = set(flipped_indices)
        detected_suspicious = set(suspicious_indices)
        
        true_positives = len(true_flipped & detected_suspicious)
        false_positives = len(detected_suspicious - true_flipped)
        false_negatives = len(true_flipped - detected_suspicious)
        true_negatives = len(usefulness_scores) - len(true_flipped) - false_positives
        
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

def analyze_shapley_distribution(shapley_vectors, labels, save_dir):
    """分析Shapley值的分布"""
    usefulness_scores = calculate_usefulness_scores(shapley_vectors)
    
    # 创建分布图
    plt.figure(figsize=(15, 10))
    
    # 子图1: 整体分布直方图
    plt.subplot(2, 3, 1)
    plt.hist(usefulness_scores, bins=50, alpha=0.7, color='skyblue')
    plt.xlabel('Shapley Value Norm')
    plt.ylabel('Frequency')
    plt.title('Distribution of Shapley Value Norms')
    plt.axvline(np.percentile(usefulness_scores, 5), color='red', linestyle='--', label='5th percentile')
    plt.axvline(np.percentile(usefulness_scores, 95), color='green', linestyle='--', label='95th percentile')
    plt.legend()
    
    # 子图2: 按类别分组的箱线图
    plt.subplot(2, 3, 2)
    unique_labels = np.unique(labels)
    scores_by_class = [usefulness_scores[labels == label] for label in unique_labels]
    plt.boxplot(scores_by_class, labels=unique_labels)
    plt.xlabel('Class')
    plt.ylabel('Shapley Value Norm')
    plt.title('Shapley Values by Class')
    
    # 子图3: 累积分布函数
    plt.subplot(2, 3, 3)
    sorted_scores = np.sort(usefulness_scores)
    cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    plt.plot(sorted_scores, cumulative)
    plt.xlabel('Shapley Value Norm')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution')
    plt.grid(True, alpha=0.3)
    
    # 子图4: 每个类别的平均Shapley值
    plt.subplot(2, 3, 4)
    class_means = [np.mean(usefulness_scores[labels == label]) for label in unique_labels]
    plt.bar(unique_labels, class_means, color='lightcoral')
    plt.xlabel('Class')
    plt.ylabel('Mean Shapley Value Norm')
    plt.title('Mean Shapley Values by Class')
    
    # 子图5: 散点图 - 样本索引 vs Shapley值
    plt.subplot(2, 3, 5)
    plt.scatter(range(len(usefulness_scores)), usefulness_scores, alpha=0.6, s=1)
    plt.xlabel('Sample Index')
    plt.ylabel('Shapley Value Norm')
    plt.title('Shapley Values vs Sample Index')
    
    # 子图6: 热力图 - Shapley向量的前几个维度
    plt.subplot(2, 3, 6)
    if shapley_vectors.shape[1] > 1:
        # 选择前10个维度或所有维度（如果少于10个）
        dims_to_show = min(10, shapley_vectors.shape[1])
        sample_indices = np.random.choice(shapley_vectors.shape[0], min(100, shapley_vectors.shape[0]), replace=False)
        heatmap_data = shapley_vectors[sample_indices, :dims_to_show].numpy()
        sns.heatmap(heatmap_data, cmap='coolwarm', center=0, cbar_kws={'label': 'Shapley Value'})
        plt.xlabel('Shapley Dimension')
        plt.ylabel('Sample (Random Selection)')
        plt.title('Shapley Vector Heatmap')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'shapley_distribution_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def visualize_detection_results(detection_results, shapley_data, save_dir):
    """可视化错误检测结果"""
    usefulness_scores = detection_results['usefulness_scores']
    suspicious_indices = detection_results['suspicious_indices']
    threshold = detection_results['threshold']
    
    # 获取标签信息
    flipped_indices = shapley_data.get('flipped_indices', [])
    
    plt.figure(figsize=(15, 10))
    
    # 子图1: 检测结果散点图
    plt.subplot(2, 3, 1)
    # 正常样本
    normal_mask = np.ones(len(usefulness_scores), dtype=bool)
    normal_mask[suspicious_indices] = False
    plt.scatter(np.where(normal_mask)[0], usefulness_scores[normal_mask], 
               alpha=0.6, s=10, color='green', label='Normal Samples')
    
    # 可疑样本
    plt.scatter(suspicious_indices, usefulness_scores[suspicious_indices], 
               alpha=0.8, s=15, color='red', label='Suspicious Samples')
    
    # 如果有真实翻转信息，标记出来
    if len(flipped_indices) > 0:
        plt.scatter(flipped_indices, usefulness_scores[flipped_indices], 
                   alpha=0.8, s=20, color='orange', marker='x', label='Actually Flipped')
    
    plt.axhline(threshold, color='black', linestyle='--', label=f'Threshold ({threshold:.4f})')
    plt.xlabel('Sample Index')
    plt.ylabel('Shapley Value Norm')
    plt.title('Mislabel Detection Results')
    plt.legend()
    
    # 子图2: 检测性能指标（如果有真实标签）
    if 'precision' in detection_results:
        plt.subplot(2, 3, 2)
        metrics = ['Precision', 'Recall', 'F1-Score']
        values = [detection_results['precision'], detection_results['recall'], detection_results['f1_score']]
        bars = plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'coral'])
        plt.ylabel('Score')
        plt.title('Detection Performance Metrics')
        plt.ylim(0, 1)
        
        # 在柱状图上显示数值
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
    
    # 子图3: 混淆矩阵（如果有真实标签）
    if len(flipped_indices) > 0:
        plt.subplot(2, 3, 3)
        
        # 创建真实标签（0=正常，1=翻转）和预测标签（0=正常，1=可疑）
        y_true = np.zeros(len(usefulness_scores))
        y_true[flipped_indices] = 1
        
        y_pred = np.zeros(len(usefulness_scores))
        y_pred[suspicious_indices] = 1
        
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Suspicious'],
                   yticklabels=['Normal', 'Flipped'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
    
    # 子图4: 有用性分数分布（正常 vs 可疑）
    plt.subplot(2, 3, 4)
    normal_scores = usefulness_scores[normal_mask]
    suspicious_scores = usefulness_scores[suspicious_indices]
    
    plt.hist(normal_scores, bins=30, alpha=0.7, label='Normal', color='green', density=True)
    plt.hist(suspicious_scores, bins=30, alpha=0.7, label='Suspicious', color='red', density=True)
    plt.axvline(threshold, color='black', linestyle='--', label='Threshold')
    plt.xlabel('Shapley Value Norm')
    plt.ylabel('Density')
    plt.title('Score Distribution: Normal vs Suspicious')
    plt.legend()
    
    # 子图5: ROC曲线（如果有真实标签）
    if len(flipped_indices) > 0:
        plt.subplot(2, 3, 5)
        from sklearn.metrics import roc_curve, auc
        
        # 使用负的有用性分数作为"异常分数"（越低越可疑）
        y_scores = -usefulness_scores
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
    
    # 子图6: 样本排序图
    plt.subplot(2, 3, 6)
    sorted_indices = np.argsort(usefulness_scores)
    sorted_scores = usefulness_scores[sorted_indices]
    
    plt.plot(range(len(sorted_scores)), sorted_scores, color='blue', alpha=0.7)
    
    # 标记可疑样本的位置
    suspicious_positions = np.where(np.isin(sorted_indices, suspicious_indices))[0]
    plt.scatter(suspicious_positions, sorted_scores[suspicious_positions], 
               color='red', s=10, alpha=0.8, label='Suspicious')
    
    # 如果有真实翻转信息
    if len(flipped_indices) > 0:
        flipped_positions = np.where(np.isin(sorted_indices, flipped_indices))[0]
        plt.scatter(flipped_positions, sorted_scores[flipped_positions], 
                   color='orange', s=15, marker='x', alpha=0.8, label='Actually Flipped')
    
    plt.axhline(threshold, color='black', linestyle='--', label='Threshold')
    plt.xlabel('Sample Rank (by Shapley Value)')
    plt.ylabel('Shapley Value Norm')
    plt.title('Samples Sorted by Usefulness')
    plt.legend()
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'mislabel_detection_results.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def generate_analysis_report(detection_results, experiment_type, save_dir):
    """生成分析报告"""
    report_lines = []
    report_lines.append("# Shapley值错误检测分析报告\n")
    report_lines.append(f"实验类型: {experiment_type.upper()}\n")
    report_lines.append(f"分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    report_lines.append("## 检测结果概述\n")
    report_lines.append(f"- 总样本数: {len(detection_results['usefulness_scores'])}\n")
    report_lines.append(f"- 可疑样本数: {detection_results['num_suspicious']}\n")
    report_lines.append(f"- 可疑样本比例: {detection_results['num_suspicious']/len(detection_results['usefulness_scores'])*100:.2f}%\n")
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
    
    # 统计信息
    scores = detection_results['usefulness_scores']
    report_lines.append("## 有用性分数统计\n")
    report_lines.append(f"- 平均值: {np.mean(scores):.6f}\n")
    report_lines.append(f"- 标准差: {np.std(scores):.6f}\n")
    report_lines.append(f"- 最小值: {np.min(scores):.6f}\n")
    report_lines.append(f"- 最大值: {np.max(scores):.6f}\n")
    report_lines.append(f"- 5%分位数: {np.percentile(scores, 5):.6f}\n")
    report_lines.append(f"- 95%分位数: {np.percentile(scores, 95):.6f}\n\n")
    
    report_lines.append("## 建议\n")
    if 'f1_score' in detection_results:
        if detection_results['f1_score'] > 0.7:
            report_lines.append("- ✅ 检测性能良好，可以信任检测结果\n")
        elif detection_results['f1_score'] > 0.5:
            report_lines.append("- ⚠️ 检测性能中等，建议结合其他方法验证\n")
        else:
            report_lines.append("- ❌ 检测性能较差，建议调整阈值或方法\n")
    
    report_lines.append("- 建议手动检查有用性分数最低的样本\n")
    report_lines.append("- 可以考虑移除或重新标注检测到的可疑样本\n")
    report_lines.append("- 监控数据质量，定期进行错误检测分析\n")
    
    # 保存报告
    report_path = os.path.join(save_dir, f'analysis_report_{experiment_type}.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report_lines)
    
    return report_path

def run_error_analysis(experiment_type='resnet', threshold_percentile=5):
    """运行完整的错误分析"""
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
    
    # 检测错误标签
    print("🕵️ 检测可疑样本...")
    detection_results = detect_mislabeled_samples(shapley_data, threshold_percentile)
    
    # 分析Shapley分布
    print("📈 分析Shapley值分布...")
    dist_plot_path = analyze_shapley_distribution(
        shapley_data['shapley_vectors'], 
        shapley_data['noisy_labels'],
        save_dir
    )
    
    # 可视化检测结果
    print("🎨 生成可视化结果...")
    detection_plot_path = visualize_detection_results(detection_results, shapley_data, save_dir)
    
    # 生成报告
    print("📝 生成分析报告...")
    report_path = generate_analysis_report(detection_results, experiment_type, save_dir)
    
    # 打印结果
    print(f"\n=== 🎯 错误分析结果 ===")
    print(f"总样本数: {len(detection_results['usefulness_scores'])}")
    print(f"可疑样本数: {detection_results['num_suspicious']}")
    print(f"可疑样本比例: {detection_results['num_suspicious']/len(detection_results['usefulness_scores'])*100:.2f}%")
    
    if 'f1_score' in detection_results:
        print(f"检测精确率: {detection_results['precision']:.4f}")
        print(f"检测召回率: {detection_results['recall']:.4f}")
        print(f"F1分数: {detection_results['f1_score']:.4f}")
    
    print(f"\n📁 结果保存在: {save_dir}")
    print(f"📊 分布分析图: {dist_plot_path}")
    print(f"🎨 检测结果图: {detection_plot_path}")
    print(f"📝 分析报告: {report_path}")
    
    return {
        'detection_results': detection_results,
        'save_dir': save_dir,
        'plots': {
            'distribution': dist_plot_path,
            'detection': detection_plot_path
        },
        'report': report_path
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="基于Shapley值的错误检测分析")
    parser.add_argument("--type", choices=["resnet", "transformer"], 
                       default="resnet", help="实验类型")
    parser.add_argument("--threshold", type=float, default=5.0,
                       help="检测阈值百分位数 (default: 5.0)")
    args = parser.parse_args()
    
    run_error_analysis(args.type, args.threshold)