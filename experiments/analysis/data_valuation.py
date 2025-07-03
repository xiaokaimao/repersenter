"""
基于Shapley值的数据价值评估
用于评估数据样本对模型性能的贡献，识别高价值和低价值数据
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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

def calculate_value_metrics(shapley_vectors):
    """计算多种数据价值指标"""
    # L2范数 - 整体价值
    l2_norms = torch.linalg.norm(shapley_vectors, dim=1).numpy()
    
    # L1范数 - 稀疏性度量
    l1_norms = torch.norm(shapley_vectors, p=1, dim=1).numpy()
    
    # 最大值 - 对某个类别的最大贡献
    max_values = torch.max(shapley_vectors, dim=1)[0].numpy()
    
    # 最小值 - 可能的负面影响
    min_values = torch.min(shapley_vectors, dim=1)[0].numpy()
    
    # 方差 - 跨类别贡献的不均匀性
    variances = torch.var(shapley_vectors, dim=1).numpy()
    
    # 熵 - 贡献的分散程度
    # 先进行softmax归一化，然后计算熵
    softmax_shapley = torch.softmax(shapley_vectors, dim=1)
    entropies = -torch.sum(softmax_shapley * torch.log(softmax_shapley + 1e-8), dim=1).numpy()
    
    return {
        'l2_norm': l2_norms,
        'l1_norm': l1_norms,
        'max_value': max_values,
        'min_value': min_values,
        'variance': variances,
        'entropy': entropies
    }

def categorize_samples_by_value(value_metrics, num_categories=5):
    """根据价值指标对样本进行分类"""
    l2_norms = value_metrics['l2_norm']
    
    # 使用分位数进行分类
    percentiles = np.linspace(0, 100, num_categories + 1)
    thresholds = np.percentile(l2_norms, percentiles)
    
    categories = np.digitize(l2_norms, thresholds) - 1
    categories = np.clip(categories, 0, num_categories - 1)
    
    category_names = [
        'Very Low Value',
        'Low Value', 
        'Medium Value',
        'High Value',
        'Very High Value'
    ][:num_categories]
    
    return categories, category_names, thresholds

def cluster_samples_by_shapley(shapley_vectors, n_clusters=5):
    """基于Shapley向量对样本进行聚类"""
    # 如果维度太高，先进行PCA降维
    if shapley_vectors.shape[1] > 50:
        pca = PCA(n_components=50)
        features = pca.fit_transform(shapley_vectors.numpy())
        print(f"PCA降维: {shapley_vectors.shape[1]} -> {features.shape[1]} (保留方差: {pca.explained_variance_ratio_.sum():.3f})")
    else:
        features = shapley_vectors.numpy()
    
    # K-means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features)
    
    return cluster_labels, kmeans

def visualize_value_distribution(value_metrics, labels, save_dir):
    """可视化数据价值分布"""
    plt.figure(figsize=(20, 15))
    
    metrics_to_plot = ['l2_norm', 'l1_norm', 'max_value', 'min_value', 'variance', 'entropy']
    metric_titles = ['L2 Norm (Overall Value)', 'L1 Norm (Sparsity)', 'Max Value (Peak Contribution)', 
                    'Min Value (Negative Impact)', 'Variance (Inconsistency)', 'Entropy (Dispersion)']
    
    for i, (metric, title) in enumerate(zip(metrics_to_plot, metric_titles)):
        values = value_metrics[metric]
        
        # 整体分布
        plt.subplot(3, 4, i*2 + 1)
        plt.hist(values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel(title)
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {title}')
        plt.axvline(np.mean(values), color='red', linestyle='--', label=f'Mean: {np.mean(values):.4f}')
        plt.axvline(np.median(values), color='green', linestyle='--', label=f'Median: {np.median(values):.4f}')
        plt.legend()
        
        # 按类别分组
        plt.subplot(3, 4, i*2 + 2)
        unique_labels = np.unique(labels)
        values_by_class = [values[labels == label] for label in unique_labels]
        plt.boxplot(values_by_class, labels=unique_labels)
        plt.xlabel('Class')
        plt.ylabel(title)
        plt.title(f'{title} by Class')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'value_distribution_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def visualize_sample_categories(value_metrics, categories, category_names, save_dir):
    """可视化样本价值分类"""
    plt.figure(figsize=(15, 10))
    
    # 分类饼图
    plt.subplot(2, 3, 1)
    category_counts = [np.sum(categories == i) for i in range(len(category_names))]
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(category_names)))
    plt.pie(category_counts, labels=category_names, autopct='%1.1f%%', colors=colors)
    plt.title('Sample Distribution by Value Category')
    
    # 每个类别的价值分布
    plt.subplot(2, 3, 2)
    l2_norms = value_metrics['l2_norm']
    for i, name in enumerate(category_names):
        mask = categories == i
        if np.any(mask):
            plt.hist(l2_norms[mask], bins=20, alpha=0.7, label=name, color=colors[i])
    plt.xlabel('L2 Norm (Overall Value)')
    plt.ylabel('Frequency')
    plt.title('Value Distribution by Category')
    plt.legend()
    
    # 散点图：不同指标的关系
    plt.subplot(2, 3, 3)
    scatter = plt.scatter(value_metrics['l2_norm'], value_metrics['entropy'], 
                         c=categories, cmap='RdYlGn', alpha=0.6)
    plt.xlabel('L2 Norm (Overall Value)')
    plt.ylabel('Entropy (Dispersion)')
    plt.title('Value vs Dispersion')
    plt.colorbar(scatter, label='Value Category')
    
    # 样本索引 vs 价值
    plt.subplot(2, 3, 4)
    plt.scatter(range(len(l2_norms)), l2_norms, c=categories, cmap='RdYlGn', alpha=0.6, s=1)
    plt.xlabel('Sample Index')
    plt.ylabel('L2 Norm (Overall Value)')
    plt.title('Sample Value Distribution')
    plt.colorbar(label='Value Category')
    
    # 累积价值贡献
    plt.subplot(2, 3, 5)
    sorted_indices = np.argsort(-l2_norms)  # 降序排列
    sorted_values = l2_norms[sorted_indices]
    cumulative_values = np.cumsum(sorted_values)
    cumulative_percentage = cumulative_values / cumulative_values[-1] * 100
    sample_percentage = np.arange(1, len(sorted_values) + 1) / len(sorted_values) * 100
    
    plt.plot(sample_percentage, cumulative_percentage, 'b-', linewidth=2)
    plt.axline((0, 0), slope=1, color='red', linestyle='--', alpha=0.7, label='Perfect Equality')
    plt.xlabel('Percentage of Samples (sorted by value)')
    plt.ylabel('Cumulative Percentage of Total Value')
    plt.title('Value Concentration Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 价值 vs 方差
    plt.subplot(2, 3, 6)
    plt.scatter(value_metrics['l2_norm'], value_metrics['variance'], 
               c=categories, cmap='RdYlGn', alpha=0.6)
    plt.xlabel('L2 Norm (Overall Value)')
    plt.ylabel('Variance (Inconsistency)')
    plt.title('Value vs Inconsistency')
    plt.colorbar(label='Value Category')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'sample_categorization.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def visualize_clustering_results(shapley_vectors, cluster_labels, value_metrics, save_dir):
    """可视化聚类结果"""
    n_clusters = len(np.unique(cluster_labels))
    
    plt.figure(figsize=(15, 10))
    
    # 使用t-SNE进行可视化（如果样本数不太多）
    if shapley_vectors.shape[0] <= 5000:
        plt.subplot(2, 3, 1)
        # 如果维度太高，先PCA降维
        if shapley_vectors.shape[1] > 50:
            pca = PCA(n_components=50)
            features_for_tsne = pca.fit_transform(shapley_vectors.numpy())
        else:
            features_for_tsne = shapley_vectors.numpy()
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, shapley_vectors.shape[0]//4))
        tsne_result = tsne.fit_transform(features_for_tsne)
        
        scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=cluster_labels, cmap='tab10', alpha=0.6)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title('Cluster Visualization (t-SNE)')
        plt.colorbar(scatter, label='Cluster')
    
    # PCA可视化
    plt.subplot(2, 3, 2)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(shapley_vectors.numpy())
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, cmap='tab10', alpha=0.6)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
    plt.title('Cluster Visualization (PCA)')
    plt.colorbar(scatter, label='Cluster')
    
    # 每个聚类的价值分布
    plt.subplot(2, 3, 3)
    l2_norms = value_metrics['l2_norm']
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        plt.hist(l2_norms[mask], bins=20, alpha=0.7, label=f'Cluster {cluster_id}')
    plt.xlabel('L2 Norm (Overall Value)')
    plt.ylabel('Frequency')
    plt.title('Value Distribution by Cluster')
    plt.legend()
    
    # 聚类统计
    plt.subplot(2, 3, 4)
    cluster_sizes = [np.sum(cluster_labels == i) for i in range(n_clusters)]
    plt.bar(range(n_clusters), cluster_sizes, color=plt.cm.tab10(np.arange(n_clusters)))
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Samples')
    plt.title('Cluster Sizes')
    
    # 每个聚类的平均价值
    plt.subplot(2, 3, 5)
    cluster_mean_values = [np.mean(l2_norms[cluster_labels == i]) for i in range(n_clusters)]
    plt.bar(range(n_clusters), cluster_mean_values, color=plt.cm.tab10(np.arange(n_clusters)))
    plt.xlabel('Cluster ID')
    plt.ylabel('Mean L2 Norm')
    plt.title('Average Value by Cluster')
    
    # 聚类价值 vs 大小
    plt.subplot(2, 3, 6)
    plt.scatter(cluster_sizes, cluster_mean_values, s=100, c=range(n_clusters), cmap='tab10')
    for i, (size, value) in enumerate(zip(cluster_sizes, cluster_mean_values)):
        plt.annotate(f'C{i}', (size, value), xytext=(5, 5), textcoords='offset points')
    plt.xlabel('Cluster Size')
    plt.ylabel('Mean Value')
    plt.title('Cluster Size vs Mean Value')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'clustering_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def generate_valuation_report(value_metrics, categories, category_names, cluster_labels, experiment_type, save_dir):
    """生成数据估值报告"""
    report_lines = []
    report_lines.append("# 数据价值评估报告\n")
    report_lines.append(f"实验类型: {experiment_type.upper()}\n")
    report_lines.append(f"分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # 基本统计
    total_samples = len(value_metrics['l2_norm'])
    report_lines.append("## 数据集概览\n")
    report_lines.append(f"- 总样本数: {total_samples}\n")
    report_lines.append(f"- 特征维度: {len(np.unique(cluster_labels))} 个聚类\n\n")
    
    # 价值分布统计
    l2_norms = value_metrics['l2_norm']
    report_lines.append("## 价值分布统计\n")
    report_lines.append(f"- 平均价值: {np.mean(l2_norms):.6f}\n")
    report_lines.append(f"- 标准差: {np.std(l2_norms):.6f}\n")
    report_lines.append(f"- 最小值: {np.min(l2_norms):.6f}\n")
    report_lines.append(f"- 最大值: {np.max(l2_norms):.6f}\n")
    report_lines.append(f"- 中位数: {np.median(l2_norms):.6f}\n\n")
    
    # 分位数分析
    percentiles = [10, 25, 75, 90, 95, 99]
    report_lines.append("## 价值分位数分析\n")
    for p in percentiles:
        value = np.percentile(l2_norms, p)
        count = np.sum(l2_norms >= value)
        report_lines.append(f"- {p}%分位数: {value:.6f} (≥此值的样本: {count}个, {count/total_samples*100:.1f}%)\n")
    report_lines.append("\n")
    
    # 价值类别分析
    report_lines.append("## 价值分类分析\n")
    for i, name in enumerate(category_names):
        count = np.sum(categories == i)
        percentage = count / total_samples * 100
        mean_value = np.mean(l2_norms[categories == i])
        report_lines.append(f"- {name}: {count}个样本 ({percentage:.1f}%), 平均价值: {mean_value:.6f}\n")
    report_lines.append("\n")
    
    # 聚类分析
    n_clusters = len(np.unique(cluster_labels))
    report_lines.append("## 聚类分析\n")
    report_lines.append(f"- 聚类数量: {n_clusters}\n")
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        size = np.sum(mask)
        mean_val = np.mean(l2_norms[mask])
        std_val = np.std(l2_norms[mask])
        report_lines.append(f"- 聚类 {cluster_id}: {size}个样本 ({size/total_samples*100:.1f}%), 平均价值: {mean_val:.6f}±{std_val:.6f}\n")
    report_lines.append("\n")
    
    # 价值集中度分析
    sorted_values = np.sort(l2_norms)[::-1]
    cumsum_values = np.cumsum(sorted_values)
    total_value = cumsum_values[-1]
    
    report_lines.append("## 价值集中度分析\n")
    concentration_points = [0.1, 0.2, 0.5]
    for point in concentration_points:
        idx = int(point * total_samples)
        concentrated_value = cumsum_values[idx]
        percentage = concentrated_value / total_value * 100
        report_lines.append(f"- 前{point*100:.0f}%的样本贡献了总价值的{percentage:.1f}%\n")
    report_lines.append("\n")
    
    # 建议
    report_lines.append("## 数据优化建议\n")
    
    # 高价值样本建议
    high_value_threshold = np.percentile(l2_norms, 90)
    high_value_count = np.sum(l2_norms >= high_value_threshold)
    report_lines.append(f"### 高价值样本 (前10%, {high_value_count}个样本)\n")
    report_lines.append("- ✅ 这些样本对模型贡献最大，建议优先保留\n")
    report_lines.append("- 💡 可以分析这些样本的共同特征，指导数据收集策略\n")
    report_lines.append("- 🔄 在数据增强时，可以重点关注这类样本\n\n")
    
    # 低价值样本建议
    low_value_threshold = np.percentile(l2_norms, 10)
    low_value_count = np.sum(l2_norms <= low_value_threshold)
    report_lines.append(f"### 低价值样本 (后10%, {low_value_count}个样本)\n")
    report_lines.append("- ⚠️ 这些样本对模型贡献较小，可考虑移除\n")
    report_lines.append("- 🔍 建议检查是否存在标签错误或数据质量问题\n")
    report_lines.append("- 💾 可以优先从训练集中移除以减少计算成本\n\n")
    
    # 聚类建议
    cluster_values = [np.mean(l2_norms[cluster_labels == i]) for i in range(n_clusters)]
    best_cluster = np.argmax(cluster_values)
    worst_cluster = np.argmin(cluster_values)
    
    report_lines.append(f"### 聚类策略建议\n")
    report_lines.append(f"- 🏆 聚类{best_cluster}价值最高，建议深入分析其特征模式\n")
    report_lines.append(f"- 📉 聚类{worst_cluster}价值最低，建议检查数据质量\n")
    report_lines.append("- 🎯 可以基于聚类结果设计分层采样策略\n")
    report_lines.append("- 🔄 考虑对不同聚类采用不同的数据增强策略\n\n")
    
    # 保存报告
    report_path = os.path.join(save_dir, f'valuation_report_{experiment_type}.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report_lines)
    
    return report_path

def run_data_valuation(experiment_type='resnet', n_clusters=5):
    """运行完整的数据价值评估"""
    print(f"💎 开始数据价值评估: {experiment_type.upper()}")
    
    # 创建结果目录
    save_dir = os.path.join(settings.RESULTS_DIR, f'data_valuation_{experiment_type}')
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载数据
    print("📊 加载Shapley值数据...")
    try:
        shapley_data = load_shapley_data(experiment_type)
    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
        print("请先运行Shapley值计算脚本生成数据。")
        return
    
    shapley_vectors = shapley_data['shapley_vectors']
    labels = shapley_data['noisy_labels']
    
    # 计算价值指标
    print("📈 计算价值指标...")
    value_metrics = calculate_value_metrics(shapley_vectors)
    
    # 样本分类
    print("🏷️ 对样本进行价值分类...")
    categories, category_names, thresholds = categorize_samples_by_value(value_metrics)
    
    # 聚类分析
    print("🧩 进行聚类分析...")
    cluster_labels, kmeans = cluster_samples_by_shapley(shapley_vectors, n_clusters)
    
    # 可视化分析
    print("🎨 生成可视化结果...")
    
    # 价值分布可视化
    dist_plot_path = visualize_value_distribution(value_metrics, labels, save_dir)
    
    # 样本分类可视化
    cat_plot_path = visualize_sample_categories(value_metrics, categories, category_names, save_dir)
    
    # 聚类结果可视化
    cluster_plot_path = visualize_clustering_results(shapley_vectors, cluster_labels, value_metrics, save_dir)
    
    # 生成报告
    print("📝 生成评估报告...")
    report_path = generate_valuation_report(
        value_metrics, categories, category_names, cluster_labels, experiment_type, save_dir
    )
    
    # 保存详细结果
    results_path = os.path.join(save_dir, 'valuation_results.pt')
    torch.save({
        'value_metrics': value_metrics,
        'categories': categories,
        'category_names': category_names,
        'cluster_labels': cluster_labels,
        'thresholds': thresholds,
        'shapley_vectors': shapley_vectors,
        'labels': labels
    }, results_path)
    
    # 打印摘要
    print(f"\n=== 💎 数据价值评估结果 ===")
    print(f"总样本数: {len(value_metrics['l2_norm'])}")
    print(f"平均价值: {np.mean(value_metrics['l2_norm']):.6f}")
    print(f"价值标准差: {np.std(value_metrics['l2_norm']):.6f}")
    print(f"聚类数量: {n_clusters}")
    
    # 分类统计
    for i, name in enumerate(category_names):
        count = np.sum(categories == i)
        percentage = count / len(categories) * 100
        print(f"{name}: {count}个样本 ({percentage:.1f}%)")
    
    print(f"\n📁 结果保存在: {save_dir}")
    print(f"📊 分布分析图: {dist_plot_path}")
    print(f"🏷️ 分类结果图: {cat_plot_path}")
    print(f"🧩 聚类分析图: {cluster_plot_path}")
    print(f"📝 评估报告: {report_path}")
    print(f"💾 详细结果: {results_path}")
    
    return {
        'value_metrics': value_metrics,
        'categories': categories,
        'category_names': category_names,
        'cluster_labels': cluster_labels,
        'save_dir': save_dir,
        'plots': {
            'distribution': dist_plot_path,
            'categorization': cat_plot_path,
            'clustering': cluster_plot_path
        },
        'report': report_path,
        'results_file': results_path
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="基于Shapley值的数据价值评估")
    parser.add_argument("--type", choices=["resnet", "transformer"], 
                       default="resnet", help="实验类型")
    parser.add_argument("--clusters", type=int, default=5,
                       help="聚类数量 (default: 5)")
    args = parser.parse_args()
    
    run_data_valuation(args.type, args.clusters)