"""
统一的Shapley值分析入口脚本
支持错误检测、数据估值和综合分析
"""
import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from analysis_error import run_error_analysis
from data_valuation import run_data_valuation

def run_comprehensive_analysis(experiment_type='resnet', 
                             error_threshold=5.0, 
                             n_clusters=5):
    """运行综合分析"""
    print("🔬 开始综合Shapley值分析")
    print("=" * 50)
    
    results = {}
    
    # 1. 错误检测分析
    print("\n📍 第1步: 错误检测分析")
    try:
        error_results = run_error_analysis(experiment_type, error_threshold)
        results['error_analysis'] = error_results
        print("✅ 错误检测分析完成")
    except Exception as e:
        print(f"❌ 错误检测分析失败: {e}")
        results['error_analysis'] = None
    
    # 2. 数据价值评估
    print("\n📍 第2步: 数据价值评估")
    try:
        valuation_results = run_data_valuation(experiment_type, n_clusters)
        results['valuation_analysis'] = valuation_results
        print("✅ 数据价值评估完成")
    except Exception as e:
        print(f"❌ 数据价值评估失败: {e}")
        results['valuation_analysis'] = None
    
    # 3. 生成综合总结
    print("\n📍 第3步: 生成综合分析报告")
    try:
        summary_path = generate_comprehensive_summary(results, experiment_type)
        results['summary_report'] = summary_path
        print("✅ 综合分析报告生成完成")
    except Exception as e:
        print(f"❌ 综合报告生成失败: {e}")
        results['summary_report'] = None
    
    # 输出结果总览
    print("\n" + "=" * 50)
    print("🎯 分析完成总览")
    print("=" * 50)
    
    if results['error_analysis']:
        error_res = results['error_analysis']['detection_results']
        print(f"📊 错误检测: 发现 {error_res['num_suspicious']} 个可疑样本")
        if 'f1_score' in error_res:
            print(f"   检测性能: F1={error_res['f1_score']:.3f}, Precision={error_res['precision']:.3f}, Recall={error_res['recall']:.3f}")
    
    if results['valuation_analysis']:
        val_metrics = results['valuation_analysis']['value_metrics']
        print(f"💎 数据估值: 平均价值 {val_metrics['l2_norm'].mean():.6f} ± {val_metrics['l2_norm'].std():.6f}")
        print(f"   样本分布: {len(val_metrics['l2_norm'])} 个样本，{n_clusters} 个聚类")
    
    if results['summary_report']:
        print(f"📝 综合报告: {results['summary_report']}")
    
    return results

def generate_comprehensive_summary(results, experiment_type):
    """生成综合分析总结报告"""
    import config.settings as settings
    import pandas as pd
    
    save_dir = os.path.join(settings.RESULTS_DIR, f'comprehensive_analysis_{experiment_type}')
    os.makedirs(save_dir, exist_ok=True)
    
    report_lines = []
    report_lines.append("# Shapley值综合分析报告\n")
    report_lines.append(f"实验类型: {experiment_type.upper()}\n")
    report_lines.append(f"分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # 执行概览
    report_lines.append("## 分析执行概览\n")
    if results['error_analysis']:
        report_lines.append("- ✅ 错误检测分析: 成功完成\n")
    else:
        report_lines.append("- ❌ 错误检测分析: 执行失败\n")
    
    if results['valuation_analysis']:
        report_lines.append("- ✅ 数据价值评估: 成功完成\n")
    else:
        report_lines.append("- ❌ 数据价值评估: 执行失败\n")
    
    report_lines.append("\n")
    
    # 错误检测结果总结
    if results['error_analysis']:
        error_res = results['error_analysis']['detection_results']
        report_lines.append("## 错误检测分析总结\n")
        report_lines.append(f"- 总样本数: {len(error_res['usefulness_scores'])}\n")
        report_lines.append(f"- 可疑样本数: {error_res['num_suspicious']}\n")
        report_lines.append(f"- 可疑样本比例: {error_res['num_suspicious']/len(error_res['usefulness_scores'])*100:.2f}%\n")
        
        if 'f1_score' in error_res:
            report_lines.append(f"- 检测精确率: {error_res['precision']:.4f}\n")
            report_lines.append(f"- 检测召回率: {error_res['recall']:.4f}\n")
            report_lines.append(f"- F1分数: {error_res['f1_score']:.4f}\n")
        
        report_lines.append(f"- 检测阈值: {error_res['threshold']:.6f}\n\n")
    
    # 数据价值评估总结
    if results['valuation_analysis']:
        val_metrics = results['valuation_analysis']['value_metrics']
        categories = results['valuation_analysis']['categories']
        category_names = results['valuation_analysis']['category_names']
        
        report_lines.append("## 数据价值评估总结\n")
        report_lines.append(f"- 样本总数: {len(val_metrics['l2_norm'])}\n")
        report_lines.append(f"- 平均价值: {val_metrics['l2_norm'].mean():.6f}\n")
        report_lines.append(f"- 价值标准差: {val_metrics['l2_norm'].std():.6f}\n")
        report_lines.append(f"- 价值范围: [{val_metrics['l2_norm'].min():.6f}, {val_metrics['l2_norm'].max():.6f}]\n")
        
        report_lines.append("\n### 价值分类分布\n")
        total_samples = len(categories)
        for i, name in enumerate(category_names):
            count = sum(categories == i)
            percentage = count / total_samples * 100
            report_lines.append(f"- {name}: {count}个样本 ({percentage:.1f}%)\n")
        
        report_lines.append("\n")
    
    # 综合建议
    report_lines.append("## 综合建议与行动方案\n")
    
    if results['error_analysis'] and results['valuation_analysis']:
        error_res = results['error_analysis']['detection_results']
        val_metrics = results['valuation_analysis']['value_metrics']
        
        # 数据清理建议
        report_lines.append("### 1. 数据清理策略\n")
        if error_res['num_suspicious'] > 0:
            suspicious_ratio = error_res['num_suspicious'] / len(error_res['usefulness_scores'])
            if suspicious_ratio > 0.1:
                report_lines.append("- ⚠️ 可疑样本比例较高，建议优先清理错误标签\n")
            else:
                report_lines.append("- ✅ 可疑样本比例较低，数据质量良好\n")
            
            report_lines.append(f"- 🔍 手动检查有用性分数最低的{min(100, error_res['num_suspicious'])}个样本\n")
        
        # 数据优化建议
        report_lines.append("\n### 2. 数据优化策略\n")
        high_value_threshold = val_metrics['l2_norm'].mean() + val_metrics['l2_norm'].std()
        high_value_count = sum(val_metrics['l2_norm'] >= high_value_threshold)
        low_value_threshold = val_metrics['l2_norm'].mean() - val_metrics['l2_norm'].std()
        low_value_count = sum(val_metrics['l2_norm'] <= low_value_threshold)
        
        report_lines.append(f"- 🏆 保护高价值样本: {high_value_count}个样本价值超过均值+1σ\n")
        report_lines.append(f"- 📉 考虑移除低价值样本: {low_value_count}个样本价值低于均值-1σ\n")
        
        # 训练策略建议
        report_lines.append("\n### 3. 模型训练策略\n")
        if hasattr(val_metrics['l2_norm'], 'std'):
            value_cv = val_metrics['l2_norm'].std() / val_metrics['l2_norm'].mean()
            if value_cv > 0.5:
                report_lines.append("- 🎯 价值分布差异较大，建议使用重要性采样\n")
                report_lines.append("- ⚖️ 对高价值样本增加权重或重复采样\n")
            else:
                report_lines.append("- ✅ 价值分布相对均匀，可使用标准训练策略\n")
    
    # 后续分析建议
    report_lines.append("\n### 4. 后续分析建议\n")
    report_lines.append("- 🔄 定期重新计算Shapley值，监控数据质量变化\n")
    report_lines.append("- 📊 结合模型性能指标验证数据优化效果\n")
    report_lines.append("- 🎨 进行更细粒度的特征级别Shapley分析\n")
    report_lines.append("- 🤝 与领域专家合作验证检测到的异常样本\n")
    
    # 文件路径信息
    if results['error_analysis']:
        report_lines.append(f"\n## 详细分析结果\n")
        report_lines.append(f"- 错误检测详细结果: {results['error_analysis']['save_dir']}\n")
    
    if results['valuation_analysis']:
        if 'error_analysis' not in locals():
            report_lines.append(f"\n## 详细分析结果\n")
        report_lines.append(f"- 数据估值详细结果: {results['valuation_analysis']['save_dir']}\n")
    
    # 保存综合报告
    summary_path = os.path.join(save_dir, f'comprehensive_summary_{experiment_type}.md')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.writelines(report_lines)
    
    return summary_path

def main():
    parser = argparse.ArgumentParser(description="Shapley值综合分析工具")
    parser.add_argument("--type", choices=["resnet", "transformer"], 
                       default="resnet", help="实验类型")
    parser.add_argument("--analysis", choices=["error", "valuation", "comprehensive"],
                       default="comprehensive", help="分析类型")
    parser.add_argument("--error-threshold", type=float, default=5.0,
                       help="错误检测阈值百分位数 (default: 5.0)")
    parser.add_argument("--clusters", type=int, default=5,
                       help="聚类数量 (default: 5)")
    
    args = parser.parse_args()
    
    if args.analysis == "error":
        print("🕵️ 运行错误检测分析")
        run_error_analysis(args.type, args.error_threshold)
    elif args.analysis == "valuation":
        print("💎 运行数据价值评估")
        run_data_valuation(args.type, args.clusters)
    elif args.analysis == "comprehensive":
        print("🔬 运行综合分析")
        run_comprehensive_analysis(args.type, args.error_threshold, args.clusters)
    
    print("\n🎉 分析完成！")

if __name__ == "__main__":
    main()