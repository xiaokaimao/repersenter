#!/usr/bin/env python
"""
调试 detect_mislabeled_samples 函数中 true_positives = 0 的问题
"""
import torch
import numpy as np
import os
import sys

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__)))
import config.settings as settings
from experiments.analysis.analysis_error import load_shapley_data, detect_mislabeled_samples

def debug_performance_calculation(experiment_type='resnet'):
    """调试性能指标计算问题"""
    print("=" * 60)
    print("🔍 调试 detect_mislabeled_samples 性能计算问题")
    print("=" * 60)
    
    # 1. 加载shapley数据
    print("\n📊 步骤1: 加载Shapley数据...")
    try:
        shapley_data = load_shapley_data(experiment_type)
        print("✅ 成功加载Shapley数据")
    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
        return
    
    # 打印数据结构信息
    print("\n📋 数据结构信息:")
    for key, value in shapley_data.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {type(value)} shape={value.shape} dtype={value.dtype}")
        elif isinstance(value, (list, np.ndarray)):
            print(f"  {key}: {type(value)} len={len(value)} dtype={type(value[0]) if len(value) > 0 else 'empty'}")
        else:
            print(f"  {key}: {type(value)} value={value}")
    
    # 提取关键数据
    flipped_indices = shapley_data.get('flipped_indices', [])
    print(f"\n🎯 步骤2: 检查 flipped_indices")
    print(f"  数据类型: {type(flipped_indices)}")
    print(f"  长度: {len(flipped_indices)}")
    
    # 转换为numpy数组（如果需要）
    if isinstance(flipped_indices, torch.Tensor):
        flipped_indices_np = flipped_indices.numpy()
        print(f"  转换为numpy后类型: {type(flipped_indices_np)}")
    elif isinstance(flipped_indices, list):
        flipped_indices_np = np.array(flipped_indices)
        print(f"  转换为numpy后类型: {type(flipped_indices_np)}")
    else:
        flipped_indices_np = flipped_indices
        print(f"  原始类型保持: {type(flipped_indices_np)}")
    
    print(f"  前10个元素: {flipped_indices_np[:10] if len(flipped_indices_np) > 0 else '无数据'}")
    print(f"  元素数据类型: {flipped_indices_np.dtype if hasattr(flipped_indices_np, 'dtype') else 'N/A'}")
    print(f"  最小值: {np.min(flipped_indices_np) if len(flipped_indices_np) > 0 else 'N/A'}")
    print(f"  最大值: {np.max(flipped_indices_np) if len(flipped_indices_np) > 0 else 'N/A'}")
    
    # 3. 调用detect_mislabeled_samples函数
    print(f"\n🕵️ 步骤3: 调用detect_mislabeled_samples函数...")
    detection_results = detect_mislabeled_samples(shapley_data, threshold=0.0)
    
    suspicious_indices = detection_results['suspicious_indices']
    print(f"\n🎯 suspicious_indices 信息:")
    print(f"  数据类型: {type(suspicious_indices)}")
    print(f"  长度: {len(suspicious_indices)}")
    print(f"  前10个元素: {suspicious_indices[:10] if len(suspicious_indices) > 0 else '无数据'}")
    print(f"  元素数据类型: {suspicious_indices.dtype if hasattr(suspicious_indices, 'dtype') else 'N/A'}")
    print(f"  最小值: {np.min(suspicious_indices) if len(suspicious_indices) > 0 else 'N/A'}")
    print(f"  最大值: {np.max(suspicious_indices) if len(suspicious_indices) > 0 else 'N/A'}")
    
    # 4. 检查两个集合的交集
    print(f"\n🔍 步骤4: 检查集合交集...")
    
    # 手动重新计算，使用相同的逻辑
    true_flipped = set(flipped_indices_np)
    detected_suspicious = set(suspicious_indices)
    
    print(f"  true_flipped (set) 长度: {len(true_flipped)}")
    print(f"  detected_suspicious (set) 长度: {len(detected_suspicious)}")
    print(f"  true_flipped 前10个元素: {list(true_flipped)[:10] if len(true_flipped) > 0 else '无数据'}")
    print(f"  detected_suspicious 前10个元素: {list(detected_suspicious)[:10] if len(detected_suspicious) > 0 else '无数据'}")
    
    # 计算交集
    intersection = true_flipped & detected_suspicious
    print(f"  交集长度: {len(intersection)}")
    print(f"  交集前10个元素: {list(intersection)[:10] if len(intersection) > 0 else '无数据'}")
    
    # 分析数据类型兼容性
    print(f"\n🔬 步骤5: 数据类型兼容性分析...")
    if len(true_flipped) > 0 and len(detected_suspicious) > 0:
        flipped_sample = list(true_flipped)[0]
        suspicious_sample = list(detected_suspicious)[0]
        print(f"  flipped_indices 元素类型: {type(flipped_sample)}")
        print(f"  suspicious_indices 元素类型: {type(suspicious_sample)}")
        print(f"  类型是否相同: {type(flipped_sample) == type(suspicious_sample)}")
        
        # 尝试直接比较
        if flipped_sample in detected_suspicious:
            print(f"  ✅ 样本 {flipped_sample} 在两个集合中都存在")
        else:
            print(f"  ❌ 样本 {flipped_sample} 不在 detected_suspicious 中")
            
        # 检查是否有相同的值但不同的类型
        if len(intersection) == 0:
            print(f"  🔍 检查值是否相同但类型不同...")
            for i, flip_idx in enumerate(list(true_flipped)[:5]):  # 只检查前5个
                for j, sus_idx in enumerate(list(detected_suspicious)[:5]):
                    if flip_idx == sus_idx:
                        print(f"    找到相同值但可能类型不同: {flip_idx}({type(flip_idx)}) == {sus_idx}({type(sus_idx)})")
                        break
    
    # 6. 查看贡献分数分布
    print(f"\n📊 步骤6: 贡献分数分析...")
    contribution_scores = detection_results['contribution_scores']
    print(f"  贡献分数统计:")
    print(f"    均值: {np.mean(contribution_scores):.6f}")
    print(f"    标准差: {np.std(contribution_scores):.6f}")
    print(f"    最小值: {np.min(contribution_scores):.6f}")
    print(f"    最大值: {np.max(contribution_scores):.6f}")
    print(f"    小于0的样本数: {np.sum(contribution_scores < 0)}")
    print(f"    小于等于0的样本数: {np.sum(contribution_scores <= 0)}")
    
    # 查看翻转样本的贡献分数
    if len(flipped_indices_np) > 0:
        flipped_scores = contribution_scores[flipped_indices_np]
        print(f"  翻转样本贡献分数:")
        print(f"    均值: {np.mean(flipped_scores):.6f}")
        print(f"    最小值: {np.min(flipped_scores):.6f}")
        print(f"    最大值: {np.max(flipped_scores):.6f}")
        print(f"    小于0的翻转样本数: {np.sum(flipped_scores < 0)}")
        print(f"    小于等于0的翻转样本数: {np.sum(flipped_scores <= 0)}")
    
    # 7. 手动重新计算性能指标
    print(f"\n🔧 步骤7: 手动重新计算性能指标...")
    
    # 确保类型一致
    flipped_indices_set = set(int(x) for x in flipped_indices_np)
    suspicious_indices_set = set(int(x) for x in suspicious_indices)
    
    true_positives_manual = len(flipped_indices_set & suspicious_indices_set)
    false_positives_manual = len(suspicious_indices_set - flipped_indices_set)
    false_negatives_manual = len(flipped_indices_set - suspicious_indices_set)
    
    print(f"  手动计算结果:")
    print(f"    True Positives: {true_positives_manual}")
    print(f"    False Positives: {false_positives_manual}")
    print(f"    False Negatives: {false_negatives_manual}")
    
    precision_manual = true_positives_manual / (true_positives_manual + false_positives_manual) if (true_positives_manual + false_positives_manual) > 0 else 0
    recall_manual = true_positives_manual / (true_positives_manual + false_negatives_manual) if (true_positives_manual + false_negatives_manual) > 0 else 0
    f1_manual = 2 * precision_manual * recall_manual / (precision_manual + recall_manual) if (precision_manual + recall_manual) > 0 else 0
    
    print(f"    Precision: {precision_manual:.4f}")
    print(f"    Recall: {recall_manual:.4f}")
    print(f"    F1 Score: {f1_manual:.4f}")
    
    # 8. 对比原始函数的结果
    print(f"\n📋 步骤8: 对比原始函数结果...")
    print(f"  原始函数结果:")
    print(f"    True Positives: {detection_results.get('true_positives', 'N/A')}")
    print(f"    False Positives: {detection_results.get('false_positives', 'N/A')}")
    print(f"    False Negatives: {detection_results.get('false_negatives', 'N/A')}")
    print(f"    Precision: {detection_results.get('precision', 'N/A'):.4f}")
    print(f"    Recall: {detection_results.get('recall', 'N/A'):.4f}")
    print(f"    F1 Score: {detection_results.get('f1_score', 'N/A'):.4f}")
    
    # 9. 问题诊断
    print(f"\n🩺 步骤9: 问题诊断...")
    if detection_results.get('true_positives', 0) == 0 and true_positives_manual > 0:
        print("  ❌ 发现问题: 原始函数计算的 true_positives = 0，但手动计算 > 0")
        print("  可能原因:")
        print("    1. 数据类型不匹配导致集合操作失败")
        print("    2. flipped_indices 或 suspicious_indices 中包含意外的数据类型")
        print("    3. 集合操作中的类型转换问题")
    elif detection_results.get('true_positives', 0) == true_positives_manual:
        print("  ✅ 原始函数和手动计算结果一致")
    else:
        print(f"  ⚠️ 原始函数和手动计算结果不一致: {detection_results.get('true_positives', 0)} vs {true_positives_manual}")
    
    # 10. 额外的调试信息
    print(f"\n🔬 步骤10: 额外调试信息...")
    
    # 检查原始标签和噪声标签
    original_labels = shapley_data.get('original_labels')
    noisy_labels = shapley_data.get('noisy_labels')
    
    if original_labels is not None and noisy_labels is not None:
        print(f"  标签信息:")
        print(f"    original_labels 类型: {type(original_labels)} 长度: {len(original_labels)}")
        print(f"    noisy_labels 类型: {type(noisy_labels)} 长度: {len(noisy_labels)}")
        
        # 验证翻转索引是否正确
        if len(flipped_indices_np) > 0:
            sample_idx = flipped_indices_np[0]
            if sample_idx < len(original_labels) and sample_idx < len(noisy_labels):
                print(f"    样本 {sample_idx}: 原始标签={original_labels[sample_idx]}, 噪声标签={noisy_labels[sample_idx]}")
                print(f"    是否确实翻转: {original_labels[sample_idx] != noisy_labels[sample_idx]}")
    
    print("=" * 60)
    print("🎯 调试完成")
    print("=" * 60)
    
    return {
        'original_results': detection_results,
        'manual_results': {
            'true_positives': true_positives_manual,
            'false_positives': false_positives_manual,
            'false_negatives': false_negatives_manual,
            'precision': precision_manual,
            'recall': recall_manual,
            'f1_score': f1_manual
        },
        'flipped_indices_info': {
            'type': type(flipped_indices),
            'length': len(flipped_indices),
            'dtype': getattr(flipped_indices_np, 'dtype', None),
            'sample_values': flipped_indices_np[:5].tolist() if len(flipped_indices_np) > 0 else []
        },
        'suspicious_indices_info': {
            'type': type(suspicious_indices),
            'length': len(suspicious_indices),
            'dtype': getattr(suspicious_indices, 'dtype', None),
            'sample_values': suspicious_indices[:5].tolist() if len(suspicious_indices) > 0 else []
        }
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="调试 detect_mislabeled_samples 性能计算问题")
    parser.add_argument("--type", choices=["resnet", "transformer"], 
                       default="resnet", help="实验类型")
    args = parser.parse_args()
    
    debug_results = debug_performance_calculation(args.type)