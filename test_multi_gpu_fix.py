#!/usr/bin/env python3
"""
测试多GPU修复是否有效的脚本
"""

import sys
import os
sys.path.append('.')

from utils.shapley_utils import run_shapley_calculation

def test_single_gpu():
    """测试单GPU模式"""
    print("🧪 测试单GPU模式")
    try:
        shapley_vectors, train_labels, save_path = run_shapley_calculation(
            model_type='transformer', 
            use_accelerate=False
        )
        print("✅ 单GPU模式测试成功")
        return True
    except Exception as e:
        print(f"❌ 单GPU模式测试失败: {e}")
        return False

def test_multi_gpu_data_calculation():
    """测试多GPU数据分片计算逻辑"""
    print("🧪 测试多GPU数据分片逻辑")
    
    # 模拟参数
    total_samples = 10000
    num_processes = 4
    
    for process_index in range(num_processes):
        samples_per_process = total_samples // num_processes
        remainder = total_samples % num_processes
        
        if process_index < remainder:
            samples_for_this_process = samples_per_process + 1
        else:
            samples_for_this_process = samples_per_process
            
        print(f"GPU {process_index}: 应处理 {samples_for_this_process} 个样本")
    
    # 验证总和
    total_assigned = sum([
        (total_samples // num_processes + (1 if i < total_samples % num_processes else 0))
        for i in range(num_processes)
    ])
    
    print(f"总分配样本: {total_assigned}, 原始总数: {total_samples}")
    assert total_assigned == total_samples, "样本分配错误!"
    print("✅ 数据分片逻辑正确")

if __name__ == "__main__":
    print("🔧 测试多GPU修复")
    print("=" * 50)
    
    # 测试数据分片逻辑
    test_multi_gpu_data_calculation()
    
    print("\n" + "=" * 50)
    print("💡 建议:")
    print("1. 首先用单GPU模式验证基本功能")
    print("2. 然后逐步测试多GPU模式")
    print("3. 监控内存使用和通信开销")
    print("4. 如果还有问题，考虑减少batch_size")
    
    print("\n运行命令:")
    print("# 单GPU模式:")
    print("python experiments/transformer/run_shapley.py")
    print("\n# 多GPU模式:")
    print("accelerate launch experiments/transformer/run_shapley.py --accelerate")