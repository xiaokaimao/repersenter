#!/usr/bin/env python3
"""
ViT Shapley值计算脚本
使用统一的计算工具，支持单GPU和多GPU模式

用法:
    # 单GPU模式
    python experiments/vit/run_shapley.py
    
    # 多GPU模式
    accelerate launch experiments/vit/run_shapley.py --accelerate
"""

import sys
import os
import argparse

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.shapley_utils import run_shapley_calculation

def main():
    parser = argparse.ArgumentParser(description="ViT Shapley值计算")
    parser.add_argument("--accelerate", action="store_true", 
                       help="使用多GPU加速")
    
    args = parser.parse_args()
    
    print("🎯 ViT Shapley值计算")
    if args.accelerate:
        print("🚀 多GPU加速模式")
        print("请确保使用 'accelerate launch experiments/vit/run_shapley.py --accelerate' 运行")
    else:
        print("💻 单GPU模式")
    
    # 运行计算
    shapley_vectors, train_labels, save_path = run_shapley_calculation(
        model_type='vit', 
        use_accelerate=args.accelerate
    )
    
    if shapley_vectors is not None:
        print(f"\n✅ ViT Shapley值计算成功完成！")
        print(f"📁 结果保存在: {save_path}")
        print(f"📊 Shapley向量形状: {shapley_vectors.shape}")
        
        if args.accelerate:
            print("⚡ 多GPU加速显著提升了计算速度")
        
        print(f"\n📋 下一步分析:")
        print(f"1. 错误检测分析:")
        print(f"   python experiments/analysis/analysis_error.py --type vit")
        print(f"2. 数据价值评估:")
        print(f"   python experiments/analysis/data_valuation.py --type vit")
        print(f"3. 综合分析:")
        print(f"   python experiments/analysis/run_analysis.py --type vit")

if __name__ == "__main__":
    main()