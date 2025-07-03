#!/usr/bin/env python3
"""
Transformer Shapley值计算脚本
使用统一的计算工具，支持单GPU和多GPU模式

用法:
    # 单GPU模式
    python experiments/transformer/run_shapley.py
    
    # 多GPU模式
    accelerate launch experiments/transformer/run_shapley.py --accelerate
"""

import sys
import os
import argparse

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.shapley_utils import run_shapley_calculation

def main():
    parser = argparse.ArgumentParser(description="Transformer Shapley值计算")
    parser.add_argument("--accelerate", action="store_true", 
                       help="使用多GPU加速")
    
    args = parser.parse_args()
    
    print("🔬 Transformer Shapley值计算")
    if args.accelerate:
        print("🚀 多GPU加速模式")
        print("请确保使用 'accelerate launch experiments/transformer/run_shapley.py --accelerate' 运行")
    else:
        print("💻 单GPU模式")
    
    # 运行计算
    run_shapley_calculation(model_type='transformer', use_accelerate=args.accelerate)

if __name__ == "__main__":
    main()