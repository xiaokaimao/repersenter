#!/usr/bin/env python3
"""
ViT使用示例
演示如何使用项目进行ViT模型的训练、Shapley值计算和分析
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import config.settings as settings

def print_vit_workflow():
    """打印ViT完整工作流程"""
    print("🎯 Vision Transformer (ViT) 完整工作流程")
    print("=" * 60)
    
    print("\n📋 可用的ViT模型:")
    models = [
        "google/vit-base-patch16-224",
        "google/vit-large-patch16-224", 
        "google/vit-base-patch16-384",
        "microsoft/swin-base-patch4-window7-224",
        "facebook/deit-base-distilled-patch16-224"
    ]
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")
    
    print(f"\n⚙️ 当前ViT配置:")
    config = settings.VIT_CONFIG
    print(f"  模型: {config['model_name']}")
    print(f"  数据集: {config['dataset_name']}")
    print(f"  图像大小: {config['image_size']}")
    print(f"  批次大小: {config['batch_size']}")
    print(f"  训练轮数: {config['num_epochs']}")
    print(f"  学习率: {config['learning_rate']}")
    print(f"  正则化: {config['lambda_reg']}")
    print(f"  标签噪声: {config['label_noise_rate']}")
    
    print(f"\n🚀 完整工作流程:")
    
    print(f"\n1️⃣ 训练ViT模型:")
    print(f"   # 单GPU训练")
    print(f"   python experiments/vit/train.py")
    print(f"   ")
    print(f"   # 多GPU训练") 
    print(f"   accelerate launch experiments/vit/train.py")
    
    print(f"\n2️⃣ 计算Shapley值:")
    print(f"   # 单GPU模式")
    print(f"   python experiments/vit/run_shapley.py")
    print(f"   ")
    print(f"   # 多GPU加速模式")
    print(f"   accelerate launch experiments/vit/run_shapley.py --accelerate")
    print(f"   ")
    print(f"   # 使用统一工具")
    print(f"   python utils/shapley_utils.py --model-type vit --accelerate")
    
    print(f"\n3️⃣ 数据质量分析:")
    print(f"   # 错误标签检测")
    print(f"   python experiments/analysis/analysis_error.py --type vit")
    print(f"   ")
    print(f"   # 数据价值评估")
    print(f"   python experiments/analysis/data_valuation.py --type vit")
    print(f"   ")
    print(f"   # 综合分析")
    print(f"   python experiments/analysis/run_analysis.py --type vit")
    
    print(f"\n4️⃣ 核心数据集实验:")
    print(f"   python experiments/analysis/core_set_experiment.py --type vit")
    
    print(f"\n💡 自定义配置:")
    print(f"   # 修改 config/settings.py 中的 VIT_CONFIG")
    print(f"   # 可以更改模型、数据集、超参数等")
    
    print(f"\n📊 支持的数据集:")
    print(f"   - CIFAR10 (默认)")
    print(f"   - CIFAR100")
    print(f"   - 自定义数据集 (需要修改数据加载函数)")
    
    print(f"\n🔧 依赖安装:")
    print(f"   pip install transformers torch torchvision accelerate")
    print(f"   pip install datasets tokenizers")
    print(f"   pip install matplotlib seaborn scikit-learn")

def test_vit_config():
    """测试ViT配置"""
    print("\n🧪 测试ViT配置...")
    
    try:
        config = settings.VIT_CONFIG
        print(f"✅ ViT配置加载成功")
        print(f"  - 模型: {config['model_name']}")
        print(f"  - 数据集: {config['dataset_name']}")
        print(f"  - Checkpoint: {config['checkpoint_name']}")
        
        # 测试核心数据集配置
        core_config = settings.CORE_SET_EXPERIMENT_CONFIG['vit']
        print(f"✅ 核心数据集配置OK")
        print(f"  - 核心集比例: {core_config['core_set_percent']}%")
        
    except Exception as e:
        print(f"❌ 配置测试失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print_vit_workflow()
    test_vit_config()
    
    print(f"\n🎉 ViT集成完成！")
    print(f"现在可以使用Vision Transformer进行:")
    print(f"  ✓ 图像分类微调")
    print(f"  ✓ Shapley值计算")  
    print(f"  ✓ 数据质量分析")
    print(f"  ✓ 错误检测")
    print(f"  ✓ 数据估值")