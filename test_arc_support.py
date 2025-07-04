#!/usr/bin/env python3
"""
测试ARC-Challenge支持
验证数据加载、模型构建和基本功能
"""

import sys
import os
sys.path.append('.')

import torch
import utils
import config.settings as settings

def test_arc_data_loading():
    """测试ARC-Challenge数据加载"""
    print("🧪 测试ARC-Challenge数据加载...")
    
    try:
        config = settings.ARC_CHALLENGE_CONFIG
        
        # 测试数据加载
        train_loader, test_loader, _, _, num_classes, original_labels, flipped_indices = utils.get_text_data_loaders(
            dataset_name="arc-challenge",
            tokenizer_name=config['model_name'],
            batch_size=4,  # 小批次测试
            label_noise_rate=0.0
        )
        
        print(f"✅ 数据加载成功!")
        print(f"  训练样本数: {len(train_loader.dataset)}")
        print(f"  测试样本数: {len(test_loader.dataset)}")
        print(f"  类别数: {num_classes}")
        
        # 检查第一个batch
        first_batch = next(iter(train_loader))
        print(f"  Batch shape: {first_batch['input_ids'].shape}")
        print(f"  Label range: {first_batch['labels'].min().item()} - {first_batch['labels'].max().item()}")
        
        # 显示一个样本示例
        if 'input_ids' in first_batch:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
            sample_text = tokenizer.decode(first_batch['input_ids'][0], skip_special_tokens=True)
            sample_label = first_batch['labels'][0].item()
            
            print(f"\n📝 样本示例:")
            print(f"  文本: {sample_text[:200]}...")
            print(f"  标签: {sample_label} ({'ABCD'[sample_label]})")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_arc_model():
    """测试ARC模型构建"""
    print("\n🧪 测试ARC模型构建...")
    
    try:
        config = settings.ARC_CHALLENGE_CONFIG
        
        # 构建模型
        model = utils.get_transformer_model(
            model_name=config['model_name'],
            num_classes=4,  # ARC是4选1
            use_bf16=False  # 测试时使用float32
        )
        
        print(f"✅ 模型构建成功!")
        print(f"  模型类型: {type(model).__name__}")
        print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 测试前向传播
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # 创建测试输入
        batch_size = 2
        seq_length = 128
        test_input = {
            'input_ids': torch.randint(0, 1000, (batch_size, seq_length)).to(device),
            'attention_mask': torch.ones(batch_size, seq_length).to(device)
        }
        
        model.eval()
        with torch.no_grad():
            outputs = model(**test_input)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        print(f"  输出形状: {logits.shape}")
        print(f"  输出范围: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
        
        # 检查softmax输出
        probs = torch.softmax(logits, dim=-1)
        print(f"  概率分布示例: {probs[0].cpu().numpy()}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_arc_configuration():
    """测试ARC配置"""
    print("\n🧪 测试ARC配置...")
    
    try:
        config = settings.ARC_CHALLENGE_CONFIG
        
        required_keys = [
            'model_name', 'dataset_name', 'num_epochs', 
            'learning_rate', 'lambda_reg', 'batch_size'
        ]
        
        for key in required_keys:
            if key not in config:
                print(f"❌ 缺少配置项: {key}")
                return False
            
        print(f"✅ 配置验证通过!")
        print(f"  模型: {config['model_name']}")
        print(f"  数据集: {config['dataset_name']}")
        print(f"  训练轮数: {config['num_epochs']}")
        print(f"  学习率: {config['learning_rate']}")
        print(f"  批次大小: {config['batch_size']}")
        print(f"  checkpoint: {config['checkpoint_name']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置测试失败: {e}")
        return False

def show_arc_examples():
    """显示ARC数据集示例"""
    print("\n📋 ARC-Challenge数据集说明:")
    print("ARC-Challenge (AI2 Reasoning Challenge) 是一个评估AI推理能力的数据集")
    print("包含科学推理的多项选择题，每题有4个选项(A, B, C, D)")
    
    print("\n典型问题示例:")
    print("Question: Which property of a mineral can be determined just by looking at it?")
    print("A: hardness")
    print("B: color") 
    print("C: melting point")
    print("D: electrical conductivity")
    print("Answer: B")
    
    print("\n🎯 在你的项目中:")
    print("- 将问题和选项组合成一个输入文本")
    print("- 使用Transformer模型进行4分类")
    print("- 可以计算Shapley值分析哪些文本特征对推理最重要")
    print("- 支持标签噪声注入用于错误检测研究")

def main():
    """主测试函数"""
    print("🔬 ARC-Challenge支持测试")
    print("=" * 50)
    
    # 显示ARC信息
    show_arc_examples()
    
    # 测试配置
    config_ok = test_arc_configuration()
    if not config_ok:
        print("⚠️ 配置测试失败，跳过其他测试")
        return
    
    # 测试数据加载
    data_ok = test_arc_data_loading()
    if not data_ok:
        print("⚠️ 数据加载测试失败，跳过模型测试")
        return
    
    # 测试模型
    model_ok = test_arc_model()
    
    # 总结
    print("\n" + "=" * 50)
    print("🎯 测试总结:")
    print(f"  配置: {'✅' if config_ok else '❌'}")
    print(f"  数据加载: {'✅' if data_ok else '❌'}")
    print(f"  模型构建: {'✅' if model_ok else '❌'}")
    
    if all([config_ok, data_ok, model_ok]):
        print("\n🎉 ARC-Challenge支持测试全部通过!")
        print("\n📋 使用方法:")
        print("1. 训练ARC模型:")
        print("   python experiments/arc/finetune_arc.py")
        print("2. 计算Shapley值:")
        print("   # 先修改配置使用ARC_CHALLENGE_CONFIG")
        print("   python utils/shapley_utils.py --model-type transformer")
        print("3. 分析数据质量:")
        print("   python experiments/analysis/run_analysis.py --type transformer")
    else:
        print("\n❌ 部分测试失败，请检查环境和依赖")
        print("确保安装了: transformers, datasets, torch")

if __name__ == "__main__":
    main()