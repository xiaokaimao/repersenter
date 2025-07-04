#!/usr/bin/env python3
"""
简单的数据集切换脚本
演示如何在IMDB和ARC-Challenge之间切换，无需额外文件
"""

import sys
import os
sys.path.append('.')

def switch_dataset(target_dataset):
    """切换数据集配置"""
    print(f"🔄 切换到{target_dataset.upper()}配置...")
    
    config_file = "config/settings.py"
    
    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 数据集映射
    dataset_configs = {
        'imdb': {'name': 'imdb', 'batch_size': 64},
        'arc': {'name': 'arc-challenge', 'batch_size': 32},
        'mmlu': {'name': 'mmlu', 'batch_size': 32}
    }
    
    if target_dataset not in dataset_configs:
        print(f"❌ 不支持的数据集: {target_dataset}")
        return False
    
    config = dataset_configs[target_dataset]
    
    # 更新数据集名
    import re
    content = re.sub(
        r'TRANSFORMER_DATASET = "[^"]*"',
        f'TRANSFORMER_DATASET = "{config["name"]}"',
        content
    )
    
    # 更新批次大小
    content = re.sub(
        r'TRANSFORMER_BATCH_SIZE = \d+',
        f'TRANSFORMER_BATCH_SIZE = {config["batch_size"]}',
        content
    )
    
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ 已切换到{target_dataset.upper()}配置")
    print(f"  数据集: {config['name']}")
    print(f"  批次大小: {config['batch_size']}")
    print("现在可以直接使用原有脚本:")
    print("  python experiments/transformer/finetune.py")
    print("  python experiments/transformer/run_shapley.py")
    return True

def switch_to_arc():
    """切换到ARC-Challenge配置"""
    return switch_dataset('arc')

def switch_to_mmlu():
    """切换到MMLU配置"""
    return switch_dataset('mmlu')

def switch_to_imdb():
    """切换回IMDB配置"""
    return switch_dataset('imdb')

def show_current_config():
    """显示当前配置"""
    import config.settings as settings
    
    config = settings.TRANSFORMER_FINETUNE_CONFIG
    
    print("📋 当前Transformer配置:")
    print(f"  数据集: {config['dataset_name']}")
    print(f"  模型: {config['model_name']}")
    print(f"  批次大小: {config['batch_size']}")
    print(f"  学习率: {config['learning_rate']}")
    print(f"  训练轮数: {config['num_epochs']}")

def demonstrate_simplicity():
    """演示简化后的使用方式"""
    print("🎯 简化后的使用方式:")
    print("=" * 50)
    
    print("\n1️⃣ 训练IMDB模型:")
    print("   # 确保 TRANSFORMER_DATASET = 'imdb'")
    print("   python experiments/transformer/finetune.py")
    
    print("\n2️⃣ 训练ARC-Challenge模型:")
    print("   # 修改 TRANSFORMER_DATASET = 'arc-challenge'")
    print("   python experiments/transformer/finetune.py")
    
    print("\n3️⃣ 训练MMLU模型:")
    print("   # 修改 TRANSFORMER_DATASET = 'mmlu'")
    print("   python experiments/transformer/finetune.py")
    
    print("\n4️⃣ 计算Shapley值 (对任何数据集):")
    print("   python experiments/transformer/run_shapley.py")
    
    print("\n🔑 关键理解:")
    print("   - 底层都是文本分类任务")
    print("   - 只需要改变数据集名称")
    print("   - 微调逻辑完全相同")
    print("   - Shapley计算也完全相同")
    
    print("\n💡 数据格式差异处理:")
    print("   - IMDB: 'text' → sentiment (0/1)")
    print("   - ARC: 'question + choices' → answer (0/1/2/3)")
    print("   - MMLU: 'question + choices' → answer (0/1/2/3)")
    print("   - 数据加载器自动处理格式转换")

def test_arc_with_existing_scripts():
    """测试用现有脚本处理ARC"""
    print("\n🧪 测试现有脚本处理ARC...")
    
    try:
        # 临时切换到ARC
        switch_to_arc()
        
        # 测试数据加载
        import utils
        import importlib
        importlib.reload(utils.data_utils)  # 重新加载配置
        
        print("正在测试ARC数据加载...")
        train_loader, test_loader, _, _, num_classes, original_labels, flipped_indices = utils.get_text_data_loaders(
            dataset_name="arc-challenge",
            tokenizer_name="/opt/models/Qwen3-0.6B-Base",
            batch_size=4,
            label_noise_rate=0.0
        )
        
        print(f"✅ 测试成功!")
        print(f"  数据集: ARC-Challenge")
        print(f"  训练样本: {len(train_loader.dataset)}")
        print(f"  测试样本: {len(test_loader.dataset)}")
        print(f"  类别数: {num_classes}")
        
        # 切换回IMDB
        switch_to_imdb()
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        switch_to_imdb()  # 确保切换回去
        return False

if __name__ == "__main__":
    print("🔄 简化的ARC-Challenge支持")
    print("=" * 50)
    
    # 显示当前配置
    show_current_config()
    
    # 演示简化方式
    demonstrate_simplicity()
    
    # 提供切换选项
    print("\n" + "=" * 50)
    print("切换选项:")
    print("1. 切换到IMDB: python switch_to_arc.py imdb")
    print("2. 切换到ARC-Challenge: python switch_to_arc.py arc") 
    print("3. 切换到MMLU: python switch_to_arc.py mmlu")
    print("4. 测试多数据集支持: python switch_to_arc.py test")
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == "arc":
            switch_to_arc()
        elif command == "mmlu":
            switch_to_mmlu()
        elif command == "imdb":
            switch_to_imdb()
        elif command == "test":
            test_arc_with_existing_scripts()
        else:
            print(f"未知命令: {command}")
            print("支持的命令: imdb, arc, mmlu, test")
    
    print("\n🎉 结论: 你说得对!")
    print("多项选择题数据集 (ARC, MMLU) 都可以用现有的transformer脚本")
    print("只需要修改数据集配置即可，不需要额外文件！")
    print("都是文本分类任务的不同表现形式而已。")