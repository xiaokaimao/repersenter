#!/usr/bin/env python3
"""
快速测试MMLU支持
验证数据加载和基本功能
"""

import sys
sys.path.append('.')

def test_mmlu_loading():
    """测试MMLU数据加载"""
    print("🧪 快速测试MMLU数据加载...")
    
    try:
        import utils
        
        # 测试MMLU数据加载
        train_loader, test_loader, _, _, num_classes, original_labels, flipped_indices = utils.get_text_data_loaders(
            dataset_name="mmlu",
            tokenizer_name="/opt/models/Qwen3-4B-Base",  # 使用当前配置的模型
            batch_size=4,  # 小批次测试
            label_noise_rate=0.0
        )
        
        print(f"✅ MMLU数据加载成功!")
        print(f"  训练样本数: {len(train_loader.dataset)}")
        print(f"  测试样本数: {len(test_loader.dataset)}")
        print(f"  类别数: {num_classes}")
        
        # 检查第一个batch
        first_batch = next(iter(train_loader))
        print(f"  Batch shape: {first_batch['input_ids'].shape}")
        print(f"  Label range: {first_batch['labels'].min().item()} - {first_batch['labels'].max().item()}")
        
        # 显示样本示例
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("/opt/models/Qwen3-4B-Base")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        sample_text = tokenizer.decode(first_batch['input_ids'][0], skip_special_tokens=True)
        sample_label = first_batch['labels'][0].item()
        
        print(f"\n📝 MMLU样本示例:")
        print(f"  文本: {sample_text[:300]}...")
        print(f"  标签: {sample_label} ({'ABCD'[sample_label]})")
        print(f"  文本长度: {len(sample_text)} 字符")
        
        return True
        
    except Exception as e:
        print(f"❌ MMLU测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_datasets():
    """比较三个数据集的特点"""
    print("\n📊 数据集对比:")
    print("=" * 60)
    
    datasets_info = {
        "IMDB": {
            "类型": "情感分析",
            "类别数": 2,
            "格式": "电影评论文本 → 正面/负面",
            "基准准确率": "50% (随机)",
            "任务难度": "中等"
        },
        "ARC-Challenge": {
            "类型": "科学推理",
            "类别数": 4,
            "格式": "科学问题+选项 → A/B/C/D",
            "基准准确率": "25% (随机)",
            "任务难度": "困难"
        },
        "MMLU": {
            "类型": "综合知识",
            "类别数": 4,
            "格式": "57学科问题+选项 → A/B/C/D",
            "基准准确率": "25% (随机)",
            "任务难度": "非常困难"
        }
    }
    
    for dataset, info in datasets_info.items():
        print(f"\n🎯 {dataset}:")
        for key, value in info.items():
            print(f"   {key}: {value}")

def show_usage():
    """显示使用方法"""
    print("\n🚀 使用方法:")
    print("=" * 50)
    
    print("\n1️⃣ 切换到MMLU:")
    print("   python switch_to_arc.py mmlu")
    
    print("\n2️⃣ 训练MMLU模型:")
    print("   python experiments/transformer/finetune.py")
    
    print("\n3️⃣ 计算Shapley值:")
    print("   python experiments/transformer/run_shapley.py")
    
    print("\n4️⃣ 分析数据质量:")
    print("   python experiments/analysis/run_analysis.py --type transformer")
    
    print("\n💡 提示:")
    print("   - MMLU有57个学科，数据量较大")
    print("   - 建议使用较小的batch size (32)")
    print("   - 推理任务可能需要更多训练轮数")
    print("   - Shapley分析可以揭示模型依赖的知识类型")

if __name__ == "__main__":
    print("🧠 MMLU (Massive Multitask Language Understanding) 支持测试")
    print("=" * 70)
    
    # 比较数据集
    compare_datasets()
    
    # 测试MMLU加载
    success = test_mmlu_loading()
    
    if success:
        print("\n✅ MMLU支持测试成功!")
        show_usage()
        
        print("\n🎊 现在你的项目支持三种类型的文本分类任务:")
        print("   📺 IMDB - 情感分析")
        print("   🧪 ARC-Challenge - 科学推理") 
        print("   🎓 MMLU - 综合知识评估")
        print("\n完全使用同一套代码，只需要切换数据集配置！")
    else:
        print("\n❌ MMLU测试失败，请检查网络连接和依赖环境")
        print("需要确保能访问HuggingFace数据集")