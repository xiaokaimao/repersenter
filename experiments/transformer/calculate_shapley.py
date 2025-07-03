#!/usr/bin/env python3
"""
使用 Transformer 模型计算 Shapley 值的示例脚本

这个脚本展示了如何为微调后的 Qwen3 模型计算 Shapley 值。
在运行此脚本之前，请确保：
1. 已经微调了 Qwen3 模型（使用 finetune_qwen3.py）
2. 在 settings.py 中正确配置了 TRANSFORMER_SHAPLEY_CONFIG

用法:
    python calculate_shapley_transformer.py
"""

import torch
import os
import sys

# 设置导入路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# 导入计算函数 - 使用绝对路径避免命名冲突
import importlib.util
resnet_shapley_path = os.path.join(os.path.dirname(__file__), '..', 'resnet', 'calculate_shapley.py')
spec = importlib.util.spec_from_file_location("resnet_calculate_shapley", resnet_shapley_path)
resnet_shapley_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resnet_shapley_module)
calculate_shapley_vectors = resnet_shapley_module.calculate_shapley_vectors

import utils
import config.settings as settings

def main():
    # 使用专门的 Transformer 配置
    config = settings.TRANSFORMER_SHAPLEY_CONFIG
    device = torch.device(settings.DEVICE if torch.cuda.is_available() else "cpu")
    
    print("=== Shapley值计算：Transformer模型 ===")
    print(f"模型: {config['model_name']}")
    print(f"数据集: {config['dataset_name']}")
    print(f"设备: {device}")
    print()
    
    # 设置随机种子
    utils.set_seed(settings.SEED)
    
    # 加载文本数据
    print("加载数据...")
    train_loader, test_loader, _, _, num_classes, original_labels, flipped_indices = utils.get_text_data_loaders(
        dataset_name=config['dataset_name'],
        tokenizer_name=config['model_name'],
        batch_size=settings.BATCH_SIZE,
        label_noise_rate=config.get('label_noise_rate', 0.0)
    )
    
    # 加载微调后的模型
    print("加载模型...")
    checkpoint_path = None
    if 'checkpoint_name' in config:
        checkpoint_path = os.path.join(settings.CHECKPOINT_DIR, config['checkpoint_name'])
        if not os.path.exists(checkpoint_path):
            print(f"警告: 找不到checkpoint文件 {checkpoint_path}")
            print("将使用未微调的模型")
            checkpoint_path = None
        print(f"使用 checkpoint: {checkpoint_path}")
    
    model = utils.get_transformer_model(
        model_name=config['model_name'],
        num_classes=num_classes,
        checkpoint_path=checkpoint_path,
        use_bf16=config.get('use_bf16', False)
    ).to(device)
    
    # 计算 Shapley 值
    print("开始计算 Shapley 值...")
    shapley_vectors, train_labels = calculate_shapley_vectors(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        device=device,
        lambda_reg=config['lambda_reg'],
        num_test_samples_to_use=config['num_test_samples_to_use'],
        model_type='transformer'
    )
    
    # 保存结果
    save_dir = settings.RESULTS_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    model_name_safe = config['model_name'].replace('/', '_').replace('-', '_')
    save_path = os.path.join(
        save_dir, 
        f"shapley_vectors_{config['dataset_name']}_{model_name_safe}_noise_{config['label_noise_rate']}.pt"
    )
    
    torch.save({
        'shapley_vectors': shapley_vectors,
        'train_labels': train_labels,
        'original_labels': torch.tensor(original_labels),
        'flipped_indices': torch.tensor(flipped_indices),
        'config': config
    }, save_path)
    
    print(f"\n计算完成！")
    print(f"结果已保存到: {save_path}")
    print(f"Shapley 向量形状: {shapley_vectors.shape}")
    print(f"训练样本数量: {len(train_labels)}")
    
    # 简单的统计分析
    print("\n=== 简单统计分析 ===")
    shapley_norms = torch.linalg.norm(shapley_vectors, dim=1)
    print(f"Shapley 值范数的平均值: {shapley_norms.mean():.6f}")
    print(f"Shapley 值范数的标准差: {shapley_norms.std():.6f}")
    print(f"最大 Shapley 值范数: {shapley_norms.max():.6f}")
    print(f"最小 Shapley 值范数: {shapley_norms.min():.6f}")
    
    # 如果有标签噪声，显示相关统计
    if len(flipped_indices) > 0:
        print(f"\n标签噪声统计:")
        print(f"翻转样本数量: {len(flipped_indices)}")
        print(f"翻转比例: {len(flipped_indices)/len(train_labels)*100:.2f}%")

if __name__ == "__main__":
    main()