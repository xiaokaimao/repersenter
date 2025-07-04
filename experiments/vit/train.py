#!/usr/bin/env python3
"""
ViT训练脚本
使用预训练的Vision Transformer进行图像分类微调

用法:
    # 单GPU训练
    python experiments/vit/train.py
    
    # 多GPU训练
    accelerate launch experiments/vit/train.py
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import utils
import config.settings as settings

def main():
    # 从 settings.py 加载ViT配置
    config = settings.VIT_CONFIG
    dataset_name = config['dataset_name']
    batch_size = config['batch_size']
    device = torch.device(settings.DEVICE if torch.cuda.is_available() else "cpu")

    print(f"🎯 ViT微调训练")
    print(f"使用设备: {device}")
    print(f"模型: {config['model_name']}")
    print(f"数据集: {dataset_name}")
    print(f"图像大小: {config['image_size']}")

    # --- 设置随机种子 ---
    utils.set_seed(settings.SEED)

    # --- 数据加载 ---
    print("📊 加载数据...")
    train_loader, test_loader, _, _, num_classes, original_labels, flipped_indices = utils.get_vit_data_loaders(
        dataset_name=dataset_name,
        batch_size=batch_size,
        image_size=config['image_size'],
        label_noise_rate=config.get('label_noise_rate', 0.0)
    )
    
    print(f"训练样本数: {len(train_loader.dataset)}")
    print(f"测试样本数: {len(test_loader.dataset)}")
    print(f"类别数量: {num_classes}")
    if len(flipped_indices) > 0:
        print(f"标签噪声: {len(flipped_indices)} 个样本 ({len(flipped_indices)/len(train_loader.dataset)*100:.1f}%)")

    # --- 模型构建 ---
    print("🤖 构建ViT模型...")
    model = utils.get_vit_model(
        model_name=config['model_name'],
        num_classes=num_classes,
        use_bf16=config.get('use_bf16', False),
        freeze_backbone=config.get('freeze_backbone', False)
    ).to(device)

    # --- 冻结backbone（如果配置要求） ---
    if config.get('freeze_backbone', False):
        print("🔒 冻结ViT backbone，仅训练分类头...")
        trainable_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
                print(f"  - {name}")
    else:
        print("🔓 微调整个ViT模型...")
        trainable_params = [p for p in model.parameters() if p.requires_grad]

    # --- 定义损失函数和优化器 ---
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(
        trainable_params,
        lr=config['learning_rate'],
        weight_decay=config['lambda_reg']
    )
    
    print(f"优化器: AdamW, 学习率: {config['learning_rate']}, L2正则: {config['lambda_reg']}")
    print(f"可训练参数: {sum(p.numel() for p in trainable_params):,}")

    # --- 训练循环 ---
    print("\n🚀 开始训练...")
    best_accuracy = 0.0
    output_dir = settings.CHECKPOINT_DIR
    os.makedirs(output_dir, exist_ok=True)
    model_save_path = os.path.join(output_dir, config['checkpoint_name'])

    for epoch in range(config["num_epochs"]):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Train]")
        for batch_idx, (inputs, targets) in enumerate(train_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # ViT forward pass
            if hasattr(model, 'forward') and 'pixel_values' in str(model.forward.__code__.co_varnames):
                outputs = model(pixel_values=inputs)
                logits = outputs.logits
            else:
                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = logits.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            # 更新进度条
            train_bar.set_postfix({
                'Loss': f'{train_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Val]")
            for batch_idx, (inputs, targets) in enumerate(val_bar):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # ViT forward pass
                if hasattr(model, 'forward') and 'pixel_values' in str(model.forward.__code__.co_varnames):
                    outputs = model(pixel_values=inputs)
                    logits = outputs.logits
                else:
                    outputs = model(inputs)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
                loss = criterion(logits, targets)
                val_loss += loss.item()
                _, predicted = logits.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                # 更新进度条
                val_bar.set_postfix({
                    'Loss': f'{val_loss/(batch_idx+1):.4f}',
                    'Acc': f'{100.*val_correct/val_total:.2f}%'
                })

        # 计算准确率
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}:")
        print(f"  训练 - Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%")
        print(f"  验证 - Loss: {val_loss/len(test_loader):.4f}, Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"  💾 最佳模型已保存: {model_save_path} (准确率: {best_accuracy:.2f}%)")

    print(f"\n🎉 训练完成!")
    print(f"最佳验证准确率: {best_accuracy:.2f}%")
    print(f"模型保存路径: {model_save_path}")
    
    # 显示下一步指令
    print(f"\n📋 下一步:")
    print(f"1. 计算Shapley值:")
    print(f"   python utils/shapley_utils.py --model-type vit")
    print(f"2. 或者运行多GPU Shapley计算:")
    print(f"   accelerate launch utils/shapley_utils.py --model-type vit --accelerate")

if __name__ == "__main__":
    main()