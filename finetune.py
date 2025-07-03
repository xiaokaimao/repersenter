

import torch
import torch.nn as nn
import torch.optim as optim
import os

# 导入重构后的模块和配置
import utils
import settings

# --- 1. 主程序 ---
if __name__ == "__main__":
    # 从 settings.py 加载配置
    config = settings.FINETUNE_CONFIG
    dataset_name = settings.DATASET
    batch_size = settings.BATCH_SIZE
    device = torch.device(settings.DEVICE if torch.cuda.is_available() else "cpu")

    print(f"使用设备: {device}")
    print(f"微调模型: {config['model_name']} on {dataset_name}")

    # --- 2. 数据加载 ---
    # for_finetuning=True 表示使用ImageNet的预处理方法
    utils.set_seed(settings.SEED)
    train_loader, test_loader, _, _, num_classes = utils.get_data_loaders(
        dataset_name=dataset_name, 
        batch_size=batch_size,
        for_finetuning=True,
        label_noise_rate=config.get('label_noise_rate', 0.0)
    )
    
    # --- 3. 模型构建 ---
    # from_scratch=False 表示加载预训练模型
    model = utils.get_model(
        model_name=config['model_name'], 
        num_classes=num_classes, 
        from_scratch=False
    ).to(device)

    # --- 4. 冻结特征提取器层 ---
    print("正在冻结特征提取器层...")
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False
    
    # 验证哪些层是可训练的
    print("\n可训练的参数:")
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
            print(f"- {name}")

    # --- 5. 定义损失函数和优化器 ---
    criterion = nn.CrossEntropyLoss()
    
    # 只将可训练的参数(分类头)传给优化器，并施加L2正则
    optimizer = optim.AdamW(
        trainable_params, 
        lr=config['learning_rate'],
        weight_decay=config['lambda_reg']
    )
    print(f"\n优化器已设置. 只优化分类头. L2正则强度λ = {config['lambda_reg']}")
    
    # --- 6. 训练与评估循环 ---
    print("\n--- 开始微调 ---")
    best_accuracy = 0.0
    output_dir = settings.CHECKPOINT_DIR
    model_save_path = os.path.join(output_dir, f"finetuned_{config['model_name']}_on_{dataset_name}_lambda_{config['lambda_reg']}.pth")

    for epoch in range(config["num_epochs"]):
        train_loss = utils.train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = utils.evaluate(model, test_loader, criterion, device)
        
        print(
            f"Epoch {epoch+1}/{config['num_epochs']} | "
            f"训练损失: {train_loss:.4f} | "
            f"验证损失: {val_loss:.4f} | "
            f"验证准确率: {val_acc:.2f}%"
        )
        
        # 如果找到更好的模型，则保存
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"模型已更新并保存至: {model_save_path} (准确率: {best_accuracy:.2f}%)")

    print("\n--- 微调完成 ---")
    print(f"最佳验证准确率: {best_accuracy:.2f}%")
    print(f"最终模型保存在: {model_save_path}")
