"""
简单的多卡训练工具函数
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.data_utils import set_seed, get_transformer_model, get_model
import models
import config.settings as settings

def train_model_with_accelerate(
    train_loader, 
    test_loader, 
    model_config, 
    num_classes, 
    experiment_name="Training",
    model_type="transformer",  # "transformer" 或 "resnet"
    save_model=True  # 是否保存最佳模型
):
    """
    简单的多卡训练函数 - 支持 Transformer 和 ResNet
    
    Args:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器  
        model_config: 模型配置字典，包含:
            - model_name: 模型名称
            - learning_rate: 学习率
            - lambda_reg: L2正则化
            - num_epochs: 训练轮数
            - use_bf16: 是否使用bf16 (可选)
        num_classes: 类别数
        experiment_name: 实验名称
        model_type: "transformer" 或 "resnet"
        save_model: 是否保存最佳模型
    
    Returns:
        history: 训练历史 {'epoch': [...], 'accuracy': [...]}
    """
    # 初始化 Accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=model_config.get("gradient_accumulation_steps", 1),
        kwargs_handlers=[ddp_kwargs]
    )
    
    if accelerator.is_main_process:
        print(f"🚀 开始训练: {experiment_name}")
        print(f"模型类型: {model_type}")
        print(f"模型: {model_config['model_name']}")
        print(f"GPU数量: {accelerator.num_processes}")
    
    set_seed(settings.SEED)
    
    # 创建模型
    if model_type == "transformer":
        model = get_transformer_model(
            model_name=model_config['model_name'],
            num_classes=num_classes,
            use_bf16=model_config.get('use_bf16', False)
        )
        # Transformer只训练分类头
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    else:
        # ResNet
        model = get_model(
            model_name=model_config['model_name'],
            num_classes=num_classes,
            from_scratch=True
        )
        # ResNet分层设置正则化
        classifier_keywords = ['fc', 'linear', 'classifier']
        classifier_params = []
        base_params = []
        for name, param in model.named_parameters():
            if any(kw in name for kw in classifier_keywords):
                classifier_params.append(param)
            else:
                base_params.append(param)
        trainable_params = [
            {'params': base_params, 'weight_decay': 0},
            {'params': classifier_params, 'weight_decay': model_config['lambda_reg']}
        ]
    
    # 创建优化器
    if model_type == "transformer":
        optimizer = optim.AdamW(
            trainable_params,
            lr=model_config['learning_rate'],
            weight_decay=model_config['lambda_reg']
        )
    else:
        optimizer = optim.AdamW(trainable_params, lr=model_config['learning_rate'])
    
    criterion = nn.CrossEntropyLoss()
    
    if accelerator.is_main_process:
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"可训练参数: {num_trainable / 1e6:.2f}M")
    
    # Accelerator prepare
    model, optimizer, train_loader, test_loader, criterion = accelerator.prepare(
        model, optimizer, train_loader, test_loader, criterion
    )
    
    # 训练循环
    history = {'epoch': [], 'accuracy': []}
    best_accuracy = 0.0
    
    # 设置保存路径
    model_save_path = None
    if save_model and accelerator.is_main_process:
        output_dir = settings.CHECKPOINT_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        if model_type == "transformer":
            model_name_safe = model_config['model_name'].replace('/', '_')
            model_save_path = os.path.join(
                output_dir, 
                f"finetuned_{model_name_safe}_on_{model_config.get('dataset_name', 'unknown')}_lambda_{model_config['lambda_reg']}_noise_{model_config.get('label_noise_rate', 0.0)}.pth"
            )
        else:
            model_save_path = os.path.join(
                output_dir, 
                f"{model_config['model_name']}_on_{model_config.get('dataset_name', 'unknown')}_lambda_{model_config['lambda_reg']}_noise_{model_config.get('label_noise_rate', 0.0)}.pth"
            )
        
        print(f"模型将保存到: {model_save_path}")
    
    for epoch in range(model_config["num_epochs"]):
        # 训练
        model.train()
        running_loss = 0.0
        num_batches = 0
        
        train_progress = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1} Training", 
            disable=not accelerator.is_main_process
        )
        
        for batch in train_progress:
            with accelerator.accumulate(model):
                if model_type == "transformer":
                    # Transformer数据格式
                    inputs = {k: v for k, v in batch.items() if k != 'labels'}
                    labels = batch['labels']
                    with accelerator.autocast():
                        outputs = model(**inputs)
                        loss = criterion(outputs.logits, labels)
                else:
                    # ResNet数据格式
                    inputs, labels = batch
                    with accelerator.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                num_batches += 1
                
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    optimizer.step()
                    optimizer.zero_grad()
                
                train_progress.set_postfix(loss=f"{loss.item():.4f}")
        
        # 评估
        model.eval()
        correct = 0
        total = 0
        
        eval_progress = tqdm(
            test_loader, 
            desc=f"Epoch {epoch+1} Evaluating", 
            disable=not accelerator.is_main_process
        )
        
        with torch.no_grad():
            for batch in eval_progress:
                if model_type == "transformer":
                    inputs = {k: v for k, v in batch.items() if k != 'labels'}
                    labels = batch['labels']
                    with accelerator.autocast():
                        outputs = model(**inputs)
                    predictions = outputs.logits.argmax(dim=-1)
                else:
                    inputs, labels = batch
                    with accelerator.autocast():
                        outputs = model(inputs)
                    predictions = outputs.argmax(dim=-1)
                
                predictions, labels = accelerator.gather_for_metrics((predictions, labels))
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        val_acc = 100 * correct / total
        train_loss = running_loss / num_batches if num_batches > 0 else 0.0
        
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1}/{model_config['num_epochs']} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Acc: {val_acc:.2f}%")
            
            # 保存最佳模型
            if save_model and val_acc > best_accuracy:
                best_accuracy = val_acc
                unwrapped_model = accelerator.unwrap_model(model)
                accelerator.save(unwrapped_model.state_dict(), model_save_path)
                print(f"📀 模型已保存 (准确率: {best_accuracy:.2f}%): {model_save_path}")
        
        history['epoch'].append(epoch + 1)
        history['accuracy'].append(val_acc)
    
    if accelerator.is_main_process:
        print(f"✅ {experiment_name} 完成")
        print(f"最终准确率: {val_acc:.2f}%")
        if save_model:
            print(f"🏆 最佳准确率: {best_accuracy:.2f}%")
            print(f"📁 模型保存路径: {model_save_path}")
    
    return history