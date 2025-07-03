# finetune_gpt2.py

import torch
import torch.nn as nn
import torch.optim as optim
import os
# DDP imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import utils
import settings

def setup_ddp():
    """初始化分布式训练环境"""
    # 使用 NCCL 后端，这是 NVIDIA GPU 推荐的
    dist.init_process_group(backend="nccl")
    # local_rank 由 torchrun/torch.distributed.launch 自动设置
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    """清理分布式环境"""
    dist.destroy_process_group()

def main():
    # --- 0. 设置分布式环境 ---
    local_rank = setup_ddp()
    world_size = dist.get_world_size()
    is_main_process = (local_rank == 0)
    device = torch.device(f"cuda:{local_rank}")

    # --- 1. 加载 GPT-2 专属配置 ---
    config = settings.GPT2_FINETUNE_CONFIG
    if is_main_process:
        print(f"正在使用 {world_size} 个 GPU 进行分布式训练。")
        print(f"微调模型: {config['model_name']} on {config['dataset_name']}")
    
    utils.set_seed(settings.SEED)

    # --- 计算每个设备的批量大小 ---
    # 注意: 在这个脚本中，我们从全局 settings 读取 BATCH_SIZE
    global_batch_size = settings.BATCH_SIZE
    if global_batch_size % world_size != 0:
        raise ValueError(
            f"Global batch size ({global_batch_size}) must be divisible by the number of GPUs ({world_size})."
        )
    per_device_batch_size = global_batch_size // world_size
    if is_main_process:
        print(f"Global batch size: {global_batch_size}")
        print(f"Per-device batch size: {per_device_batch_size}")

    # --- 2. 使用支持分布式的函数加载文本数据 ---
    # 注意: batch_size 现在是每个GPU的batch_size
    train_loader, test_loader, _, _, num_classes, _, _ = utils.get_text_data_loaders(
        dataset_name=config['dataset_name'], 
        tokenizer_name=config['model_name'],
        batch_size=per_device_batch_size, # <-- 使用计算出的 per-device batch size
        label_noise_rate=config.get('label_noise_rate', 0.0),
        distributed=True # <-- 启用分布式采样器
    )
    
    # --- 3. 加载模型并移动到指定GPU ---
    # 模型先加载到CPU，然后移动到目标GPU，最后用DDP封装
    model = utils.get_transformer_model(
        model_name=config['model_name'], 
        num_classes=num_classes,
        device='cpu' 
    ).to(device)
    
    # --- 3.5. 使用DDP封装模型 ---
    model = DDP(model, device_ids=[local_rank])
    
    # --- 4. 优化器设置 ---
    # 在DDP中，我们仍然从原始模型(model.module)中获取需要优化的参数
    trainable_params = filter(lambda p: p.requires_grad, model.module.parameters())
    
    optimizer = optim.AdamW(
        trainable_params, 
        lr=config['learning_rate'],
        weight_decay=config['lambda_reg']
    )
    if is_main_process:
        print("\n优化器已设置. 只优化分类头，并施加L2正则。")
    
    criterion = nn.CrossEntropyLoss()
    
    # --- 5. 训练与评估循环 ---
    if is_main_process:
        print("\n--- 开始微调 GPT-2 (分布式) ---")
    
    best_accuracy = 0.0
    output_dir = settings.CHECKPOINT_DIR
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
    
    model_save_path = os.path.join(output_dir, f"finetuned_{config['model_name']}_on_{config['dataset_name']}_lambda_{config['lambda_reg']}.pth")

    for epoch in range(config["num_epochs"]):
        # 【重要】设置sampler的epoch，以确保每个epoch的shuffle都不同
        train_loader.sampler.set_epoch(epoch)
        
        # 训练函数现在使用DDP封装后的模型
        train_loss = utils.train_one_epoch_hf(model, train_loader, criterion, optimizer, device)
        
        # 所有进程都参与评估，以同步结果
        val_loss, val_acc = utils.evaluate_hf(model, test_loader, device, criterion)
        
        # 只在主进程上打印日志和保存模型
        if is_main_process:
            print(
                f"Epoch {epoch+1}/{config['num_epochs']} | "
                f"训练损失: {train_loss:.4f} | "
                f"验证损失: {val_loss:.4f} | "
                f"验证准确率: {val_acc:.2f}%"
            )
            
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                # 【重要】保存模型时，要保存 model.module 的 state_dict
                torch.save(model.module.state_dict(), model_save_path)
                print(f"模型已更新并保存至: {model_save_path} (准确率: {best_accuracy:.2f}%)")

    if is_main_process:
        print("\n--- 微调完成 ---")
        print(f"最佳验证准确率: {best_accuracy:.2f}%")
        print(f"最终模型保存在: {model_save_path}")
        
    # --- 6. 清理分布式环境 ---
    cleanup_ddp()

if __name__ == "__main__":
    main()
