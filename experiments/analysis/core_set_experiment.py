# core_set_experiment.py - 简化版本

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import utils
import config.settings as settings
from utils.train_utils import train_model_with_accelerate

def run_core_set_experiment(experiment_type='resnet'):
    """
    简化的核心数据集实验 - 使用统一的多卡训练函数
    运行方式: accelerate launch experiments/analysis/core_set_experiment.py --type transformer
    """
    config = settings.CORE_SET_EXPERIMENT_CONFIG[experiment_type]
    
    # 不需要外部accelerator，让train_utils内部处理
    print(f"🚀 Core-Set实验 ({experiment_type.upper()})")
    print(f"模型: {config['model_name']}")
    print(f"数据集: {config['dataset_name']}")
    print(f"Core-Set大小: {config['core_set_percent']}%")

    # --- 1. 加载Shapley值 ---
    if experiment_type == 'transformer':
        model_name_safe = config['model_name'].replace('/', '_')
        shapley_file_name = f"shapley_vectors_{config['dataset_name']}_{model_name_safe}_noise_{config.get('label_noise_rate', 0.0)}.pt"
    else:
        shapley_file_name = f"shapley_vectors_{config['dataset_name']}_{config['model_name']}_noise_{config.get('label_noise_rate', 0.0)}.pt"
    
    load_path = os.path.join(settings.RESULTS_DIR, shapley_file_name)
    
    if not os.path.exists(load_path):
        print(f"❌ 找不到Shapley文件: {load_path}")
        return

    shapley_data = torch.load(load_path, map_location='cpu')
    shapley_vectors = shapley_data['shapley_vectors']
    usefulness_scores = torch.linalg.norm(shapley_vectors, dim=1).numpy()
    
    # --- 2. 准备数据 ---
    # 使用与原始微调相同的batch size
    if experiment_type == 'transformer':
        batch_size = settings.TRANSFORMER_FINETUNE_CONFIG['batch_size']
        _, test_loader, train_set, _, num_classes, _, _ = utils.get_text_data_loaders(
            dataset_name=config['dataset_name'],
            tokenizer_name=config['model_name'],
            batch_size=batch_size,
            label_noise_rate=config.get('label_noise_rate', 0.0)
        )
        # 设置collator
        from transformers import AutoTokenizer, DataCollatorWithPadding
        tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)
    else:
        batch_size = settings.BATCH_SIZE
        _, test_loader, train_set, _, num_classes, _, _ = utils.get_data_loaders(
            dataset_name=config['dataset_name'], 
            batch_size=batch_size,
            shuffle_train=False,
            label_noise_rate=config.get('label_noise_rate', 0.0)
        )
        collate_fn = None
    
    # 创建核心集
    n_train = len(train_set)
    core_set_size = int(n_train * config['core_set_percent'] / 100)
    
    sorted_indices = np.argsort(usefulness_scores)[::-1]
    shapley_core_indices = sorted_indices[:core_set_size]
    shapley_core_dataset = torch.utils.data.Subset(train_set, shapley_core_indices)
    
    np.random.seed(settings.SEED)
    random_core_indices = np.random.choice(n_train, core_set_size, replace=False)
    random_core_dataset = torch.utils.data.Subset(train_set, random_core_indices)
    
    # 创建DataLoader - 简化处理，让train_utils处理accelerator prepare
    def make_loader(dataset, shuffle=True):
        return torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn
        )
    
    full_train_loader = make_loader(train_set)
    shapley_core_loader = make_loader(shapley_core_dataset)
    random_core_loader = make_loader(random_core_dataset)
    
    print(f"完整训练集: {n_train}")
    print(f"Shapley核心集: {len(shapley_core_dataset)}")
    print(f"随机核心集: {len(random_core_dataset)}")
    
    # --- 3. 训练配置 ---
    # 使用与原始微调完全相同的配置
    if experiment_type == 'transformer':
        train_config = settings.TRANSFORMER_FINETUNE_CONFIG.copy()
        train_config['dataset_name'] = config['dataset_name']  # 确保数据集名称正确
    else:
        train_config = {
            'model_name': config['model_name'],
            'learning_rate': config['comparison_learning_rate'],
            'lambda_reg': config['lambda_reg'],
            'num_epochs': config['comparison_train_epochs'],
            'dataset_name': config['dataset_name']
        }
    
    # --- 4. 运行三个实验 ---
    print("\n🔥 开始三个训练实验...")
    
    experiments = [
        (full_train_loader, "完整数据集 (100%)"),
        (shapley_core_loader, f"Shapley核心集 ({config['core_set_percent']}%)"),
        (random_core_loader, f"随机核心集 ({config['core_set_percent']}%)")
    ]
    
    results = []
    
    for train_loader, exp_name in experiments:
        print(f"\n--- {exp_name} ---")
        
        # 直接传递loader，让train_utils内部处理accelerator prepare
        history = train_model_with_accelerate(
            train_loader=train_loader,
            test_loader=test_loader,
            model_config=train_config,
            num_classes=num_classes,
            experiment_name=exp_name,
            model_type=experiment_type,
            save_model=False  # 核心集实验不保存模型
        )
        
        results.append(history['accuracy'][-1])
    
    # --- 5. 可视化结果 ---
    final_acc_full, final_acc_shapley, final_acc_random = results
    
    exp_names = [
        f'完整数据集\n(100%)',
        f'Shapley核心集\n({config["core_set_percent"]}%)',
        f'随机核心集\n({config["core_set_percent"]}%)'
    ]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(exp_names, results, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    
    plt.ylabel('准确率 (%)')
    plt.title(f'Core-Set实验结果 - {experiment_type.upper()}')
    plt.ylim(0, 100)
    
    for bar, acc in zip(bars, results):
        plt.text(bar.get_x() + bar.get_width()/2.0, acc + 1, 
                f'{acc:.2f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    os.makedirs(settings.RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(settings.RESULTS_DIR, f"core_set_simple_{experiment_type}.png")
    plt.savefig(save_path)
    print(f"\n📊 结果图保存到: {save_path}")
    plt.close()
    
    print(f"\n=== 🎯 实验结果 ===")
    print(f"完整数据集: {final_acc_full:.2f}%")
    print(f"Shapley核心集: {final_acc_shapley:.2f}%")
    print(f"随机核心集: {final_acc_random:.2f}%")
    print(f"Shapley提升: {final_acc_shapley - final_acc_random:.2f}%")
    
    return {
        'full': final_acc_full,
        'shapley': final_acc_shapley, 
        'random': final_acc_random,
        'improvement': final_acc_shapley - final_acc_random
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="简化的核心数据集实验")
    parser.add_argument("--type", choices=["resnet", "transformer"], 
                       default="resnet", help="实验类型")
    args = parser.parse_args()
    
    run_core_set_experiment(args.type)