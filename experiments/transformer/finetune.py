"""
简化的Transformer微调脚本 - 使用统一的多卡训练函数
运行方式: accelerate launch experiments/transformer/finetune.py
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import utils
import config.settings as settings
from utils.train_utils import train_model_with_accelerate

def main():
    config = settings.TRANSFORMER_FINETUNE_CONFIG
    
    print(f"🚀 开始微调: {config['model_name']}")
    print(f"数据集: {config['dataset_name']}")
    
    # 加载数据
    train_loader, test_loader, _, _, num_classes, _, _ = utils.get_text_data_loaders(
        dataset_name=config['dataset_name'], 
        tokenizer_name=config['model_name'],
        batch_size=config['batch_size'],  # 这个会在train_utils中被accelerator自动调整
        label_noise_rate=config['label_noise_rate']
    )
    
    # 直接调用统一的训练函数
    history = train_model_with_accelerate(
        train_loader=train_loader,
        test_loader=test_loader,
        model_config=config,
        num_classes=num_classes,
        experiment_name=f"微调 {config['model_name']}",
        model_type="transformer",
        save_model=True  # 微调时保存最佳模型
    )
    
    print(f"🎉 微调完成！最终准确率: {history['accuracy'][-1]:.2f}%")

if __name__ == "__main__":
    main()