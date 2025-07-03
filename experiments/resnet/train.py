"""
简化的ResNet训练脚本 - 使用统一的多卡训练函数
运行方式: accelerate launch experiments/resnet/train.py
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import utils
import config.settings as settings
from utils.train_utils import train_model_with_accelerate

def main():
    config = settings.RESNET_FINETUNE_CONFIG
    
    print(f"🚀 开始训练: {config['model_name']}")
    print(f"数据集: {settings.RESNET_DATASET}")
    
    # 加载数据
    train_loader, test_loader, _, _, num_classes, _, _ = utils.get_data_loaders(
        dataset_name=settings.RESNET_DATASET, 
        batch_size=settings.RESNET_BATCH_SIZE,
        shuffle_train=True,
        label_noise_rate=config['label_noise_rate']
    )
    
    # 直接调用统一的训练函数
    history = train_model_with_accelerate(
        train_loader=train_loader,
        test_loader=test_loader,
        model_config=config,
        num_classes=num_classes,
        experiment_name=f"训练 {config['model_name']}",
        model_type="resnet",
        save_model=True  # 训练时保存最佳模型
    )
    
    print(f"🎉 训练完成！最终准确率: {history['accuracy'][-1]:.2f}%")

if __name__ == "__main__":
    main()