# settings.py
# 
# 核心任务配置：
# 1. 微调ResNet模型 + 计算Shapley值
# 2. 微调Transformer模型 + 计算Shapley值
# 3. 基于Shapley值的数据分析实验

# ========================================
# 全局设置
# ========================================

# --- 路径配置 ---
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"

# --- 基础参数 ---
SEED = 42
DEVICE = "cuda:0"  # 如果可用，使用GPU

# ========================================
# ResNet 工作流配置
# ========================================

# --- ResNet 统一配置 ---
RESNET_CONFIG = {
    # 数据集配置
    "dataset_name": "CIFAR10",  # 可选: "CIFAR10", "CIFAR100"
    "batch_size": 64,
    
    # 模型配置
    "model_name": "ResNet18",
    "model_type": "resnet",
    
    # 训练配置
    "num_epochs": 50,
    "learning_rate": 1e-3,
    "lambda_reg": 1e-2,  # L2正则化强度
    "label_noise_rate": 0.1,  # 标签噪声比例，用于测试错误检测
    
    # Shapley计算配置
    "num_test_samples_to_use": -1,  # -1表示使用全部测试集
}

# 自动生成checkpoint文件名
RESNET_CONFIG["checkpoint_name"] = f"{RESNET_CONFIG['model_name']}_on_{RESNET_CONFIG['dataset_name']}_lambda_{RESNET_CONFIG['lambda_reg']}_noise_{RESNET_CONFIG['label_noise_rate']}.pth"

# ========================================
# Vision Transformer (ViT) 工作流配置
# ========================================

# --- ViT 统一配置 ---
VIT_CONFIG = {
    # 数据集配置
    "dataset_name": "CIFAR10",  # 可选: "CIFAR10", "CIFAR100", "ImageNet"
    "batch_size": 32,  # ViT通常需要较小的batch size
    
    # 模型配置
    "model_name": "google/vit-base-patch16-224",  # 预训练ViT模型
    "model_type": "vit",
    "image_size": 224,  # 输入图像大小
    
    # 训练配置
    "num_epochs": 30,
    "learning_rate": 5e-4,  # ViT通常用较小的学习率
    "lambda_reg": 1e-2,
    "label_noise_rate": 0.1,
    
    # ViT特定配置
    "use_pretrained": True,  # 使用预训练权重
    "freeze_backbone": False,  # 是否冻结主干网络
    "use_bf16": True,  # 使用混合精度训练
    
    # Shapley计算配置
    "num_test_samples_to_use": -1,
}

# 自动生成checkpoint文件名
VIT_CONFIG["checkpoint_name"] = f"{VIT_CONFIG['model_name'].replace('/', '_')}_on_{VIT_CONFIG['dataset_name']}_lambda_{VIT_CONFIG['lambda_reg']}_noise_{VIT_CONFIG['label_noise_rate']}.pth"

# ========================================
# Transformer 工作流配置  
# ========================================

# --- Transformer 数据集配置 ---
TRANSFORMER_DATASET = "arc-challenge"  # 文本分类数据集，可选: "imdb", "arc-challenge", "mmlu"
TRANSFORMER_BATCH_SIZE = 32

# --- Transformer 微调配置 ---
TRANSFORMER_FINETUNE_CONFIG = {
    "model_name": "/opt/models/Qwen3-4B-Base", 
    "dataset_name": TRANSFORMER_DATASET,
    "num_epochs": 50,
    "learning_rate": 1e-4,
    "lambda_reg": 1e-2,  # L2正则化强度
    "label_noise_rate": 0.0,  # 文本任务通常不添加人工噪声
    "batch_size": TRANSFORMER_BATCH_SIZE,
    # 优化配置
    "use_4bit_quantization": False,
    "use_bf16": True,
    "gradient_accumulation_steps": 1,
    # Shapley计算优化
    "shapley_buffer_size": 8,  # 多GPU模式下的buffer大小，减少通信频率
}

# 注释：多项选择题数据集都可以直接使用TRANSFORMER_FINETUNE_CONFIG
# ARC-Challenge: 修改 dataset_name 为 "arc-challenge"
# MMLU: 修改 dataset_name 为 "mmlu"

# --- Transformer Shapley值计算配置 ---
TRANSFORMER_SHAPLEY_CONFIG = {
    "model_type": "transformer",
    "model_name": TRANSFORMER_FINETUNE_CONFIG["model_name"],
    "dataset_name": TRANSFORMER_FINETUNE_CONFIG["dataset_name"],
    "lambda_reg": TRANSFORMER_FINETUNE_CONFIG["lambda_reg"],
    "label_noise_rate": TRANSFORMER_FINETUNE_CONFIG["label_noise_rate"],
    "num_test_samples_to_use": -1,  # -1表示使用全部测试集
    "use_bf16": TRANSFORMER_FINETUNE_CONFIG["use_bf16"],
    # 自动生成checkpoint文件名 - 与微调脚本保持一致
    "checkpoint_name": f"finetuned_{TRANSFORMER_FINETUNE_CONFIG['model_name'].replace('/', '_')}_on_{TRANSFORMER_DATASET}_lambda_{TRANSFORMER_FINETUNE_CONFIG['lambda_reg']}_noise_{TRANSFORMER_FINETUNE_CONFIG.get('label_noise_rate', 0.0)}.pth",
}

# 注释：所有transformer数据集都使用TRANSFORMER_SHAPLEY_CONFIG

# ========================================
# 数据分析实验配置
# ========================================

# --- 错误检测分析配置 ---
MISLABEL_DETECTION_CONFIG = {
    # ResNet实验
    "resnet": {
        "shapley_file": f"shapley_vectors_{RESNET_CONFIG['dataset_name']}_{RESNET_CONFIG['model_name']}_noise_{RESNET_CONFIG['label_noise_rate']}.pt",
        "analysis_type": "mislabel_detection",  # 错误标签检测
    },
    # Transformer实验
    "transformer": {
        "shapley_file": f"shapley_vectors_{TRANSFORMER_DATASET}_{TRANSFORMER_FINETUNE_CONFIG['model_name'].split('/')[-1]}_noise_{TRANSFORMER_FINETUNE_CONFIG['label_noise_rate']}.pt",
        "analysis_type": "data_valuation",  # 数据估值
    }
}

# --- 核心数据集实验配置 ---
CORE_SET_EXPERIMENT_CONFIG = {
    "resnet": {
        "model_name": RESNET_CONFIG["model_name"],
        "dataset_name": RESNET_CONFIG["dataset_name"],
        "core_set_percent": 20,  # 选择20%的核心数据
        "lambda_reg": RESNET_CONFIG["lambda_reg"],
        "label_noise_rate": RESNET_CONFIG["label_noise_rate"],
        "comparison_train_epochs": RESNET_CONFIG["num_epochs"],
        "comparison_learning_rate": RESNET_CONFIG["learning_rate"],
        "checkpoint_name": RESNET_CONFIG["checkpoint_name"],
    },
    "vit": {
        "model_name": VIT_CONFIG["model_name"],
        "dataset_name": VIT_CONFIG["dataset_name"],
        "core_set_percent": 30,  # 选择30%的核心数据
        "lambda_reg": VIT_CONFIG["lambda_reg"],
        "label_noise_rate": VIT_CONFIG["label_noise_rate"],
        "comparison_train_epochs": VIT_CONFIG["num_epochs"],
        "comparison_learning_rate": VIT_CONFIG["learning_rate"],
        "checkpoint_name": VIT_CONFIG["checkpoint_name"],
    },
    "transformer": {
        "model_name": TRANSFORMER_FINETUNE_CONFIG["model_name"],
        "dataset_name": TRANSFORMER_DATASET,
        "core_set_percent": 50,  # 选择50%的核心数据
        "lambda_reg": TRANSFORMER_FINETUNE_CONFIG["lambda_reg"],
        "label_noise_rate": TRANSFORMER_FINETUNE_CONFIG["label_noise_rate"],
        "comparison_train_epochs": TRANSFORMER_FINETUNE_CONFIG["num_epochs"],
        "comparison_learning_rate": TRANSFORMER_FINETUNE_CONFIG["learning_rate"]
    }
}

# ========================================
# 便捷访问配置（向后兼容）
# ========================================

# 为了与现有脚本兼容，保留这些别名
RESNET_DATASET = RESNET_CONFIG["dataset_name"]  # 向后兼容
RESNET_BATCH_SIZE = RESNET_CONFIG["batch_size"]  # 向后兼容
BATCH_SIZE = RESNET_CONFIG["batch_size"]
DATASET = RESNET_CONFIG["dataset_name"]
RESNET_FINETUNE_CONFIG = RESNET_CONFIG  # 向后兼容
RESNET_SHAPLEY_CONFIG = RESNET_CONFIG  # 现在训练和Shapley使用同一个配置
TRAIN_FROM_SCRATCH_CONFIG = RESNET_CONFIG
SHAPLEY_CONFIG = RESNET_CONFIG

# ========================================
# 使用说明
# ========================================

"""
使用指南：

1. ResNet工作流：
   - 训练: python experiments/resnet/train.py (使用 RESNET_CONFIG)
   - Shapley计算: python experiments/resnet/run_shapley.py --accelerate(使用 RESNET_CONFIG)
   - 分析: python experiments/analysis/run_analysis.py --type resnet

2. Transformer工作流：
   - 微调: python experiments/transformer/finetune.py (使用 TRANSFORMER_FINETUNE_CONFIG)
   - Shapley计算: python experiments/transformer/run_shapley.py (使用 TRANSFORMER_FINETUNE_CONFIG)
   - 分析: python experiments/analysis/run_analysis.py --type transformer

3. 核心数据集实验：
   - python experiments/analysis/core_set_experiment.py (使用 CORE_SET_EXPERIMENT_CONFIG)

4. 统一Shapley工具：
   - python utils/shapley_utils.py --model-type resnet [--accelerate]
   - python utils/shapley_utils.py --model-type transformer [--accelerate]

配置说明：
- RESNET_CONFIG: ResNet的统一配置，包含训练和Shapley计算所需的所有参数
- TRANSFORMER_FINETUNE_CONFIG: Transformer的训练和Shapley配置
- 修改相应配置中的参数即可调整实验设置
- 使用 --accelerate 参数启用多GPU加速

重要提醒：
- 修改 TRANSFORMER_FINETUNE_CONFIG 中的 model_name 为你的实际模型路径
- 根据显存情况调整 batch_size 和 use_bf16 设置
- label_noise_rate 在ResNet实验中用于测试错误检测，在Transformer实验中通常设为0
"""