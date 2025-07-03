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

# --- ResNet 数据集配置 ---
RESNET_DATASET = "CIFAR10"  # 可选: "CIFAR10", "CIFAR100"
RESNET_BATCH_SIZE = 64

# --- ResNet 微调配置 ---
RESNET_FINETUNE_CONFIG = {
    "model_name": "ResNet18",
    "num_epochs": 50,
    "learning_rate": 1e-3,
    "lambda_reg": 1e-2,  # L2正则化强度
    "label_noise_rate": 0.1,  # 标签噪声比例，用于测试错误检测
}

# --- ResNet Shapley值计算配置 ---
RESNET_SHAPLEY_CONFIG = {
    "model_type": "resnet",
    "model_name": RESNET_FINETUNE_CONFIG["model_name"],
    "dataset_name": RESNET_DATASET,
    "lambda_reg": RESNET_FINETUNE_CONFIG["lambda_reg"],
    "label_noise_rate": RESNET_FINETUNE_CONFIG["label_noise_rate"],
    "num_test_samples_to_use": -1,  # -1表示使用全部测试集
    # 自动生成checkpoint文件名
    "checkpoint_name": f"{RESNET_FINETUNE_CONFIG['model_name']}_on_{RESNET_DATASET}_lambda_{RESNET_FINETUNE_CONFIG['lambda_reg']}_noise_{RESNET_FINETUNE_CONFIG['label_noise_rate']}.pth",
}

# ========================================
# Transformer 工作流配置  
# ========================================

# --- Transformer 数据集配置 ---
TRANSFORMER_DATASET = "imdb"  # 文本分类数据集
TRANSFORMER_BATCH_SIZE = 64

# --- Transformer 微调配置 ---
TRANSFORMER_FINETUNE_CONFIG = {
    "model_name": "/opt/models/Qwen3-0.6B-Base", 
    "dataset_name": TRANSFORMER_DATASET,
    "num_epochs": 5,
    "learning_rate": 5e-4,
    "lambda_reg": 1e-2,  # L2正则化强度
    "label_noise_rate": 0.0,  # 文本任务通常不添加人工噪声
    "batch_size": TRANSFORMER_BATCH_SIZE,
    # 优化配置
    "use_4bit_quantization": False,
    "use_bf16": True,
    "gradient_accumulation_steps": 1,
}

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

# ========================================
# 数据分析实验配置
# ========================================

# --- 错误检测分析配置 ---
MISLABEL_DETECTION_CONFIG = {
    # ResNet实验
    "resnet": {
        "shapley_file": f"shapley_vectors_{RESNET_DATASET}_{RESNET_FINETUNE_CONFIG['model_name']}_noise_{RESNET_FINETUNE_CONFIG['label_noise_rate']}.pt",
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
        "model_name": RESNET_FINETUNE_CONFIG["model_name"],
        "dataset_name": RESNET_DATASET,
        "core_set_percent": 20,  # 选择20%的核心数据
        "lambda_reg": RESNET_FINETUNE_CONFIG["lambda_reg"],
        "label_noise_rate": RESNET_FINETUNE_CONFIG["label_noise_rate"],
        "comparison_train_epochs": 50,
        "comparison_learning_rate": 1e-3,
    },
    "transformer": {
        "model_name": TRANSFORMER_FINETUNE_CONFIG["model_name"],
        "dataset_name": TRANSFORMER_DATASET,
        "core_set_percent": 50,  # 选择20%的核心数据
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
DATASET = RESNET_DATASET
BATCH_SIZE = RESNET_BATCH_SIZE
TRAIN_FROM_SCRATCH_CONFIG = RESNET_FINETUNE_CONFIG
SHAPLEY_CONFIG = RESNET_SHAPLEY_CONFIG

# ========================================
# 使用说明
# ========================================

"""
使用指南：

1. ResNet工作流：
   - 微调: python train_from_scratch.py (使用 RESNET_FINETUNE_CONFIG)
   - Shapley计算: python calculate_shapley.py (使用 RESNET_SHAPLEY_CONFIG)
   - 分析: python experiment.py (错误检测分析)

2. Transformer工作流：
   - 微调: python finetune_qwen3.py (使用 TRANSFORMER_FINETUNE_CONFIG)
   - Shapley计算: python calculate_shapley_transformer.py (使用 TRANSFORMER_SHAPLEY_CONFIG)
   - 分析: python experiment.py (数据估值分析)

3. 核心数据集实验：
   - python run_core_set_experiment.py (使用 CORE_SET_EXPERIMENT_CONFIG)

重要提醒：
- 修改 TRANSFORMER_FINETUNE_CONFIG 中的 model_name 为你的实际模型路径
- 根据显存情况调整 batch_size 和 use_bf16 设置
- label_noise_rate 在ResNet实验中用于测试错误检测，在Transformer实验中通常设为0
"""