import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import models
import os
import random
import numpy as np
import config.settings as settings
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

@dataclass
class RewardDataCollator:
    """专门为奖励模型任务自定义的数据整理器。"""
    tokenizer: AutoTokenizer
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 从特征列表中分离 chosen 和 rejected
        chosen_features = []
        rejected_features = []
        for feature in features:
            chosen_features.append({
                "input_ids": feature["chosen_input_ids"],
                "attention_mask": feature["chosen_attention_mask"],
            })
            rejected_features.append({
                "input_ids": feature["rejected_input_ids"],
                "attention_mask": feature["rejected_attention_mask"],
            })

        # 分别对 chosen 和 rejected 进行填充
        batch_chosen = self.tokenizer.pad(
            chosen_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch_rejected = self.tokenizer.pad(
            rejected_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # 将填充后的数据合并回一个字典
        return {
            "chosen_input_ids": batch_chosen["input_ids"],
            "chosen_attention_mask": batch_chosen["attention_mask"],
            "rejected_input_ids": batch_rejected["input_ids"],
            "rejected_attention_mask": batch_rejected["attention_mask"],
        }


def set_seed(seed):
    """固定所有随机种子以确保实验的可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    # 确保cuDNN的确定性，这可能会牺牲一些性能
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_vit_data_loaders(dataset_name, batch_size, image_size=224, label_noise_rate=0.0, shuffle_train=True):
    """
    为ViT模型加载数据，使用适当的预处理
    
    Args:
        dataset_name: 数据集名称 ('CIFAR10', 'CIFAR100')
        batch_size: 批处理大小
        image_size: 输入图像大小，默认224 (ViT标准)
        label_noise_rate: 标签噪声比例
        shuffle_train: 是否打乱训练集
    """
    from transformers import ViTImageProcessor
    
    # 使用ViT的标准预处理器
    try:
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    except:
        # 如果无法下载，使用手动设置
        processor = None
    
    # ViT的标准预处理
    if processor:
        # 使用HuggingFace预处理器
        def vit_transform(image):
            # 将PIL图像转换为ViT输入格式
            inputs = processor(images=image, return_tensors="pt")
            return inputs['pixel_values'].squeeze(0)  # 移除batch维度
        
        transform_train = vit_transform
        transform_test = vit_transform
    else:
        # 手动设置ViT预处理
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        
        transform_train = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])
        transform_test = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])
    
    # 加载数据集
    if dataset_name == "CIFAR10":
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 10
    elif dataset_name == "CIFAR100":
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 100
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    # 添加标签噪声
    original_labels = list(train_dataset.targets)
    flipped_indices = []
    
    if label_noise_rate > 0:
        n_samples = len(train_dataset)
        n_flipped = int(n_samples * label_noise_rate)
        
        flipped_indices = np.random.choice(n_samples, n_flipped, replace=False)
        print(f"为{n_flipped}个样本添加标签噪声 ({label_noise_rate*100:.1f}%)")
        
        for idx in flipped_indices:
            original_label = train_dataset.targets[idx]
            # 随机选择一个不同的标签
            new_label = np.random.choice([i for i in range(num_classes) if i != original_label])
            train_dataset.targets[idx] = new_label
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, train_dataset, test_dataset, num_classes, original_labels, flipped_indices

def get_data_loaders(dataset_name, batch_size, for_finetuning=False, shuffle_train=True, label_noise_rate=0.0):
    """
    加载并返回数据加载器。
    Args:
        dataset_name (str): 'CIFAR10' 或 'CIFAR100'.
        batch_size (int): 批处理大小.
        for_finetuning (bool): 如果为True, 则使用ImageNet的预处理方法.
        shuffle_train (bool): 是否打乱训练集.
    """
    if for_finetuning:
        # 用于微调的预处理 (放大到224x224, 使用ImageNet均值/标准差)
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])
    else:
        # 用于从头训练的预处理 (使用CIFAR自身的均值/标准差)
        cifar_mean = [0.4914, 0.4822, 0.4465]
        cifar_std  = [0.2023, 0.1994, 0.2010]
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std),
        ])

    if dataset_name.upper() == "CIFAR10":
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 10
    elif dataset_name.upper() == "CIFAR100":
        train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 100
    else:
        raise ValueError("数据集名称必须是 'CIFAR10' 或 'CIFAR100'")

    original_targets = list(train_set.targets)
    flipped_indices = []
    
    if label_noise_rate > 0:
        print(f"正在为训练集引入 {label_noise_rate*100:.2f}% 的标签噪声...")
        num_samples_to_flip = int(len(train_set) * label_noise_rate)
        # 确保每次翻转的样本都一样
        np.random.seed(settings.SEED) 
        flip_indices_set = set(np.random.choice(len(train_set), num_samples_to_flip, replace=False))
        flipped_indices = sorted(list(flip_indices_set))

        train_labels = list(train_set.targets) # 使用可修改的副本
        
        for idx in flipped_indices:
            original_label = train_labels[idx]
            new_label = random.choice([i for i in range(num_classes) if i != original_label])
            train_labels[idx] = new_label
        
        train_set.targets = train_labels
        print(f"已翻转 {len(flipped_indices)} 个训练样本的标签。")

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle_train, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"数据加载完成. 数据集: {dataset_name}, 类别数: {num_classes}")
    # 【关键修改】: 返回额外的信息
    return train_loader, test_loader, train_set, test_set, num_classes, original_targets, flipped_indices



def get_text_data_loaders(dataset_name, tokenizer_name, batch_size, label_noise_rate=0.0):
    """为文本分类任务加载并准备数据。支持IMDB、ARC-Challenge等数据集"""
    print(f"Loading dataset: {dataset_name}")
    
    # 特殊处理多项选择题数据集
    if dataset_name.lower() in ['arc-challenge', 'ai2_arc', 'ai2_arc/ARC-Challenge']:
        return get_arc_challenge_loaders(tokenizer_name, batch_size, label_noise_rate)
    elif dataset_name.lower() in ['mmlu', 'cais/mmlu']:
        return get_mmlu_loaders(tokenizer_name, batch_size, label_noise_rate)
    
    raw_datasets = load_dataset(dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        # 支持更多文本字段
        if 'text' in examples:
            text_field = 'text'
        elif 'sentence' in examples:
            text_field = 'sentence'
        elif 'question' in examples:
            text_field = 'question'
        else:
            # 尝试找到文本字段
            text_fields = [k for k in examples.keys() if isinstance(examples[k], list) and isinstance(examples[k][0], str)]
            text_field = text_fields[0] if text_fields else 'text'
        
        return tokenizer(
            examples[text_field],
            truncation=True,
            padding=False,
            max_length=512  # ARC可能需要更长的上下文
        )

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    train_dataset = tokenized_datasets["train"]
    test_dataset = tokenized_datasets["test"]
    
    num_classes = raw_datasets['train'].features['label'].num_classes
    original_targets = list(train_dataset['labels'].numpy())
    flipped_indices = []
    
    # 标签翻转逻辑
    if label_noise_rate > 0:
        print(f"正在为训练集引入 {label_noise_rate*100:.2f}% 的标签噪声...")
        num_samples_to_flip = int(len(train_dataset) * label_noise_rate)
        # 确保每次翻转的样本都一样
        np.random.seed(settings.SEED) 
        flip_indices_set = set(np.random.choice(len(train_dataset), num_samples_to_flip, replace=False))
        flipped_indices = sorted(list(flip_indices_set))

        train_labels = list(train_dataset['labels'].numpy()) # 使用可修改的副本
        
        for idx in flipped_indices:
            original_label = train_labels[idx]
            new_label = random.choice([i for i in range(num_classes) if i != original_label])
            train_labels[idx] = new_label
        
        # 更新dataset中的标签
        train_dataset = train_dataset.remove_columns(['labels'])
        train_dataset = train_dataset.add_column('labels', train_labels)
        print(f"已翻转 {len(flipped_indices)} 个训练样本的标签。")
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # --- Distributed‑aware DataLoaders ---
    is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    if is_distributed:
        from torch.utils.data import DistributedSampler
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler = None
        test_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),  # Only shuffle when not using DistributedSampler
        collate_fn=data_collator
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        shuffle=False,
        collate_fn=data_collator
    )
    
    return train_loader, test_loader, train_dataset, test_dataset, num_classes, original_targets, flipped_indices


def get_arc_challenge_loaders(tokenizer_name, batch_size, label_noise_rate=0.0):
    """
    专门为ARC-Challenge数据集加载和预处理数据
    ARC-Challenge是多项选择题格式，需要特殊处理
    """
    print("Loading ARC-Challenge dataset...")
    
    try:
        # 加载ARC数据集
        raw_datasets = load_dataset("ai2_arc", "ARC-Challenge")
    except Exception as e:
        print(f"尝试加载allenai/ai2_arc...")
        raw_datasets = load_dataset("allenai/ai2_arc", "ARC-Challenge")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess_arc_function(examples):
        """
        将ARC-Challenge的多项选择题转换为分类任务
        格式：[Question] [Choice A] [Choice B] [Choice C] [Choice D]
        """
        processed_texts = []
        processed_labels = []
        
        for i in range(len(examples['question'])):
            question = examples['question'][i]
            choices = examples['choices'][i]
            answer_key = examples['answerKey'][i]
            
            # 构建输入文本：问题 + 所有选项
            choice_text = []
            for j, choice in enumerate(choices['text']):
                choice_label = choices['label'][j]
                choice_text.append(f"{choice_label}: {choice}")
            
            # 组合成完整文本
            full_text = f"Question: {question} Choices: {' '.join(choice_text)}"
            processed_texts.append(full_text)
            
            # 将答案键转换为数字标签 (A=0, B=1, C=2, D=3)
            label_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
            label = label_mapping.get(answer_key, 0)
            processed_labels.append(label)
        
        # Tokenize
        tokenized = tokenizer(
            processed_texts,
            truncation=True,
            padding=False,
            max_length=512
        )
        
        tokenized['labels'] = processed_labels
        return tokenized

    # 处理数据集
    tokenized_datasets = raw_datasets.map(preprocess_arc_function, batched=True)
    
    # 移除原始列，只保留tokenized的结果
    columns_to_remove = [col for col in tokenized_datasets["train"].column_names 
                        if col not in ['input_ids', 'attention_mask', 'labels']]
    tokenized_datasets = tokenized_datasets.remove_columns(columns_to_remove)
    tokenized_datasets.set_format("torch")

    train_dataset = tokenized_datasets["train"]
    # ARC-Challenge可能没有标准的test集，使用validation
    if "test" in tokenized_datasets:
        test_dataset = tokenized_datasets["test"]
    elif "validation" in tokenized_datasets:
        test_dataset = tokenized_datasets["validation"]
        print("使用validation集作为测试集")
    else:
        # 如果没有测试集，从训练集分割
        print("从训练集分割测试集...")
        split_dataset = train_dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = split_dataset["train"]
        test_dataset = split_dataset["test"]
    
    num_classes = 4  # ARC-Challenge是4选项多项选择题
    original_targets = list(train_dataset['labels'].numpy())
    flipped_indices = []
    
    # 标签噪声处理
    if label_noise_rate > 0:
        print(f"为ARC-Challenge训练集引入 {label_noise_rate*100:.2f}% 的标签噪声...")
        num_samples_to_flip = int(len(train_dataset) * label_noise_rate)
        
        random.seed(42)  # 确保可重现
        flip_indices_set = set(random.sample(range(len(train_dataset)), num_samples_to_flip))
        flipped_indices = sorted(list(flip_indices_set))

        train_labels = list(train_dataset['labels'].numpy())
        
        for idx in flipped_indices:
            original_label = train_labels[idx]
            # 随机选择其他选项 (0-3)
            new_label = random.choice([i for i in range(4) if i != original_label])
            train_labels[idx] = new_label
        
        # 更新dataset中的标签
        train_dataset = train_dataset.remove_columns(['labels'])
        train_dataset = train_dataset.add_column('labels', train_labels)
        print(f"已翻转 {len(flipped_indices)} 个ARC训练样本的标签。")
    
    # 创建数据加载器
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator
    )
    
    print(f"ARC-Challenge数据加载完成:")
    print(f"  训练样本: {len(train_dataset)}")
    print(f"  测试样本: {len(test_dataset)}")
    print(f"  类别数: {num_classes} (A, B, C, D)")
    
    return train_loader, test_loader, train_dataset, test_dataset, num_classes, original_targets, flipped_indices


def get_mmlu_loaders(tokenizer_name, batch_size, label_noise_rate=0.0):
    """
    为MMLU数据集加载和预处理数据
    MMLU和ARC-Challenge格式类似，都是多项选择题
    """
    print("Loading MMLU dataset...")
    
    try:
        # 尝试加载MMLU数据集
        raw_datasets = load_dataset("cais/mmlu", "all")
    except Exception as e:
        print(f"尝试加载lukaemon/mmlu...")
        try:
            raw_datasets = load_dataset("lukaemon/mmlu")
        except Exception as e2:
            print(f"尝试加载tasksource/mmlu...")
            raw_datasets = load_dataset("tasksource/mmlu")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess_mmlu_function(examples):
        """
        将MMLU的多项选择题转换为分类任务
        MMLU格式: question, choices (list), answer (int 0-3)
        """
        processed_texts = []
        processed_labels = []
        
        for i in range(len(examples['question'])):
            question = examples['question'][i]
            choices = examples['choices'][i]  # MMLU直接提供选项列表
            answer = examples['answer'][i]    # MMLU直接提供数字答案0-3
            
            # 构建输入文本：问题 + 所有选项
            choice_text = []
            for j, choice in enumerate(choices):
                choice_label = ['A', 'B', 'C', 'D'][j]
                choice_text.append(f"{choice_label}: {choice}")
            
            # 组合成完整文本
            full_text = f"Question: {question} Choices: {' '.join(choice_text)}"
            processed_texts.append(full_text)
            processed_labels.append(answer)
        
        # Tokenize
        tokenized = tokenizer(
            processed_texts,
            truncation=True,
            padding=False,
            max_length=512
        )
        
        tokenized['labels'] = processed_labels
        return tokenized

    # 处理数据集
    tokenized_datasets = raw_datasets.map(preprocess_mmlu_function, batched=True)
    
    # 移除原始列，只保留tokenized的结果
    columns_to_remove = [col for col in tokenized_datasets["auxiliary_train"].column_names 
                        if col not in ['input_ids', 'attention_mask', 'labels']]
    tokenized_datasets = tokenized_datasets.remove_columns(columns_to_remove)
    tokenized_datasets.set_format("torch")

    train_dataset = tokenized_datasets["auxiliary_train"]
    # MMLU通常有validation和test集
    if "test" in tokenized_datasets:
        test_dataset = tokenized_datasets["test"]
    elif "validation" in tokenized_datasets:
        test_dataset = tokenized_datasets["validation"]
        print("使用validation集作为测试集")
    elif "dev" in tokenized_datasets:
        test_dataset = tokenized_datasets["dev"]
        print("使用dev集作为测试集")
    else:
        # 如果没有测试集，从训练集分割
        print("从训练集分割测试集...")
        split_dataset = train_dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = split_dataset["train"]
        test_dataset = split_dataset["test"]
    
    num_classes = 4  # MMLU是4选项多项选择题
    original_targets = list(train_dataset['labels'].numpy())
    flipped_indices = []
    
    # 标签噪声处理
    if label_noise_rate > 0:
        print(f"为MMLU训练集引入 {label_noise_rate*100:.2f}% 的标签噪声...")
        num_samples_to_flip = int(len(train_dataset) * label_noise_rate)
        
        random.seed(42)  # 确保可重现
        flip_indices_set = set(random.sample(range(len(train_dataset)), num_samples_to_flip))
        flipped_indices = sorted(list(flip_indices_set))

        train_labels = list(train_dataset['labels'].numpy())
        
        for idx in flipped_indices:
            original_label = train_labels[idx]
            # 随机选择其他选项 (0-3)
            new_label = random.choice([i for i in range(4) if i != original_label])
            train_labels[idx] = new_label
        
        # 更新dataset中的标签
        train_dataset = train_dataset.remove_columns(['labels'])
        train_dataset = train_dataset.add_column('labels', train_labels)
        print(f"已翻转 {len(flipped_indices)} 个MMLU训练样本的标签。")
    
    # 创建数据加载器
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator
    )
    
    print(f"MMLU数据加载完成:")
    print(f"  训练样本: {len(train_dataset)}")
    print(f"  测试样本: {len(test_dataset)}")
    print(f"  类别数: {num_classes} (A, B, C, D)")
    print(f"  学科覆盖: 57个学科的综合知识评估")
    
    return train_loader, test_loader, train_dataset, test_dataset, num_classes, original_targets, flipped_indices


def get_vit_model(model_name, num_classes, checkpoint_path=None, use_bf16=False, freeze_backbone=False, **kwargs):
    """
    加载Vision Transformer模型
    
    Args:
        model_name: HuggingFace模型名称，如 'google/vit-base-patch16-224'
        num_classes: 分类类别数
        checkpoint_path: 微调后的权重路径
        use_bf16: 是否使用bfloat16
        freeze_backbone: 是否冻结主干网络
    """
    try:
        from transformers import ViTForImageClassification, ViTConfig
    except ImportError:
        raise ImportError("请安装transformers库: pip install transformers")
    
    print(f"Loading ViT model: {model_name} with torch_dtype={'bfloat16' if use_bf16 else 'float32'}...")
    
    # 加载预训练模型
    model = ViTForImageClassification.from_pretrained(
        model_name, 
        num_labels=num_classes,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        ignore_mismatched_sizes=True
    )
    
    # 冻结主干网络（如果需要）
    if freeze_backbone:
        print("冻结ViT主干网络...")
        for name, param in model.named_parameters():
            if not name.startswith('classifier'):
                param.requires_grad = False
    
    # 如果有checkpoint，加载微调权重
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    
    print(f"ViT模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model

def get_transformer_model(model_name, num_classes, checkpoint_path=None, use_bf16=False, **kwargs):
    """
    Loads a Hugging Face Transformer model, freezes the base, and ensures only the classification head is trainable.
    """
    print(f"Loading {model_name} with torch_dtype={'bfloat16' if use_bf16 else 'float32'}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
    )

    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
        print(f"Warning: model.config.pad_token_id is None. Set to eos_token_id: {model.config.eos_token_id}")

    # --- 2. Freeze/Unfreeze parameters by name (most robust method) ---
    print("Freezing parameters based on name...")
    for name, param in model.named_parameters():
        # Unfreeze the classification head (e.g., 'score' or 'classifier')
        if name.startswith("score") or name.startswith("classifier"):
            param.requires_grad = True
        else:
            param.requires_grad = False

    # --- 3. Enable gradient checkpointing ---
    # Note: It's often better to apply checkpointing to the base model before DDP wrapping
    if model.supports_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Enabled gradient checkpointing on the full model.")
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = False

    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        print(f"Loaded model weights from {checkpoint_path}")

    # --- 4. Verification Step ---
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Verification inside get_transformer_model: Trainable parameters = {trainable_params}")
    non_frozen_layers = [name for name, param in model.named_parameters() if param.requires_grad]
    print(f"Non-frozen layers: {non_frozen_layers}")

    return model


def get_model(model_name, num_classes, from_scratch=False, checkpoint_path=None, device=None):
    """
    构建或加载模型。
    Args:
        model_name (str): 模型名称 (e.g., 'ResNet18').
        num_classes (int): 类别数.
        from_scratch (bool): 如果为True, 则从头构建随机初始化的模型.
        checkpoint_path (str): 如果提供, 则加载此路径下的模型权重.
    """
    if from_scratch:
        # 从 models/resnet.py 构建
        model = models.__dict__[model_name](num_classes=num_classes)
        print(f"从 models 模块构建了随机初始化的 {model_name}。")
    else:
        # 从 torchvision 加载预训练模型
        if model_name.lower() == 'resnet18':
            model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
        elif model_name.lower() == 'resnet50':
            model = torchvision.models.resnet50(weights='IMAGENET1K_V2')
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
        else:
            raise ValueError(f"不支持的预训练模型: {model_name}")
        print(f"从 torchvision 加载了预训练的 {model_name} 并替换了分类头。")

    if checkpoint_path:
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f"从 '{checkpoint_path}' 加载了模型权重。")
        else:
            print(f"警告: 找不到模型文件 '{checkpoint_path}'，将使用未加载权重的模型。")

    if device:
        model.to(device)
        
    return model