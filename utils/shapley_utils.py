#!/usr/bin/env python3
"""
统一的Shapley值计算工具
包含核心算法实现和便捷的使用接口
支持ResNet和Transformer模型，支持单GPU和多GPU加速
"""

import torch
import torch.nn as nn
import os
import sys
from tqdm import tqdm
from accelerate import Accelerator

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import utils
import config.settings as settings

# ================================
# 核心算法实现
# ================================

class ModelWrapper(nn.Module):
    """
    统一的模型包装器，确保模型在一次前向传播中同时返回logits和features。
    支持ResNet和Transformer模型。
    """
    def __init__(self, model, model_type='resnet'):
        super().__init__()
        self.model = model
        self.model_type = model_type
        
        if model_type == 'transformer':
            # Transformer模型的基础部分作为特征提取器
            # Hugging Face AutoModelForSequenceClassification 通常将基础模型命名为各种名称
            if hasattr(model, 'transformer'): 
                self.base_model = model.transformer
            elif hasattr(model, 'roberta'): 
                self.base_model = model.roberta
            elif hasattr(model, 'bert'): 
                self.base_model = model.bert
            elif hasattr(model, 'base_model'): 
                self.base_model = model.base_model
            else: 
                raise ValueError("无法自动确定Transformer的基础模型。")
            
            # 分类头通常是 'classifier' 或 'score'
            if hasattr(model, 'classifier'):
                self.classifier = model.classifier
            elif hasattr(model, 'score'):
                self.classifier = model.score
            else:
                raise ValueError("无法自动确定Transformer的分类头。")
                
        else: # for ResNet and other vision models
            # 自动检测分类层
            if hasattr(model, 'fc'):
                self.classifier = model.fc
                # 创建特征提取器（除了最后的分类层）
                self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
            elif hasattr(model, 'classifier'):
                self.classifier = model.classifier
                self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
            else:
                # 如果找不到标准的分类层，尝试最后一个Linear层
                for name, module in reversed(list(model.named_modules())):
                    if isinstance(module, nn.Linear):
                        self.classifier = module
                        # 创建不包含分类层的特征提取器
                        layers = []
                        for child_name, child in model.named_children():
                            if child_name != name.split('.')[0]:  # 不包含分类层的部分
                                layers.append(child)
                        self.feature_extractor = nn.Sequential(*layers)
                        break
                else:
                    raise ValueError("无法自动确定ResNet的分类层")
            
            # 添加全局平均池化以确保输出维度正确
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, inputs):
        if self.model_type == 'transformer':
            # 对于Transformer，inputs是一个字典
            base_outputs = self.base_model(**inputs)
            # 使用最后一层 hidden state 的平均池化作为特征
            features = base_outputs.last_hidden_state.mean(dim=1)
            # 将特征输入分类头得到logits
            logits = self.classifier(features)
        else:
            # 对于ResNet，inputs是张量
            x = self.feature_extractor(inputs)
            # 确保特征是2D的（batch_size, feature_dim）
            if len(x.shape) > 2:
                x = self.global_pool(x)
            features = x.view(x.size(0), -1)
            logits = self.classifier(features)
        return logits, features

def calculate_shapley_vectors_single_gpu(model, train_loader, test_loader, num_classes, device, lambda_reg, num_test_samples_to_use=-1, model_type='resnet'):
    """
    优化后的单GPU版本Shapley计算
    
    Args:
        model: 训练好的模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        num_classes: 类别数量
        device: 计算设备
        lambda_reg: 正则化参数
        num_test_samples_to_use: 使用的测试样本数量（-1表示全部）
        model_type: 模型类型 ('resnet' 或 'transformer')
    
    Returns:
        tuple: (shapley_vectors, train_labels)
    """
    # 使用ModelWrapper
    wrapper = ModelWrapper(model, model_type).to(device)
    wrapper.eval()
    
    # --- 步骤 1: 预计算模型在训练集上的属性 ---
    print("--- Step 1/4: Extracting features and logits...")
    all_features, all_logits, all_labels = [], [], []
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Extracting train data"):
            if model_type == 'transformer':
                # 处理文本数据
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels']
            else:
                # 处理图像数据
                inputs, labels = batch
                inputs = inputs.to(device)
            
            logits, features = wrapper(inputs)
            
            all_features.append(features.cpu())
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())     
    
    F_train = torch.cat(all_features, dim=0)
    U_train = torch.cat(all_logits, dim=0)
    Y_train = torch.cat(all_labels, dim=0)
    n_train = F_train.shape[0]
    print(f"Training set size: {n_train}, Feature dimension: {F_train.shape[1]}")

    # --- 步骤 2: 计算内在Shapley值矩阵 Φ ---
    print("--- Step 2/4: Calculating Intrinsic Shapley Value Matrix (Φ)...")
    Gamma = (torch.nn.functional.one_hot(Y_train, num_classes=num_classes) - torch.softmax(U_train, dim=-1)) / lambda_reg
    mu_list, J_mu_list = [None] * num_classes, [None] * num_classes
    psi = lambda x: torch.softmax(x, dim=-1)
    for c in tqdm(range(num_classes), desc="Calculating Jacobians per class"):
        mu_c = U_train[Y_train == c].mean(dim=0)
        J_mu_c = torch.func.jacrev(psi)(mu_c)
        mu_list[c], J_mu_list[c] = mu_c, J_mu_c
    Phi = torch.zeros(n_train, num_classes)
    for c in tqdm(range(num_classes), desc="Calculating Φ matrix"):
        grad_c = J_mu_list[c][c, :]
        phi_for_class_c = Gamma @ grad_c
        Phi[:, c] = phi_for_class_c
    print("Intrinsic Shapley Value Matrix (Φ) calculated.")

    # --- 步骤 3: 聚合测试集信息 (优化版本) ---
    print("--- Step 3/4: Aggregating similarities over the test set (optimized)...")
    if num_test_samples_to_use == -1 or num_test_samples_to_use > len(test_loader.dataset):
        num_test_samples_to_use = len(test_loader.dataset)
    F_train_gpu = F_train.to(device)
    accumulated_kernel_sim = torch.zeros(n_train, device=device)
    test_samples_processed = 0
    
    # 优化：批量处理多个batch减少矩阵乘法次数
    batch_buffer = []
    BUFFER_SIZE = 16  # 单GPU可以用更大的buffer
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Aggregating test similarities")):
            if test_samples_processed >= num_test_samples_to_use: break
            
            if model_type == 'transformer':
                # 处理文本数据
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                batch_size = inputs['input_ids'].size(0)
            else:
                # 处理图像数据
                inputs, _ = batch
                inputs = inputs.to(device)
                batch_size = inputs.size(0)

            # 计算当前batch的特征
            _, features_batch = wrapper(inputs)
            batch_buffer.append(features_batch)
            test_samples_processed += batch_size
            
            # 当buffer满了或者是最后一个batch时，进行处理
            if len(batch_buffer) >= BUFFER_SIZE or batch_idx == len(test_loader) - 1 or test_samples_processed >= num_test_samples_to_use:
                # 合并buffer中的特征
                if len(batch_buffer) > 1:
                    combined_features = torch.cat(batch_buffer, dim=0)
                else:
                    combined_features = batch_buffer[0]
                
                # 分块计算以避免内存溢出
                chunk_size = min(2000, combined_features.size(0))  # 单GPU可以处理更大的chunk
                for i in range(0, combined_features.size(0), chunk_size):
                    chunk_features = combined_features[i:i+chunk_size]
                    kernel_chunk = torch.mm(chunk_features, F_train_gpu.T)
                    accumulated_kernel_sim += kernel_chunk.sum(dim=0)
                
                # 清空buffer
                batch_buffer.clear()
    
    avg_kernel_sim = (accumulated_kernel_sim / (test_samples_processed + 1e-8)).cpu()

    # --- 步骤 4: 计算最终的综合Shapley值向量 ---
    print("--- Step 4/4: Calculating final aggregated Shapley vectors...")
    aggregated_shapley_vectors = Phi * avg_kernel_sim.unsqueeze(1)
    
    print("--- Single GPU Shapley vectors calculation is complete ---")
    return aggregated_shapley_vectors, Y_train

def calculate_shapley_vectors_multi_gpu(model, train_loader, test_loader, num_classes, lambda_reg, num_test_samples_to_use=-1, model_type='resnet'):
    """
    使用Accelerate框架的多GPU加速Shapley值计算函数
    
    Args:
        model: 训练好的模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        num_classes: 类别数量
        lambda_reg: 正则化参数
        num_test_samples_to_use: 使用的测试样本数量（-1表示全部）
        model_type: 模型类型 ('resnet' 或 'transformer')
    
    Returns:
        tuple: (shapley_vectors, train_labels) 或 (None, None) 如果不是主进程
    """
    # 初始化Accelerator
    accelerator = Accelerator()
    device = accelerator.device
    
    if accelerator.is_main_process:
        print("🚀 使用Accelerate加速Shapley值计算")
        print(f"设备数量: {accelerator.num_processes}")
        print(f"当前设备: {device}")
    
    # 使用ModelWrapper并prepare
    wrapper = ModelWrapper(model, model_type)
    wrapper.eval()
    
    # Prepare模型和数据加载器
    wrapper, train_loader, test_loader = accelerator.prepare(wrapper, train_loader, test_loader)
    
    # --- 步骤 1: 并行提取特征和logits ---
    if accelerator.is_main_process:
        print("--- Step 1/4: Extracting features and logits (accelerated)...")
    
    all_features, all_logits, all_labels = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Extracting train data", disable=not accelerator.is_main_process):
            if model_type == 'transformer':
                # 处理文本数据
                inputs = {k: v for k, v in batch.items() if k != 'labels'}
                labels = batch['labels']
            else:
                # 处理图像数据
                inputs, labels = batch
            
            # 使用accelerator自动处理设备分配
            logits, features = wrapper(inputs)
            
            # 收集所有GPU的结果
            features = accelerator.gather(features)
            logits = accelerator.gather(logits)
            labels = accelerator.gather(labels)
            
            if accelerator.is_main_process:
                all_features.append(features.cpu())
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
    
    # 确保只在主进程中进行后续计算
    if not accelerator.is_main_process:
        return None, None
        
    F_train = torch.cat(all_features, dim=0)
    U_train = torch.cat(all_logits, dim=0)
    Y_train = torch.cat(all_labels, dim=0)
    n_train = F_train.shape[0]
    print(f"Training set size: {n_train}, Feature dimension: {F_train.shape[1]}")

    # --- 步骤 2: 计算内在Shapley值矩阵 Φ ---
    print("--- Step 2/4: Calculating Intrinsic Shapley Value Matrix (Φ)...")
    # 将计算移到GPU上加速
    Gamma = (torch.nn.functional.one_hot(Y_train, num_classes=num_classes).to(device) - 
             torch.softmax(U_train.to(device), dim=-1)) / lambda_reg
    
    mu_list, J_mu_list = [None] * num_classes, [None] * num_classes
    psi = lambda x: torch.softmax(x, dim=-1)
    
    # 并行计算每个类别的Jacobian
    U_train_gpu = U_train.to(device)
    Y_train_gpu = Y_train.to(device)
    
    for c in tqdm(range(num_classes), desc="Calculating Jacobians per class"):
        mu_c = U_train_gpu[Y_train_gpu == c].mean(dim=0)
        J_mu_c = torch.func.jacrev(psi)(mu_c)
        mu_list[c], J_mu_list[c] = mu_c, J_mu_c
    
    # 使用GPU并行计算Phi矩阵
    Phi = torch.zeros(n_train, num_classes, device=device)
    for c in tqdm(range(num_classes), desc="Calculating Φ matrix"):
        grad_c = J_mu_list[c][c, :]
        phi_for_class_c = Gamma @ grad_c
        Phi[:, c] = phi_for_class_c
    print("Intrinsic Shapley Value Matrix (Φ) calculated.")

    # --- 步骤 3: 并行聚合测试集信息 (优化版本) ---
    print("--- Step 3/4: Aggregating similarities over the test set (accelerated & optimized)...")
    if num_test_samples_to_use == -1 or num_test_samples_to_use > len(test_loader.dataset):
        num_test_samples_to_use = len(test_loader.dataset)
    
    F_train_gpu = F_train.to(device)
    # 使用更大的批次来减少gather操作次数
    accumulated_kernel_sim = torch.zeros(n_train, device=device)
    test_samples_processed = 0
    
    # 优化：批量处理多个batch减少通信开销
    batch_buffer = []
    BUFFER_SIZE = 8  # 累积8个batch后再做一次gather
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Aggregating test similarities", disable=not accelerator.is_main_process)):
            if test_samples_processed >= num_test_samples_to_use: 
                break
            
            if model_type == 'transformer':
                inputs = {k: v for k, v in batch.items() if k != 'labels'}
                current_batch_size = inputs['input_ids'].size(0)
            else:
                inputs, _ = batch
                current_batch_size = inputs.size(0)

            # 计算当前batch的特征
            _, features_batch = wrapper(inputs)
            batch_buffer.append(features_batch)
            test_samples_processed += current_batch_size
            
            # 当buffer满了或者是最后一个batch时，进行处理
            if len(batch_buffer) >= BUFFER_SIZE or batch_idx == len(test_loader) - 1 or test_samples_processed >= num_test_samples_to_use:
                # 合并buffer中的特征
                if len(batch_buffer) > 1:
                    combined_features = torch.cat(batch_buffer, dim=0)
                else:
                    combined_features = batch_buffer[0]
                
                # 并行gather所有GPU的结果
                combined_features = accelerator.gather(combined_features)
                
                if accelerator.is_main_process:
                    # 使用更高效的矩阵乘法
                    # 分块计算以避免内存溢出
                    chunk_size = min(1000, combined_features.size(0))
                    for i in range(0, combined_features.size(0), chunk_size):
                        chunk_features = combined_features[i:i+chunk_size]
                        kernel_chunk = torch.mm(chunk_features, F_train_gpu.T)
                        accumulated_kernel_sim += kernel_chunk.sum(dim=0)
                
                # 清空buffer
                batch_buffer.clear()
    
    if accelerator.is_main_process:
        # 计算所有GPU的总样本数
        total_samples_processed = test_samples_processed * accelerator.num_processes
        # 但不要超过实际要处理的样本数
        total_samples_processed = min(total_samples_processed, num_test_samples_to_use)
        avg_kernel_sim = (accumulated_kernel_sim / (total_samples_processed + 1e-8))
    else:
        avg_kernel_sim = None

    # --- 步骤 4: 计算最终Shapley值 ---
    if accelerator.is_main_process:
        print("--- Step 4/4: Calculating final aggregated Shapley vectors...")
        aggregated_shapley_vectors = Phi * avg_kernel_sim.unsqueeze(1)
        
        # 移回CPU以节省GPU内存
        aggregated_shapley_vectors = aggregated_shapley_vectors.cpu()
        Y_train = Y_train.cpu()
        
        print("--- Accelerated Shapley vectors calculation is complete ---")
        print(f"⚡ 加速计算完成，使用了 {accelerator.num_processes} 个GPU")
        
        return aggregated_shapley_vectors, Y_train
    else:
        return None, None

# ================================
# 便捷使用接口
# ================================

def load_data_and_model(model_type='resnet', config=None):
    """
    统一的数据和模型加载函数
    
    Args:
        model_type: 'resnet' 或 'transformer'
        config: 配置字典，如果为None则自动选择
    
    Returns:
        tuple: (train_loader, test_loader, model, num_classes, original_labels, flipped_indices)
    """
    if config is None:
        if model_type == 'transformer':
            config = settings.TRANSFORMER_SHAPLEY_CONFIG
        else:
            config = settings.RESNET_SHAPLEY_CONFIG
    
    print(f"=== Shapley值计算：{model_type.upper()}模型 ===")
    print(f"模型: {config['model_name']}")
    if model_type == 'transformer':
        print(f"数据集: {config['dataset_name']}")
    else:
        print(f"数据集: {settings.RESNET_DATASET}")
    print()
    
    # 设置随机种子
    utils.set_seed(settings.SEED)
    
    # 加载数据
    print("📊 加载数据...")
    if model_type == 'transformer':
        train_loader, test_loader, _, _, num_classes, original_labels, flipped_indices = utils.get_text_data_loaders(
            dataset_name=config['dataset_name'],
            tokenizer_name=config['model_name'],
            batch_size=settings.BATCH_SIZE,
            label_noise_rate=config.get('label_noise_rate', 0.0)
        )
    else:
        train_loader, test_loader, _, _, num_classes, original_labels, flipped_indices = utils.get_data_loaders(
            dataset_name=settings.RESNET_DATASET,
            batch_size=settings.BATCH_SIZE,
            shuffle_train=False,
            label_noise_rate=config.get('label_noise_rate', 0.0)
        )
    
    # 加载模型
    print("🤖 加载模型...")
    if model_type == 'transformer':
        checkpoint_path = None
        if 'checkpoint_name' in config:
            checkpoint_path = os.path.join(settings.CHECKPOINT_DIR, config['checkpoint_name'])
            if not os.path.exists(checkpoint_path):
                print(f"⚠️ 警告: 找不到checkpoint文件 {checkpoint_path}")
                print("将使用未微调的模型")
                checkpoint_path = None
            else:
                print(f"✅ 使用checkpoint: {checkpoint_path}")
        
        model = utils.get_transformer_model(
            model_name=config['model_name'],
            num_classes=num_classes,
            checkpoint_path=checkpoint_path,
            use_bf16=config.get('use_bf16', False)
        )
    else:
        model = utils.get_model(
            model_name=config['model_name'],
            num_classes=num_classes,
            from_scratch=True,
            checkpoint_path=os.path.join(settings.CHECKPOINT_DIR, config['checkpoint_name'])
        )
    
    return train_loader, test_loader, model, num_classes, original_labels, flipped_indices, config

def calculate_shapley_unified(model_type='resnet', use_accelerate=False, config=None):
    """
    统一的Shapley值计算接口
    
    Args:
        model_type: 'resnet' 或 'transformer'
        use_accelerate: 是否使用多GPU加速
        config: 配置字典，如果为None则自动选择
    
    Returns:
        tuple: (shapley_vectors, train_labels, save_path)
    """
    # 加载数据和模型
    train_loader, test_loader, model, num_classes, original_labels, flipped_indices, config = load_data_and_model(
        model_type, config
    )
    
    # 选择计算函数和设备配置
    if use_accelerate:
        print("🚀 使用多GPU加速模式")
        
        # 计算Shapley值
        shapley_vectors, train_labels = calculate_shapley_vectors_multi_gpu(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            num_classes=num_classes,
            lambda_reg=config['lambda_reg'],
            num_test_samples_to_use=config['num_test_samples_to_use'],
            model_type=model_type
        )
    else:
        print("💻 使用单GPU模式")
        device = torch.device(settings.DEVICE if torch.cuda.is_available() else "cpu")
        print(f"设备: {device}")
        model = model.to(device)
        
        # 计算Shapley值
        shapley_vectors, train_labels = calculate_shapley_vectors_single_gpu(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            num_classes=num_classes,
            device=device,
            lambda_reg=config['lambda_reg'],
            num_test_samples_to_use=config['num_test_samples_to_use'],
            model_type=model_type
        )
    
    # 如果是多GPU模式且不是主进程，返回None
    if use_accelerate and shapley_vectors is None:
        return None, None, None
    
    # 保存结果
    save_path = save_shapley_results(
        shapley_vectors, train_labels, original_labels, flipped_indices, 
        config, model_type
    )
    
    # 打印统计信息
    print_shapley_statistics(shapley_vectors, train_labels, flipped_indices, save_path)
    
    return shapley_vectors, train_labels, save_path

def save_shapley_results(shapley_vectors, train_labels, original_labels, flipped_indices, config, model_type):
    """保存Shapley计算结果"""
    save_dir = settings.RESULTS_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成文件名
    if model_type == 'transformer':
        model_name_safe = config['model_name'].replace('/', '_').replace('-', '_')
        dataset_name = config['dataset_name']
    else:
        model_name_safe = config['model_name']
        dataset_name = settings.RESNET_DATASET
    
    save_path = os.path.join(
        save_dir,
        f"shapley_vectors_{dataset_name}_{model_name_safe}_noise_{config.get('label_noise_rate', 0.0)}.pt"
    )
    
    # 保存数据
    torch.save({
        'shapley_vectors': shapley_vectors,
        'noisy_labels': train_labels,  # 保持与原始代码的一致性
        'train_labels': train_labels,  # 新增的键名
        'original_labels': torch.tensor(original_labels),
        'flipped_indices': torch.tensor(flipped_indices),
        'config': config
    }, save_path)
    
    return save_path

def print_shapley_statistics(shapley_vectors, train_labels, flipped_indices, save_path):
    """打印Shapley值统计信息"""
    print(f"\n🎉 计算完成！")
    print(f"📁 结果已保存到: {save_path}")
    print(f"📊 Shapley向量形状: {shapley_vectors.shape}")
    print(f"📝 训练样本数量: {len(train_labels)}")
    
    # 基本统计
    print("\n=== 📈 统计分析 ===")
    shapley_norms = torch.linalg.norm(shapley_vectors, dim=1)
    print(f"平均值: {shapley_norms.mean():.6f}")
    print(f"标准差: {shapley_norms.std():.6f}")
    print(f"最大值: {shapley_norms.max():.6f}")
    print(f"最小值: {shapley_norms.min():.6f}")
    
    # 标签噪声统计
    if len(flipped_indices) > 0:
        print(f"\n=== 🏷️ 标签噪声统计 ===")
        print(f"翻转样本数量: {len(flipped_indices)}")
        print(f"翻转比例: {len(flipped_indices)/len(train_labels)*100:.2f}%")

def run_shapley_calculation(model_type='resnet', use_accelerate=False):
    """
    便捷的运行函数
    
    Args:
        model_type: 'resnet' 或 'transformer'
        use_accelerate: 是否使用多GPU加速
    """
    try:
        shapley_vectors, train_labels, save_path = calculate_shapley_unified(
            model_type=model_type,
            use_accelerate=use_accelerate
        )
        
        if shapley_vectors is not None:  # 确保不是多GPU的非主进程
            print(f"\n✅ {model_type.upper()} Shapley值计算成功完成！")
            if use_accelerate:
                print("⚡ 多GPU加速显著提升了计算速度")
        
        return shapley_vectors, train_labels, save_path
        
    except Exception as e:
        print(f"❌ 计算过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="统一的Shapley值计算工具")
    parser.add_argument("--model-type", choices=["resnet", "transformer"], 
                       default="resnet", help="模型类型")
    parser.add_argument("--accelerate", action="store_true", 
                       help="使用多GPU加速")
    
    args = parser.parse_args()
    
    if args.accelerate:
        print("🚀 启用多GPU加速模式")
        print("请确保使用 'accelerate launch utils/shapley_utils.py --accelerate' 运行")
    
    run_shapley_calculation(args.model_type, args.accelerate)