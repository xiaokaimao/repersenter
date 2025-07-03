# shapley_calculator.py

import torch
import torch.nn as nn
from tqdm import tqdm
import os

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import utils
import config.settings as settings

class ModelWrapper(nn.Module):
    """
    一个稳健的包装器，确保模型在一次前向传播中同时返回logits和features。
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
            self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
            self.classifier = model.fc # 假设最后一层名为fc

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
            features = self.feature_extractor(inputs).view(inputs.size(0), -1)
            logits = self.classifier(features)
        return logits, features

def calculate_shapley_vectors(model, train_loader, test_loader, num_classes, device, lambda_reg, num_test_samples_to_use=-1, model_type='resnet'):
    """
    计算每个训练样本的Shapley值向量。
    返回一个 (n_train, num_classes) 的综合Shapley值向量矩阵和训练标签。
    """
    # 使用我们新的包装器
    wrapper = ModelWrapper(model, model_type).to(device)
    wrapper.eval()
    
    # --- 步骤 1: 预计算模型在训练集上的属性 (真正的一次传播) ---
    print("--- Step 1/4: Extracting features and logits (in a single pass)...")
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

    # --- 步骤 2: 计算内在Shapley值矩阵 Φ (n_train, num_classes) ---
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

    # --- 步骤 3: 聚合测试集信息 ---
    print("--- Step 3/4: Aggregating similarities over the test set...")
    if num_test_samples_to_use == -1 or num_test_samples_to_use > len(test_loader.dataset):
        num_test_samples_to_use = len(test_loader.dataset)
    F_train_gpu = F_train.to(device)
    total_kernel_sim = torch.zeros(n_train, device=device)
    test_samples_processed = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Aggregating test similarities"):
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

            # 只调用一次，并且只取我们需要的 features
            _, features_batch = wrapper(inputs)
            
            kernel_batch = features_batch @ F_train_gpu.T
            total_kernel_sim += kernel_batch.sum(dim=0)
            test_samples_processed += batch_size
    avg_kernel_sim = (total_kernel_sim / test_samples_processed).cpu()

    # --- 步骤 4: 计算最终的综合Shapley值向量 ---
    print("--- Step 4/4: Calculating final aggregated Shapley vectors...")
    aggregated_shapley_vectors = Phi * avg_kernel_sim.unsqueeze(1)
    
    print("--- Shapley vectors calculation is complete ---")
    return aggregated_shapley_vectors, Y_train


if __name__ == "__main__":
    config = settings.RESNET_SHAPLEY_CONFIG
    dataset_name = settings.RESNET_DATASET
    batch_size = settings.BATCH_SIZE
    device = torch.device(settings.DEVICE if torch.cuda.is_available() else "cpu")

    print(f"Starting Shapley value calculation on device: {device}")

    utils.set_seed(settings.SEED)
    
    # 检测模型类型
    model_type = config.get('model_type', 'resnet')  # 默认为 resnet
    
    if model_type == 'transformer':
        # 加载文本数据和 Transformer 模型
        train_loader, test_loader, _, _, num_classes, original_labels, flipped_indices = utils.get_text_data_loaders(
            dataset_name=config.get('dataset_name', 'imdb'),
            tokenizer_name=config['model_name'],
            batch_size=batch_size,
            label_noise_rate=config.get('label_noise_rate', 0.0)
        )
        
        model = utils.get_transformer_model(
            model_name=config['model_name'], 
            num_classes=num_classes,
            checkpoint_path=os.path.join(settings.CHECKPOINT_DIR, config['checkpoint_name']) if 'checkpoint_name' in config else None,
            use_bf16=config.get('use_bf16', False)
        ).to(device)
    else:
        # 加载图像数据和传统模型
        train_loader, test_loader, _, _, num_classes, original_labels, flipped_indices = utils.get_data_loaders(
            dataset_name=dataset_name, 
            batch_size=batch_size,
            shuffle_train=False,
            label_noise_rate=config.get('label_noise_rate', 0.0)
        )
        
        model = utils.get_model(
            model_name=config['model_name'], 
            num_classes=num_classes, 
            from_scratch=True,
            checkpoint_path=os.path.join(settings.CHECKPOINT_DIR, config['checkpoint_name']),
            device=device
        )

    # 执行计算
    shapley_vectors, noisy_labels = calculate_shapley_vectors(
        model, train_loader, test_loader, num_classes, device, 
        config['lambda_reg'], config["num_test_samples_to_use"], model_type
    )
    
    # 将结果保存到文件
    save_dir = settings.RESULTS_DIR
    os.makedirs(save_dir, exist_ok=True)
    # 根据模型类型调整文件名
    if model_type == 'transformer':
        model_name_safe = config['model_name'].replace('/', '_').replace('-', '_')
        dataset_name_used = config.get('dataset_name', 'imdb')
    else:
        model_name_safe = config['model_name']
        dataset_name_used = dataset_name
    
    save_path = os.path.join(save_dir, f"shapley_vectors_{dataset_name_used}_{model_name_safe}_noise_{config['label_noise_rate']}.pt")
    
    torch.save({
        'shapley_vectors': shapley_vectors,
        'noisy_labels': noisy_labels,
        # 【关键修改】: 保存额外信息
        'original_labels': torch.tensor(original_labels),
        'flipped_indices': torch.tensor(flipped_indices)
    }, save_path)
    
    print(f"\nCalculation complete. Shapley vectors and labels saved to:\n{save_path}")