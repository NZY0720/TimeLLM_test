import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer
import transformers
import math
import os

transformers.logging.set_verbosity_error()

# 定义 ReprogrammingLayer 类
class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, d_llm, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        self.d_ff = d_ff  # 保存 d_ff

        self.query_projection = nn.Linear(d_model, d_ff)
        self.key_projection = nn.Linear(d_llm, d_ff)
        self.value_projection = nn.Linear(d_llm, d_ff)
        self.out_projection = nn.Linear(d_ff, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape

        # 投影到键、查询、值空间
        Q = self.query_projection(target_embedding)  # (B, L, d_ff)
        K = self.key_projection(source_embedding)    # (S, d_ff)
        V = self.value_projection(value_embedding)   # (S, d_ff)

        # 扩展维度以匹配注意力机制的要求
        Q = Q.unsqueeze(2)  # (B, L, 1, d_ff)
        K = K.unsqueeze(0).unsqueeze(0)  # (1, 1, S, d_ff)
        V = V.unsqueeze(0).unsqueeze(0)  # (1, 1, S, d_ff)

        # 计算注意力得分
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_ff)
        A = torch.softmax(scores, dim=-1)
        A = self.dropout(A)

        # 计算重编程后的嵌入
        reprogramming_embedding = torch.matmul(A, V)  # (B, L, 1, d_ff)
        reprogramming_embedding = reprogramming_embedding.squeeze(2)  # (B, L, d_ff)

        # 输出投影
        out = self.out_projection(reprogramming_embedding)  # (B, L, d_llm)

        return out

# 定义 TimeLLM 模型
class TimeLLM(nn.Module):
    def __init__(self, seq_len, pred_len, num_features, llm_model='GPT2', llm_dim=768, patch_len=16, stride=8, dropout=0.3, llm_layers=4):
        super(TimeLLM, self).__init__()

        # 时间序列参数
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_features = num_features
        self.patch_len = patch_len
        self.stride = stride
        self.llm_dim = llm_dim

        # 选择预训练模型
        if llm_model == 'GPT2':
            # 使用本地的 GPT-2 模型和分词器
            model_path = '/workspace/Time-LLM/models/GPT2'
            self.llm_config = GPT2Config.from_pretrained(model_path, n_layer=llm_layers)
            self.llm_model = GPT2Model.from_pretrained(model_path, config=self.llm_config)
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        else:
            raise ValueError("Unsupported LLM model. Currently only 'GPT2' is supported in this code.")

        # 确保设置了 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.pad_token = '[PAD]'
            self.llm_model.resize_token_embeddings(len(self.tokenizer))

        print(f"Pad token: {self.tokenizer.pad_token}, ID: {self.tokenizer.pad_token_id}")

        # 冻结 LLM 参数
        for param in self.llm_model.parameters():
            param.requires_grad = False

        # 时间序列的特征提取
        self.patch_embedding = nn.Conv1d(
            in_channels=num_features,
            out_channels=llm_dim,
            kernel_size=patch_len,
            stride=stride
        )
        self.mapping_layer = nn.Linear(self.llm_model.config.hidden_size, llm_dim)

        # ReprogrammingLayer
        self.reprogramming_layer = ReprogrammingLayer(
            d_model=llm_dim,
            n_heads=8,
            d_ff=llm_dim,
            d_llm=llm_dim
        )

        # 输出投影
        self.output_layer = nn.Linear(llm_dim, pred_len)
        self.dropout = nn.Dropout(dropout)

        # 数据集描述（可根据需要修改）
        self.description = 'This is a time series forecasting task.'

    def forward(self, x):
        """
        时间序列预测的前向传播。
        Args:
            x: 形状为 (batch_size, seq_len, num_features) 的输入时间序列张量。
        Returns:
            形状为 (batch_size, pred_len) 的预测输出。
        """
        batch_size, seq_len, num_features = x.shape

        # 将 x 移动到模型所在的设备上
        device = next(self.parameters()).device
        x = x.to(device)

        # 计算统计特征
        min_values = x.min(dim=1)[0]
        max_values = x.max(dim=1)[0]
        median_values = x.median(dim=1)[0]
        trends_tensor = x[:, -1, :] - x[:, 0, :]
        trends = ['upward' if t.mean().item() > 0 else 'downward' for t in trends_tensor]

        # 计算滞后项（简单示例）
        lags = []
        for i in range(batch_size):
            lags_i = []
            for j in range(num_features):
                fft_vals = torch.fft.rfft(x[i, :, j])
                topk = torch.topk(torch.abs(fft_vals), k=5, dim=-1)[1]
                lags_i.append(topk.cpu().tolist())
            lags.append(lags_i)

        # 动态生成提示
        prompts = []
        for i in range(batch_size):
            prompt = (
                f"Dataset description: {self.description} "
                f"Task description: forecast the next {self.pred_len} steps given the previous {self.seq_len} steps information. "
                f"Input statistics: min value {min_values[i].cpu().tolist()}, max value {max_values[i].cpu().tolist()}, "
                f"median value {median_values[i].cpu().tolist()}, the trend of input is {trends[i]}, "
                f"top 5 lags are: {lags[i]}"
            )
            prompts.append(prompt)

        # 对时间序列数据进行补丁嵌入
        x_emb = x.permute(0, 2, 1)  # (batch_size, num_features, seq_len)
        x_emb = self.patch_embedding(x_emb)  # (batch_size, llm_dim, num_patches)
        x_emb = x_emb.permute(0, 2, 1)  # (batch_size, num_patches, llm_dim)
        x_emb = self.dropout(x_emb)  # 添加 Dropout 层

        # 将提示转换为嵌入
        tokenized_prompt = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
        prompt_embeddings = self.llm_model.get_input_embeddings()(tokenized_prompt)  # (batch_size, prompt_len, embed_dim)

        # 获取预训练模型的词嵌入（源嵌入）
        source_embeddings = self.llm_model.get_input_embeddings().weight  # (vocab_size, embed_dim)
        source_embeddings = source_embeddings.to(device)  # 确保在同一设备上

        # 使用 ReprogrammingLayer 对齐嵌入
        x_reprogrammed = self.reprogramming_layer(x_emb, source_embeddings, source_embeddings)

        # 合并提示嵌入和重编程后的时间序列嵌入
        llm_input = torch.cat([prompt_embeddings, x_reprogrammed], dim=1)

        # 通过预训练模型进行前向传播
        llm_output = self.llm_model(inputs_embeds=llm_input).last_hidden_state  # (batch_size, seq_len + prompt_len, llm_dim)

        # 将 LLM 输出映射到目标维度
        llm_output = self.mapping_layer(llm_output[:, -x_reprogrammed.shape[1]:])  # 仅取与序列相关的部分
        llm_output = self.dropout(llm_output)  # 添加 Dropout 层

        # 输出投影
        output = self.output_layer(llm_output.mean(dim=1))  # 在补丁上聚合

        return output

# 修改后的数据集类，返回 idx
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        """
        Args:
            data: 形状为 (num_samples, num_features) 的 numpy 数组
            seq_len: 输入序列的长度
            pred_len: 预测序列的长度
        """
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]  # 输入序列
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len]  # 目标序列

        # 只取第一个特征（或根据需要调整）
        y = y[:, 0]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), idx

# 主函数
def main():
    # 检查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据
    file_path = '/workspace/Time-LLM/power_1.csv'
    df = pd.read_csv(file_path, header=None)

    # 数据预处理
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df.values)  # 将数据归一化到 [0, 1]

    # 定义序列长度和预测长度
    seq_len = 48
    pred_len = 16

    # 创建数据集
    dataset = TimeSeriesDataset(data, seq_len, pred_len)

    # 划分训练集、验证集和测试集
    test_size = int(len(dataset) * 0.2)
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - test_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

    # 创建数据加载器
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型，使用本地的 GPT-2 模型
    num_features = data.shape[1]  # 获取特征维度
    model = TimeLLM(seq_len=seq_len, pred_len=pred_len, num_features=num_features, llm_model='GPT2', llm_dim=768, llm_layers=4, dropout=0.3)
    model.to(device)  # 将模型移动到设备

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)  # 降低学习率

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    # 早停机制
    best_val_loss = float('inf')
    patience = 10  # 增加 patience
    trigger_times = 0

    # 训练模型
    num_epochs = 100  # 增加训练轮次
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)  # 将数据移动到设备
            optimizer.zero_grad()
            preds = model(x)  # 前向传播
            loss = criterion(preds, y)  # 计算损失
            loss.backward()  # 反向传播
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()  # 更新权重
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # 验证模型
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x)
                loss = criterion(preds, y)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        # 学习率调度器
        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            # 保存模型
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping!")
                break

    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))

    # 评估模型并收集预测结果
    model.eval()
    test_loss = 0.0

    # 初始化用于存储完整预测和实际值的数组
    total_length = len(test_dataset) + seq_len + pred_len - 1
    predicted_values = np.zeros(total_length)
    ground_truth_values = np.zeros(total_length)
    counts = np.zeros(total_length)

    with torch.no_grad():
        for x, y, idxs in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss = criterion(preds, y)
            test_loss += loss.item()

            preds = preds.cpu().numpy()
            y = y.cpu().numpy()
            idxs = idxs.numpy()

            for i in range(len(idxs)):
                idx = idxs[i]
                pred = preds[i]  # 形状为 (pred_len,)
                gt = y[i]        # 形状为 (pred_len,)
                for j in range(pred_len):
                    position = idx + seq_len + j
                    if position < total_length:
                        predicted_values[position] += pred[j]
                        ground_truth_values[position] += gt[j]
                        counts[position] += 1

    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")

    # 计算平均预测值和实际值
    valid_positions = counts > 0
    predicted_values[valid_positions] /= counts[valid_positions]
    ground_truth_values[valid_positions] /= counts[valid_positions]

    # 可视化结果
    plt.figure(figsize=(15, 5))
    positions = np.arange(total_length)
    plt.plot(positions[valid_positions], ground_truth_values[valid_positions], label="Ground Truth")
    plt.plot(positions[valid_positions], predicted_values[valid_positions], label="Predictions", linestyle="--")
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.title("Predictions vs Ground Truth on Test Set")
    plt.savefig('prediction_plot_full_series.png')
    plt.show()

    # 保存预测值和实际值为 CSV 文件
    output_data = pd.DataFrame({
        'Position': positions[valid_positions],
        'Ground Truth': ground_truth_values[valid_positions],
        'Predictions': predicted_values[valid_positions]
    })
    output_data.to_csv('prediction_results.csv', index=False)
    print("预测结果已保存到 prediction_results.csv")

# 运行主函数
if __name__ == "__main__":
    main()
