# ============================================================
# 风机故障诊断
# 特征对齐 + 掩码填充 + LNN（时间维）+ DepCA + QR-Ortho + SVM 对比
# 去掉 Set Transformer，只做数据对齐与缺失填充
# ============================================================

import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler as SklearnScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns

np.random.seed(0)
torch.manual_seed(0)


# ============================================================
# 1. 标准化器（带掩码）
# ============================================================
class MaskedStandardScaler:
    """
    带掩码的标准化器：
    - 只统计 mask==1 的元素
    - transform 时对缺失列输出 0，并保持 mask 不变
    """
    def __init__(self, min_std=1e-6, eps=1e-8):
        self.mean = None
        self.std = None
        self.eps = eps
        self.min_std = min_std

    def fit(self, values, mask):
        m = mask.astype(np.float64)
        count = m.sum(axis=0).clip(min=1)
        mu = (values * m).sum(axis=0) / count
        var = (((values - mu) ** 2) * m).sum(axis=0) / count
        sigma = np.sqrt(var)
        sigma = np.maximum(sigma, self.min_std)
        no_data = mask.sum(axis=0) == 0
        mu[no_data] = 0.0
        sigma[no_data] = 1.0
        self.mean = mu.astype(np.float32)
        self.std = sigma.astype(np.float32)
        return self

    def transform(self, values, mask):
        normed = (values - self.mean) / (self.std + self.eps)
        return normed * mask


# ============================================================
# 2. 数据预处理（对齐 + 掩码）
# ============================================================
class DataPreprocessor:
    """
    以所有文件特征列的并集为统一列空间，各文件对齐到该空间。
    缺失列填 0，mask 置 0；有效列 mask 置 1。
    """
    def __init__(self, debug=True):
        self.feature_cols = None
        self.col_means = {}
        self.debug = debug

    def fit(self, file_label_pairs):
        print("=" * 60)
        print("开始数据预处理：第一列作为时间戳（忽略其值）")
        print("=" * 60)

        all_feature_cols = set()
        file_info = []

        for file_path, label in file_label_pairs:
            df = pd.read_csv(file_path, nrows=5, encoding='utf-8-sig')
            df.columns = [c.lstrip('\ufeff') if isinstance(c, str) else c
                          for c in df.columns]
            feature_cols = list(df.columns[1:])
            all_feature_cols |= set(feature_cols)
            file_info.append((os.path.basename(file_path),
                              len(feature_cols), df.columns[0]))

        self.feature_cols = sorted(all_feature_cols)

        print(f"\n各文件特征列数（不含第一列）：")
        for name, n_cols, t_col in file_info:
            print(f"  {name}: {n_cols} 列, 第一列 = {t_col}")
        print(f"\n统一后特征列数（并集）: {len(self.feature_cols)}")

        # 计算各列均值，用于填充缺失列
        print("\n计算各列均值（用于填充缺失列）...")
        col_values = {c: [] for c in self.feature_cols}
        for file_path, label in file_label_pairs:
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            df.columns = [c.lstrip('\ufeff') if isinstance(c, str) else c
                          for c in df.columns]
            df_features = df.iloc[:, 1:]
            for c in self.feature_cols:
                if c in df_features.columns:
                    vals = pd.to_numeric(df_features[c], errors='coerce').values
                    col_values[c].append(vals)

        for c in self.feature_cols:
            if len(col_values[c]) > 0:
                merged = np.concatenate(col_values[c])
                merged = merged[np.isfinite(merged)]
                self.col_means[c] = float(merged.mean()) if len(merged) > 0 else 0.0
            else:
                self.col_means[c] = 0.0

        print(f"均值计算完成，共 {len(self.col_means)} 列")
        print("=" * 60)
        return self.feature_cols

    def transform(self, file_path):
        """返回 (values, mask)，均为 (T, N) 的 float32 数组。"""
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        df.columns = [c.lstrip('\ufeff') if isinstance(c, str) else c
                      for c in df.columns]

        df_features = df.iloc[:, 1:]
        df_aligned = df_features.reindex(columns=self.feature_cols)

        exist_mask = np.zeros(len(self.feature_cols), dtype=np.float32)
        for i, c in enumerate(self.feature_cols):
            if c in df_features.columns:
                exist_mask[i] = 1.0

        values = df_aligned.apply(pd.to_numeric, errors='coerce').values.astype(np.float32)
        mask = np.isfinite(values).astype(np.float32) * exist_mask[None, :]

        df_filled = df_aligned.ffill().bfill()
        for i, c in enumerate(self.feature_cols):
            if exist_mask[i] < 0.5:
                df_filled[c] = self.col_means[c]
        values_filled = df_filled.values.astype(np.float32)
        values_filled = np.nan_to_num(values_filled, nan=0.0,
                                      posinf=0.0, neginf=0.0)
        values_filled = values_filled * mask
        return values_filled, mask

    def fit_transform(self, file_label_pairs):
        self.fit(file_label_pairs)
        aligned = []
        for file_path, label in file_label_pairs:
            features, mask = self.transform(file_path)
            aligned.append((features, mask, label, file_path))
            print(f"  {os.path.basename(file_path)}: "
                  f"shape={features.shape}, 有效列比例={mask.mean():.3f}")
        return aligned


# ============================================================
# 3. 数据集类（value + mask + label）
# ============================================================
class FaultDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        window_value, window_mask, label = self.sequences[index]
        value = torch.as_tensor(window_value, dtype=torch.float32)
        mask = torch.as_tensor(window_mask, dtype=torch.float32)
        return value, mask, torch.tensor(label, dtype=torch.long)


# ============================================================
# 4. 滑窗工具
# ============================================================
def make_windows_from_segment(segment_value, segment_mask,
                              window_size, stride):
    n = len(segment_value)
    num_windows = (n - window_size) // stride + 1
    if num_windows <= 0:
        return []
    windows = []
    for i in range(num_windows):
        start = i * stride
        end = start + window_size
        windows.append((
            segment_value[start:end],
            segment_mask[start:end],
            start,
        ))
    return windows


# ============================================================
# 5. 数据加载
# ============================================================
def data_loader(aligned_data, config, scaler=None, mode='train'):
    print(f">>>>>>>>>>>>>>>>>>>>>>>>>>创建数据加载器 ({mode}) <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    window_size = config.in_len
    stride = getattr(config, 'stride', 8)

    if mode == 'train':
        all_features = np.vstack([d[0] for d in aligned_data])
        all_masks = np.vstack([d[1] for d in aligned_data])
        n_nan = np.isnan(all_features).sum()
        n_inf = np.isinf(all_features).sum()
        print(f"训练数据检查: shape={all_features.shape}, NaN={n_nan}, Inf={n_inf}")
        if n_nan > 0 or n_inf > 0:
            all_features = np.nan_to_num(all_features, nan=0.0,
                                         posinf=0.0, neginf=0.0)
        scaler = MaskedStandardScaler()
        scaler.fit(all_features, all_masks)
        data_dim = all_features.shape[1]
    else:
        if scaler is None:
            raise ValueError("测试模式必须传入预训练的 scaler")
        data_dim = len(scaler.mean)

    sequences = []
    label_counts = {}

    for features, mask, label, file_path in aligned_data:
        name = os.path.basename(file_path)
        if features.shape[1] != data_dim:
            raise ValueError(f"{name} 维度不一致")

        features = scaler.transform(features, mask)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        num_samples = len(features)
        split_point = num_samples // 2

        normal_windows = make_windows_from_segment(
            features[:split_point], mask[:split_point], window_size, stride)
        for win_v, win_m, _ in normal_windows:
            sequences.append((win_v, win_m, 0))
            label_counts[0] = label_counts.get(0, 0) + 1

        fault_windows = make_windows_from_segment(
            features[split_point:], mask[split_point:], window_size, stride)
        for win_v, win_m, _ in fault_windows:
            sequences.append((win_v, win_m, label))
            label_counts[label] = label_counts.get(label, 0) + 1

        print(f"  {name}: 总长={num_samples}, 正常段=前{split_point}, "
              f"正常窗口={len(normal_windows)}, 故障窗口={len(fault_windows)}")

    dataset = FaultDataset(sequences)
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size,
        shuffle=config.shuffle if mode == 'train' else False,
        drop_last=False, num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    print(f"传感器维度: {data_dim}")
    print(f"总样本数: {len(dataset)}")
    print(f"各类别窗口数: {dict(sorted(label_counts.items()))}")
    print(f">>>>>>>>>>>>>>>>>>>>>>>>>>数据加载器创建完成<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    return dataloader, scaler, data_dim


# ============================================================
# 6. DepCA 通道注意力
# ============================================================
class DepCA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.depthwise = nn.Conv1d(channels, channels, kernel_size=3,
                                   padding=1, groups=channels)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        k = min(7, max(3, int(np.log2(channels)) + 1))
        if k % 2 == 0:
            k += 1
        self.att_conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x):
        feat = self.depthwise(x)
        feat = self.pointwise(feat)
        feat = self.bn(feat)
        feat = F.relu(feat, inplace=True)

        avg_pool = F.adaptive_avg_pool1d(feat, 1).squeeze(-1)
        att = avg_pool.unsqueeze(1)
        att = self.att_conv(att)
        att = self.sigmoid(att).squeeze(1)

        weighted = feat * att.unsqueeze(-1)
        return weighted, att


# ============================================================
# 7. QR-Ortho 损失
# ============================================================
class QROrthoLoss(nn.Module):
    def forward(self, class_prototypes):
        C = class_prototypes.shape[0]
        if C < 2:
            return torch.tensor(0.0, device=class_prototypes.device)
        Q, _ = torch.linalg.qr(class_prototypes, mode='reduced')
        gram = Q.T @ Q
        I = torch.eye(gram.shape[0], device=gram.device)
        return F.mse_loss(gram, I)


# ============================================================
# 8. Liquid Time-Constant Cell
# ============================================================
class LiquidCell(nn.Module):
    """
    Liquid Time-Constant Cell，tau 固定为常数。
    dh/dt = -(1/tau + f) * h + f * A
    离散化：h_next = (h + dt * f * A) / (1 + dt * (1/tau + f))
    """
    def __init__(self, input_size, hidden_size, tau=1.0, A=1.0, dt=0.05):
        super().__init__()
        self.hidden_size = hidden_size
        self.dt = dt
        self.register_buffer('tau', torch.ones(hidden_size) * tau)
        self.A = nn.Parameter(torch.ones(hidden_size) * A)
        self.W_in = nn.Linear(input_size, hidden_size, bias=True)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.xavier_uniform_(self.W_in.weight, gain=0.5)
        nn.init.orthogonal_(self.W_h.weight, gain=0.5)

    def forward(self, x, h=None):
        B, T, D = x.shape
        if h is None:
            h = torch.zeros(B, self.hidden_size, device=x.device)
        outputs = []
        tau_clamped = torch.clamp(self.tau, min=0.01, max=100.0)
        for t in range(T):
            x_t = x[:, t, :]
            f = torch.tanh(self.W_in(x_t) + self.W_h(h))
            denominator = 1 + self.dt * (1 / tau_clamped + f)
            h = (h + self.dt * f * self.A) / denominator
            outputs.append(h.unsqueeze(1))
        return torch.cat(outputs, dim=1)


# ============================================================
# 9. LNN + DepCA + QR-Ortho（含掩码）
# ============================================================
class MaskedLNNSVM(nn.Module):
    def __init__(self, num_sensors, num_classes,
                 lnn_hidden=128, lnn_layers=2,
                 dropout=0.3, lnn_dt=0.05, use_diff=True):
        super().__init__()
        self.num_sensors = num_sensors
        self.num_classes = num_classes
        self.use_diff = use_diff

        # 传感器级注意力只作用在原始传感器上，保证可解释
        self.sensor_att = DepCA(num_sensors)

        # LNN 的输入维度：如果拼接差分，则为 2 * num_sensors
        lnn_input_dim = num_sensors * 2 if use_diff else num_sensors

        self.lnn_cells = nn.ModuleList()
        in_size = lnn_input_dim
        for i in range(lnn_layers):
            out_size = lnn_hidden if i == lnn_layers - 1 else max(lnn_hidden // 2, 16)
            self.lnn_cells.append(LiquidCell(
                in_size, out_size,
                tau=1.0 * (1 + i * 0.5),
                A=1.0 * (1 + i * 0.3),
                dt=lnn_dt
            ))
            in_size = out_size

        self.feature_att = DepCA(lnn_hidden)
        self.pool_proj = nn.Linear(lnn_hidden * 2, lnn_hidden)

        self.classifier = nn.Sequential(
            nn.Linear(lnn_hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

        self.qr_ortho = QROrthoLoss()
        # 原型维度：num_sensors（传感器级注意力）
        self.register_buffer('class_prototypes',
                             torch.zeros(num_classes, num_sensors))
        self.register_buffer('prototype_counts', torch.zeros(num_classes))

    def forward(self, value, mask, return_features=False, return_attention=False):
        """
        value: (B, T, N)
        mask:  (B, T, N)
        """
        # 掩码：缺失位置置 0
        x = value * mask

        # 传感器级注意力：作用在 (B, N, T) 上
        x_t = x.transpose(1, 2)                      # (B, N, T)
        x_weighted, sensor_att = self.sensor_att(x_t)
        x_weighted = x_weighted.transpose(1, 2)      # (B, T, N)

        # 拼接一阶差分
        if self.use_diff:
            x_diff = torch.diff(x_weighted, dim=1, prepend=x_weighted[:, :1, :])
            x_input = torch.cat([x_weighted, x_diff], dim=-1)  # (B, T, 2N)
        else:
            x_input = x_weighted

        h = x_input
        for cell in self.lnn_cells:
            h = cell(h)                                        # (B, T, lnn_hidden)

        h_att, _ = self.feature_att(h.transpose(1, 2))
        mean_pool = h_att.mean(dim=-1)
        max_pool = h_att.max(dim=-1)[0]
        pooled = torch.cat([mean_pool, max_pool], dim=-1)
        pooled = self.pool_proj(pooled)

        if return_features:
            return pooled

        logits = self.classifier(pooled)
        if return_attention:
            return logits, sensor_att
        return logits

    def update_prototypes(self, attention, labels):
        with torch.no_grad():
            for c in range(self.num_classes):
                mask_c = (labels == c)
                if mask_c.sum() > 0:
                    batch_mean = attention[mask_c].mean(dim=0)
                    old = self.class_prototypes[c]
                    n = self.prototype_counts[c]
                    self.class_prototypes[c] = (old * n + batch_mean) / (n + 1)
                    self.prototype_counts[c] += 1

    def get_ortho_loss(self):
        valid = self.prototype_counts > 0
        if valid.sum() < 2:
            return torch.tensor(0.0, device=self.class_prototypes.device)
        return self.qr_ortho(self.class_prototypes[valid])


# ============================================================
# 10. 训练增强（掩码随机丢弃）
# ============================================================
def augment_mask(mask, mask_prob=0.1, group_mask_prob=0.05, n_groups=8):
    B, T, N = mask.shape
    device = mask.device

    rand = torch.rand(B, T, N, device=device)
    drop = (rand < mask_prob).float()
    mask_aug = mask * (1 - drop)

    if group_mask_prob > 0:
        group_size = (N + n_groups - 1) // n_groups
        group_rand = torch.rand(B, T, n_groups, device=device)
        group_drop = (group_rand < group_mask_prob).float()
        for g in range(n_groups):
            start = g * group_size
            end = min((g + 1) * group_size, N)
            mask_aug[:, :, start:end] = (
                mask_aug[:, :, start:end] * (1 - group_drop[:, :, g:g + 1])
            )
    return mask_aug


# ============================================================
# 11. 训练
# ============================================================
def split_train_val(sequences, val_ratio=0.2, seed=0):
    rng = np.random.RandomState(seed)
    by_class = {}
    for seq in sequences:
        by_class.setdefault(seq[2], []).append(seq)

    train_seqs, val_seqs = [], []
    for c, items in by_class.items():
        idx = rng.permutation(len(items))
        n_val = max(1, int(len(items) * val_ratio))
        val_idx = set(idx[:n_val])
        for i, item in enumerate(items):
            if i in val_idx:
                val_seqs.append(item)
            else:
                train_seqs.append(item)
    return train_seqs, val_seqs


def train(model, args, train_loader, val_loader, device):
    start_time = time.time()

    all_labels = []
    for _, _, batch_y in train_loader:
        all_labels.append(batch_y.numpy())
    all_labels = np.concatenate(all_labels)
    class_counts = np.bincount(all_labels, minlength=args.num_classes)
    print(f"\n训练集类别分布: {dict(enumerate(class_counts))}")

    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * args.num_classes
    print(f"类别权重: {np.round(class_weights, 4)}")

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32).to(device)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    results_loss = []
    best_val_f1 = 0.0

    for epoch in tqdm(range(args.epochs), desc="Training"):
        model.train()
        total_loss = []
        correct, total = 0, 0
        nan_batches = 0

        # 训练后期逐步降低掩码概率
        decay = 1.0 - 0.5 * epoch / args.epochs
        cur_mask_prob = args.mask_prob * decay
        cur_group_prob = args.group_mask_prob * decay

        for batch_v, batch_m, batch_y in train_loader:
            batch_v = batch_v.to(device)
            batch_m = batch_m.to(device)
            batch_y = batch_y.to(device)

            if torch.isnan(batch_v).any() or torch.isinf(batch_v).any():
                nan_batches += 1
                continue

            batch_m_aug = augment_mask(
                batch_m, cur_mask_prob, cur_group_prob
            )
            batch_v_aug = batch_v * batch_m_aug

            optimizer.zero_grad()
            logits, att = model(batch_v_aug, batch_m_aug,
                                return_attention=True)

            if torch.isnan(logits).any():
                nan_batches += 1
                continue

            cls_loss = criterion(logits, batch_y)
            model.update_prototypes(att, batch_y)
            ortho_loss = model.get_ortho_loss()
            loss = cls_loss + args.ortho_weight * ortho_loss

            if torch.isnan(loss) or torch.isinf(loss):
                nan_batches += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss.append(loss.item())
            pred = logits.argmax(dim=1)
            correct += (pred == batch_y).sum().item()
            total += batch_y.size(0)

        scheduler.step()

        if len(total_loss) == 0:
            continue

        avg_loss = sum(total_loss) / len(total_loss)
        train_acc = correct / total if total > 0 else 0.0

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch_v, batch_m, batch_y in val_loader:
                batch_v = batch_v.to(device)
                batch_m = batch_m.to(device)
                logits = model(batch_v, batch_m)
                val_preds.append(logits.argmax(dim=1).cpu().numpy())
                val_labels.append(batch_y.numpy())
        val_preds = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels)
        val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)

        tqdm.write(f"\t Epoch {epoch + 1}/{args.epochs}, "
                   f"Loss: {avg_loss:.6f}, Train Acc: {train_acc:.4f}, "
                   f"Val Macro F1: {val_f1:.4f}"
                   + (f", NaN batches: {nan_batches}" if nan_batches > 0 else ""))
        results_loss.append(avg_loss)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'best_model.pth')

    torch.save(model.state_dict(), 'last_model.pth')
    print(f">>>>>>>>>>>>>>>>>>>>>>训练完成, 用时:{(time.time() - start_time) / 60:.4f} min<<<<<<<<<<<<<<<<<<")
    return results_loss


# ============================================================
# 12. 特征提取 + 直接预测
# ============================================================
@torch.no_grad()
def extract_features(model, dataloader, device):
    model.eval()
    features, labels = [], []
    for batch_v, batch_m, batch_y in tqdm(dataloader, desc="Extracting Features"):
        batch_v = batch_v.to(device)
        batch_m = batch_m.to(device)
        batch_features = model(batch_v, batch_m, return_features=True)
        features.append(batch_features.cpu().numpy())
        labels.append(batch_y.numpy())
    features = np.vstack(features)
    labels = np.concatenate(labels)
    if np.isnan(features).any() or np.isinf(features).any():
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features, labels


@torch.no_grad()
def predict_direct(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch_v, batch_m, batch_y in tqdm(dataloader, desc="Direct Prediction"):
        batch_v = batch_v.to(device)
        batch_m = batch_m.to(device)
        logits = model(batch_v, batch_m)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(batch_y.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


# ============================================================
# 13. 绘图
# ============================================================
def plot_loss_data(data):
    if len(data) == 0:
        print("无有效损失数据可绘制")
        return
    plt.figure(figsize=(10, 5))
    plt.plot(data, marker='o')
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Loss"])
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_percent = np.nan_to_num(cm_percent, nan=0.0)
    rounded_cm = np.round(cm_percent, 3)
    annot_array = np.array([[f'0' if val == 0 else f'{val:.3f}' for val in row]
                            for row in rounded_cm])

    plt.figure(figsize=(max(8, len(class_names)), max(6, len(class_names) * 0.8)))
    ax = sns.heatmap(rounded_cm, annot=annot_array, fmt='', cmap='Blues',
                     linewidths=0.5, cbar=True,
                     xticklabels=class_names, yticklabels=class_names)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45,
                       ha='right', rotation_mode='anchor')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_attention_heatmap(attention_weights, class_names, feature_names=None):
    plt.figure(figsize=(max(10, attention_weights.shape[1] * 0.15), 6))
    sns.heatmap(attention_weights, cmap='viridis', cbar=True,
                xticklabels=feature_names if feature_names is not None else False,
                yticklabels=class_names)
    plt.xlabel('Feature / Sensor')
    plt.ylabel('Fault Class')
    plt.title('Average Sensor Attention per Fault Class')
    plt.tight_layout()
    plt.show()


def plot_attention_bar(attention_weights, class_names, feature_names=None, top_k=10):
    num_classes = attention_weights.shape[0]
    fig, axes = plt.subplots(num_classes, 1,
                             figsize=(10, 2.5 * num_classes),
                             squeeze=False)
    for c in range(num_classes):
        ax = axes[c][0]
        weights = attention_weights[c]
        idx = np.argsort(weights)[::-1][:top_k]
        names = [feature_names[i] if feature_names is not None else f"F{i}"
                 for i in idx]
        ax.barh(range(len(idx)), weights[idx][::-1], color='steelblue')
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels(names[::-1])
        ax.set_title(f"{class_names[c]} - Top-{top_k} Features")
    plt.tight_layout()
    plt.show()


# ============================================================
# 14. 主程序
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LNN Fault Diagnosis with Feature Alignment')
    parser.add_argument('--in_len', type=int, default=32)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--lnn_hidden', type=int, default=128)
    parser.add_argument('--lnn_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lnn_dt', type=float, default=0.05)
    parser.add_argument('--ortho_weight', type=float, default=0.1)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--use_diff', type=int, default=1,
                        help='是否在 LNN 输入里拼接一阶差分，1=是，0=否')
    parser.add_argument('--mask_prob', type=float, default=0.1)
    parser.add_argument('--group_mask_prob', type=float, default=0.05)

    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-weight_decay', type=float, default=1e-4)
    parser.add_argument('-epochs', type=int, default=30)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-shuffle', action='store_true', default=True)
    parser.add_argument('-device', type=int, default=0)

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() \
        else torch.device("cpu")
    print("使用设备:", device)

    train_files = [
        ('D:\\142418-刘极-诊断助理工程师\\风机数据\\训练\\24-01-28-23_24_35_354_21294_PitchSystemSi.csv', 1),
        ('D:\\142418-刘极-诊断助理工程师\\风机数据\\训练\\25-02-07-01_34_03_189_21285_PitchHeartbea.csv', 2),
        ('D:\\142418-刘极-诊断助理工程师\\风机数据\\训练\\25-11-10-11_01_16_569_21035_PitchOverSpee.csv', 3),
        ('D:\\142418-刘极-诊断助理工程师\\风机数据\\训练\\26-07-22-19_59_31_679_21005_PitchBatteryV.csv', 4),
        ('D:\\142418-刘极-诊断助理工程师\\风机数据\\训练\\26-07-30-05_33_12_989_21048_PitchGetWrong.csv', 5),
        ('D:\\142418-刘极-诊断助理工程师\\风机数据\\训练\\26-09-09-03_37_09_700_21291_PitchFollowin.csv', 6),
    ]
    test_files = [
        ('D:\\142418-刘极-诊断助理工程师\\风机数据\\测试\\24-01-29-15_35_23_354_21294_PitchSystemSi.csv', 1),
        ('D:\\142418-刘极-诊断助理工程师\\风机数据\\测试\\25-02-07-08_21_18_189_21285_PitchHeartbea.csv', 2),
        ('D:\\142418-刘极-诊断助理工程师\\风机数据\\测试\\26-04-17-17_49_50_569_21035_PitchOverSpee.csv', 3),
        ('D:\\142418-刘极-诊断助理工程师\\风机数据\\测试\\26-09-08-19_43_04_459_21005_PitchBatteryV.csv', 4),
        ('D:\\142418-刘极-诊断助理工程师\\风机数据\\测试\\26-07-29-22_14_09_989_21048_PitchGetWrong.csv', 5),
        ('D:\\142418-刘极-诊断助理工程师\\风机数据\\测试\\26-09-09-07_01_01_129_21291_PitchFollowin.csv', 6),
    ]

    preprocessor = DataPreprocessor(debug=True)
    aligned_train = preprocessor.fit_transform(train_files)
    feature_cols = preprocessor.feature_cols

    aligned_test_data = []
    for file_path, label in test_files:
        features, mask = preprocessor.transform(file_path)
        aligned_test_data.append((features, mask, label, file_path))
        print(f"  {os.path.basename(file_path)}: "
              f"shape={features.shape}, 有效列比例={mask.mean():.3f}")

    train_loader_full, scaler, data_dim = data_loader(
        aligned_train, args, mode='train')
    test_loader, _, _ = data_loader(
        aligned_test_data, args, scaler=scaler, mode='test')
    print(f"统一后传感器维度: {data_dim}")

    all_train_seqs = train_loader_full.dataset.sequences
    train_seqs, val_seqs = split_train_val(all_train_seqs,
                                            val_ratio=args.val_ratio,
                                            seed=0)
    print(f"训练集样本数: {len(train_seqs)}, 验证集样本数: {len(val_seqs)}")

    train_loader = DataLoader(
        FaultDataset(train_seqs),
        batch_size=args.batch_size, shuffle=True, drop_last=False,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        FaultDataset(val_seqs),
        batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )

    model = MaskedLNNSVM(
        num_sensors=data_dim,
        num_classes=args.num_classes,
        lnn_hidden=args.lnn_hidden,
        lnn_layers=args.lnn_layers,
        dropout=args.dropout,
        lnn_dt=args.lnn_dt,
        use_diff=bool(args.use_diff)
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型可训练参数量: {n_params:,}")
    print(f"LNN 输入拼接一阶差分: {bool(args.use_diff)}")

    loss_curve = train(model, args, train_loader, val_loader, device)
    plot_loss_data(loss_curve)

    model.load_state_dict(torch.load('best_model.pth', map_location=device,
                                     weights_only=True))
    print("已加载验证集最佳模型")

    print("\n" + "=" * 60)
    print("评估方式一：模型直接分类")
    print("=" * 60)
    direct_pred, direct_labels = predict_direct(model, test_loader, device)
    direct_acc = accuracy_score(direct_labels, direct_pred)
    direct_f1_macro = f1_score(direct_labels, direct_pred, average='macro', zero_division=0)
    direct_f1_weighted = f1_score(direct_labels, direct_pred, average='weighted', zero_division=0)
    print(f"Test Accuracy: {direct_acc:.4f}")
    print(f"Macro F1: {direct_f1_macro:.4f}")
    print(f"Weighted F1: {direct_f1_weighted:.4f}")
    direct_cm = confusion_matrix(direct_labels, direct_pred,
                                 labels=list(range(args.num_classes)))
    print(direct_cm)

    print("\n" + "=" * 60)
    print("评估方式二：模型特征 + SVM")
    print("=" * 60)
    svm_train_loader = DataLoader(
        FaultDataset(train_seqs),
        batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )
    train_features, train_labels = extract_features(model, svm_train_loader, device)
    test_features, test_labels = extract_features(model, test_loader, device)

    feat_scaler = SklearnScaler()
    train_features_scaled = feat_scaler.fit_transform(train_features)
    test_features_scaled = feat_scaler.transform(test_features)

    svm = SVC(kernel='rbf', C=5.0, gamma='scale', class_weight='balanced')
    svm.fit(train_features_scaled, train_labels)
    svm_pred = svm.predict(test_features_scaled)

    svm_acc = accuracy_score(test_labels, svm_pred)
    svm_f1_macro = f1_score(test_labels, svm_pred, average='macro', zero_division=0)
    svm_f1_weighted = f1_score(test_labels, svm_pred, average='weighted', zero_division=0)
    print(f"SVM Test Accuracy: {svm_acc:.4f}")
    print(f"SVM Macro F1: {svm_f1_macro:.4f}")
    print(f"SVM Weighted F1: {svm_f1_weighted:.4f}")
    svm_cm = confusion_matrix(test_labels, svm_pred,
                              labels=list(range(args.num_classes)))
    print(svm_cm)

    print("\n" + "=" * 60)
    print("对比总结")
    print("=" * 60)
    print(f"{'方法':<20} {'Accuracy':<12} {'Macro F1':<12} {'Weighted F1':<12}")
    print(f"{'模型直接分类':<20} {direct_acc:<12.4f} {direct_f1_macro:<12.4f} {direct_f1_weighted:<12.4f}")
    print(f"{'模型 + SVM':<20} {svm_acc:<12.4f} {svm_f1_macro:<12.4f} {svm_f1_weighted:<12.4f}")

    class_names = ["Normal"] + [f"Fault{i}" for i in range(1, args.num_classes)]
    plot_confusion_matrix(direct_cm, class_names,
                          title="Direct - Confusion Matrix")
    plot_confusion_matrix(svm_cm, class_names,
                          title="SVM - Confusion Matrix")

    model.eval()
    attention_by_class = {c: [] for c in range(args.num_classes)}
    with torch.no_grad():
        for batch_v, batch_m, batch_y in test_loader:
            batch_v = batch_v.to(device)
            batch_m = batch_m.to(device)
            _, att = model(batch_v, batch_m, return_attention=True)
            att = att.cpu().numpy()
            for i, label in enumerate(batch_y.numpy()):
                attention_by_class[label].append(att[i])

    # 维度：num_sensors（传感器级注意力）
    avg_attention = np.zeros((args.num_classes, data_dim))
    for c in range(args.num_classes):
        if len(attention_by_class[c]) > 0:
            avg_attention[c] = np.mean(attention_by_class[c], axis=0)

    plot_attention_heatmap(avg_attention, class_names, feature_names=feature_cols)
    plot_attention_bar(avg_attention, class_names, feature_names=feature_cols, top_k=10)