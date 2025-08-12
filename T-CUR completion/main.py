import torch
import torch.fft
import numpy as np

# 设置随机种子，保证可复现性
torch.manual_seed(42)


def tensor_cur_complete(tensor, mask, d=10, l=10, max_iter=50, tau=1.0):
    """
    基于Tensor-CUR分解的缺失数据恢复主函数
    参数：
        tensor: 输入三维张量 (I1, I2, I3)，含缺失值（缺失值可设为0，由mask指示）
        mask: 掩码张量 (I1, I2, I3)，1表示观测值，0表示缺失值
        d: 每个切片采样的列数（根据论文定理3设置，通常与log(I1+I2)相关）
        l: 每个切片采样的行数（同上）
        max_iter: SVT算法的最大迭代次数
        tau: SVT算法的阈值参数
    返回：
        recovered_tensor: 恢复后的张量 (I1, I2, I3)
    """
    I1, I2, I3 = tensor.shape  # 张量维度
    
    # 步骤1：对张量沿第三维做傅里叶变换（进入频域）
    # 注：论文中使用FFT沿第三维，这里用torch.fft.fftn实现
    fft_tensor = torch.fft.fftn(tensor, dim=2)  # 结果为复数张量 (I1, I2, I3)
    fft_mask = mask.float()  # 掩码同样用于频域处理（仅保留观测值）
    
    # 步骤2：对每个频域切片应用M-CUR分解
    fft_recovered = torch.zeros_like(fft_tensor, dtype=torch.complex64)
    for k in range(I3):
        # 取第k个频域切片（二维矩阵）和对应的掩码
        slice_k = fft_tensor[..., k]  # (I1, I2)
        mask_k = fft_mask[..., k]     # (I1, I2)
        
        # 对当前切片应用M-CUR分解（算法2）
        recovered_slice = m_cur_complete(slice_k, mask_k, d, l, max_iter, tau)
        fft_recovered[..., k] = recovered_slice
    
    # 步骤3：逆傅里叶变换回到原域（取实部，因原始数据为实数）
    recovered_tensor = torch.fft.ifftn(fft_recovered, dim=2).real
    
    return recovered_tensor


def m_cur_complete(matrix, mask, d, l, max_iter, tau):
    """
    矩阵CUR（M-CUR）分解补全函数（对应论文算法2）
    参数：
        matrix: 频域中的二维切片 (I1, I2)
        mask: 该切片的掩码 (I1, I2)
        d: 采样列数
        l: 采样行数
        max_iter: SVT最大迭代次数
        tau: SVT阈值
    返回：
        recovered_matrix: 补全后的矩阵 (I1, I2)
    """
    I1, I2 = matrix.shape
    observed = matrix * mask  # 仅保留观测值（缺失值为0）
    
    # 步骤1：从观测数据中随机采样列和行
    # 采样列（确保采样的列至少有一个观测值）
    valid_cols = torch.where(mask.sum(dim=0) > 0)[0]  # 有观测值的列索引
    if len(valid_cols) < d:
        d = len(valid_cols)  # 若有效列不足，调整采样数
    sampled_col_idx = torch.randperm(len(valid_cols))[:d]
    sampled_cols = valid_cols[sampled_col_idx]  # 采样的列索引
    C_obs = observed[:, sampled_cols]  # 采样得到的列矩阵 (I1, d)
    C_mask = mask[:, sampled_cols]     # 采样列的掩码
    
    # 采样行（同理）
    valid_rows = torch.where(mask.sum(dim=1) > 0)[0]
    if len(valid_rows) < l:
        l = len(valid_rows)
    sampled_row_idx = torch.randperm(len(valid_rows))[:l]
    sampled_rows = valid_rows[sampled_row_idx]  # 采样的行索引
    R_obs = observed[sampled_rows, :]  # 采样得到的行矩阵 (l, I2)
    R_mask = mask[sampled_rows, :]     # 采样行的掩码
    
    # 步骤2：用SVT补全采样得到的列和行矩阵（MC-BASE步骤）
    C_hat = svt(C_obs, C_mask, max_iter, tau)  # 补全后的列矩阵 (I1, d)
    R_hat = svt(R_obs, R_mask, max_iter, tau)  # 补全后的行矩阵 (l, I2)
    
    # 步骤3：计算C和R的交集矩阵W（采样行和采样列的交集）
    W = observed[sampled_rows[:, None], sampled_cols[None, :]]  # (l, d)
    # 用SVT补全W中的缺失值
    W_mask = mask[sampled_rows[:, None], sampled_cols[None, :]]  # (l, d)
    W_hat = svt(W, W_mask, max_iter, tau)
    
    # 步骤4：计算U的伪逆（U = W_hat的伪逆）
    U_hat = torch.linalg.pinv(W_hat)  # (d, l)，伪逆求解
    
    # 步骤5：得到当前切片的补全结果 Y = C_hat * U_hat * R_hat
    recovered_matrix = C_hat @ U_hat @ R_hat  # (I1, I2)
    
    return recovered_matrix


def svt(matrix, mask, max_iter, tau):
    """修正后的奇异值阈值法，支持复数矩阵"""
    X = torch.zeros_like(matrix)  # 初始化补全矩阵（保持复数类型）
    for _ in range(max_iter):
        # 对复数矩阵进行SVD（PyTorch的linalg.svd支持复数）
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        
        # 奇异值阈值处理（阈值操作不改变类型）
        S_thresh = torch.clamp(S - tau, min=0.0)
        
        # 重构矩阵时保持复数类型（diag生成复数对角矩阵）
        X_new = U @ torch.diag(S_thresh.type_as(U)) @ Vh
        
        # 投影到观测值上（确保观测值不变）
        X = X_new * (1 - mask) + matrix * mask
    return X

def generate_tensor(shape, rank, device='cpu'):
    """
    基于TSVD生成指定tubal rank的三维张量
    参数：
        shape: 张量形状 (I1, I2, I3)
        rank: 指定的tubal rank（需满足 rank ≤ min(I1, I2)）
        device: 计算设备
    返回：
        tensor: 具有指定rank的三维张量（实数类型）
    """
    I1, I2, I3 = shape
    if rank > min(I1, I2):
        raise ValueError(f"rank必须 ≤ min(I1, I2)，当前min为{min(I1, I2)}")
    
    # 步骤1：生成频域中的正交矩阵U_hat和V_hat（每个切片为正交矩阵）
    U_hat = []
    V_hat = []
    for _ in range(I3):
        # 生成随机矩阵并通过QR分解得到正交矩阵
        U_rand = torch.randn(I1, I1, device=device)
        U_orth, _ = torch.linalg.qr(U_rand)
        U_hat.append(U_orth)
        
        V_rand = torch.randn(I2, I2, device=device)
        V_orth, _ = torch.linalg.qr(V_rand)
        V_hat.append(V_orth)
    
    # 步骤2：生成频域中的f-对角张量S_hat（奇异值数量为指定rank）
    S_hat = []
    for _ in range(I3):
        diag_vals = torch.zeros(min(I1, I2), device=device)
        diag_vals[:rank] = torch.rand(rank, device=device) + 0.5  # 确保奇异值为正
        S_slice = torch.diag(diag_vals)
        # 扩展为I1×I2的对角矩阵（若I1≠I2）
        if I1 > I2:
            S_slice = torch.cat([S_slice, torch.zeros(I1 - I2, I2, device=device)], dim=0)
        elif I2 > I1:
            S_slice = torch.cat([S_slice, torch.zeros(I1, I2 - I1, device=device)], dim=1)
        S_hat.append(S_slice)
    
    # 步骤3：在频域中计算 U_hat * S_hat * V_hat^T（等价于原域的t-乘积）
    A_hat = []
    for k in range(I3):
        # 频域中第k个切片的矩阵乘法
        A_slice_hat = U_hat[k] @ S_hat[k] @ V_hat[k].T
        A_hat.append(A_slice_hat)
    A_hat = torch.stack(A_hat, dim=2)  # 合并为频域张量 (I1, I2, I3)
    
    # 步骤4：逆傅里叶变换回到原域，取实部（原始张量为实数）
    tensor = torch.fft.ifftn(A_hat, dim=2).real
    
    return tensor

# ------------------------------
# 示例：测试算法效果
# ------------------------------
if __name__ == "__main__":
    # 生成模拟数据：3维张量 (200, 200, 10)，含30%缺失值
    I1, I2, I3 = 200, 200, 200
    # 生成低秩张量（模拟真实数据的低秩特性）
    tensor = generate_tensor((I1, I2, I3), rank=20)
    # 人为添加50%缺失值
    mask = torch.bernoulli(torch.ones(I1, I2, I3) * 0.9)  # 10%观测值
    tensor_with_missing = tensor * mask  # 缺失值设为0
    
    # 用Tensor-CUR算法恢复缺失值
    recovered = tensor_cur_complete(
        tensor_with_missing, 
        mask, 
        d=20,  # 采样列数（根据论文设为log(I1+I2)级别）
        l=20,  # 采样行数
        max_iter=50, 
        tau=0.5
    )
    
    # 计算恢复误差（仅对缺失部分）
    missing_mask = 1 - mask
    mse = torch.mean(((tensor - recovered) * missing_mask) **2)
    print(f"缺失部分的MSE: {mse:.6f}")