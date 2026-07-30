"""
networkE.py - 药效团引导的图注意力网络 (Pharmacophore-Guided Graph Attention Network)

核心创新点：
- PharmacophoreGATConv: 在计算注意力系数时引入药效团匹配偏置
- 注意力公式: α_ij = softmax_j(BaseScore(i,j) + β_edge * MatchScore(p_j, TPP)
                  + β_interact * InteractScore(p_i, p_j))
- β_edge 和 β_interact 是逐层可学习标量；p_i/p_j 始终为原始六维药效团特征
"""

import torch
import torch.nn as nn
from features import num_atom_features, num_bond_features
import numpy as np
from graph_featurization import build_molecular_graph_tensors
from torch.utils.data import Dataset
import torch.nn.functional as F
import os

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")
print(f'num interop threads: {torch.get_num_interop_threads()}, num intraop threads: {torch.get_num_threads()}')


# ============================================================
# 药效团特征配置
# ============================================================
# 药效团特征在原子特征向量中的位置 (最后6位)
PHARMACOPHORE_FEATURE_DIM = 6
PHARMACOPHORE_FEATURE_NAMES = [
    'is_h_bond_donor',
    'is_h_bond_acceptor',
    'is_aromatic',
    'is_positive',
    'is_negative',
    'is_hydrophobic',
]


TPP_COUNT_KEYS = [
    'hbond_donor_count',
    'hbond_acceptor_count',
    'aromatic_ring_count',
    'positive_charge_count',
    'negative_charge_count',
    'hydrophobic_count',
]


def load_tpp_vector(tpp_path, mode='binary', count_clip=20.0):
    """
    加载 Target Pharmacophore Profile (TPP) 向量
    
    Args:
        tpp_path: .pt 文件路径
    
    Returns:
        tpp_tensor:
            - binary: 六维 presence 向量
            - binary_counts: presence + log-normalized counts，共十二维
    """
    if tpp_path is None or not os.path.exists(tpp_path):
        print(f"Warning: TPP file not found at {tpp_path}, using default TPP")
        # 默认 TPP: 假设所有药效团特征都重要
        default_binary = torch.ones(
            PHARMACOPHORE_FEATURE_DIM, device=device
        )
        if mode == 'binary':
            return default_binary
        if mode == 'binary_counts':
            return torch.cat([default_binary, default_binary])
        raise ValueError(f"Unknown TPP input mode: {mode}")
    
    tpp_data = torch.load(tpp_path, map_location=device)
    tpp_dict = tpp_data.get('tpp_dict', {})
    
    # 提取布尔特征 (has_xxx)
    tpp_bool = torch.tensor([
        float(tpp_dict.get('has_hbond_donor', 1)),
        float(tpp_dict.get('has_hbond_acceptor', 1)),
        float(tpp_dict.get('has_aromatic', 1)),
        float(tpp_dict.get('has_positive_charge', 0)),
        float(tpp_dict.get('has_negative_charge', 0)),
        float(tpp_dict.get('has_hydrophobic', 1)),
    ], device=device, dtype=torch.float32)
    
    if mode == 'binary':
        tpp_vector = tpp_bool
    elif mode == 'binary_counts':
        counts = torch.tensor(
            [
                float(tpp_dict.get(key, tpp_dict.get(f'{key}_mean', 0.0)))
                for key in TPP_COUNT_KEYS
            ],
            device=device,
            dtype=torch.float32,
        )
        count_clip = max(float(count_clip), 1.0)
        count_strength = (
            torch.log1p(counts.clamp(min=0.0, max=count_clip))
            / np.log1p(count_clip)
        )
        tpp_vector = torch.cat([tpp_bool, count_strength], dim=0)
    else:
        raise ValueError(f"Unknown TPP input mode: {mode}")

    print(
        f"Loaded TPP from {tpp_path} "
        f"(mode={mode}): {tpp_vector.tolist()}"
    )
    return tpp_vector


def prepare_tpp_batch(tpp_vector, batch_size, dtype, target_device):
    """Convert a global or per-sample TPP tensor to shape [batch, dim]."""
    tpp = torch.as_tensor(
        tpp_vector, dtype=dtype, device=target_device
    )
    if tpp.dim() == 1:
        tpp = tpp.unsqueeze(0)
    if tpp.dim() != 2:
        raise ValueError(
            f"TPP must have shape [dim] or [batch, dim], got {tuple(tpp.shape)}"
        )
    if tpp.shape[0] == 1 and batch_size != 1:
        tpp = tpp.expand(batch_size, -1)
    elif tpp.shape[0] != batch_size:
        raise ValueError(
            f"TPP batch size {tpp.shape[0]} does not match molecule "
            f"batch size {batch_size}"
        )
    if tpp.shape[-1] < PHARMACOPHORE_FEATURE_DIM:
        raise ValueError(
            f"TPP must contain at least {PHARMACOPHORE_FEATURE_DIM} values, "
            f"got {tpp.shape[-1]}"
        )
    return tpp


# ============================================================
# 药效团引导的图注意力卷积层 (PharmacophoreGATConv)
# ============================================================
class PharmacophoreGATConv(nn.Module):
    """
    药效团引导的图注意力卷积层 (增强版)
    
    创新点：在标准 GAT 注意力计算中引入边级别的药效团交互偏置
    
    注意力公式:
        α_ij = softmax_j(LeakyReLU(a^T [Wh_i || Wh_j])
                         + β_edge * MatchScore(p_j, TPP)
                         + β_interact * InteractScore(p_i, p_j))
    
    其中:
        - W: 特征变换矩阵
        - a: 注意力向量
        - β_edge: 邻居节点与 TPP 匹配偏置强度 (逐层可学习标量)
        - β_interact: 药效团交互偏置强度 (可学习)
        - MatchScore: 原子药效团特征与 TPP 的匹配度
        - p_i/p_j: 始终使用输入图中的原始六维药效团二值特征
        - InteractScore: 中心-邻居原子的局部药效团配对分数

    说明:
        仅依赖中心节点 i 的加性偏置对其所有邻居均相同，会在邻居维度
        softmax 中严格相消，因此模型不再包含无效的 β_node 项。
    """
    
    def __init__(self, in_features, out_features, tpp_vector, num_heads=1, 
                 negative_slope=0.2, dropout=0.0, concat=True, disable_tpp_bias=False, hard_concat_tpp=False, fixed_bias=False):
        """
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            tpp_vector: Target Pharmacophore Profile 向量 [6,]
            num_heads: 注意力头数
            negative_slope: LeakyReLU 负斜率
            dropout: Dropout 比率
            concat: 是否拼接多头输出 (True) 或取平均 (False)
        """
        super(PharmacophoreGATConv, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.concat = concat
        self.disable_tpp_bias = disable_tpp_bias
        self.hard_concat_tpp = hard_concat_tpp
        self.fixed_bias = fixed_bias
        
        # 强制: hard_concat_tpp 模式下禁用动态偏置
        if self.hard_concat_tpp:
            self.disable_tpp_bias = True
        
        # 注册 TPP 为 buffer (不参与梯度更新，但会随模型保存/加载)
        self.register_buffer('tpp_vector', tpp_vector)
        
        # 特征变换矩阵 W: [in_features, num_heads * out_features]
        self.W = nn.Parameter(torch.empty(in_features, num_heads * out_features))
        nn.init.xavier_uniform_(self.W)
        
        # 注意力向量 a: [num_heads, 2 * out_features]
        # 用于计算 a^T [Wh_i || Wh_j]
        self.a_src = nn.Parameter(torch.empty(num_heads, out_features))
        self.a_dst = nn.Parameter(torch.empty(num_heads, out_features))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        
        # ============================================================
        # 药效团偏置参数 (核心创新)
        # ============================================================
        if self.fixed_bias:
            # Fixed Bias 消融: β 固定为常数，不参与训练
            self.register_buffer('beta_edge', torch.tensor(0.3))
            self.register_buffer('beta_interact', torch.tensor(0.4))
        else:
            # β_edge: 边级偏置 - 邻居原子匹配 TPP 的程度
            self.beta_edge = nn.Parameter(torch.tensor(0.3))
            
            # β_interact: 中心-邻居原子的局部药效团配对偏置
            self.beta_interact = nn.Parameter(torch.tensor(0.4))
        
        # ============================================================
        # 药效团特征交互网络 (学习药效团之间的协同关系)
        # ============================================================
        # 输入: 源原子药效团特征 [6] + 目标原子药效团特征 [6] = [12]
        # 输出: 交互分数 [num_heads]
        self.pharm_interact_net = nn.Sequential(
            nn.Linear(PHARMACOPHORE_FEATURE_DIM * 2, 16),
            nn.ReLU(),
            nn.Linear(16, num_heads),
        )
        
        # 药效团特征重要性权重 (学习哪些药效团特征更重要)
        # 6 个药效团特征: donor, acceptor, aromatic, positive, negative, hydrophobic
        self.pharm_importance = nn.Parameter(torch.ones(PHARMACOPHORE_FEATURE_DIM))
        
        # 偏置
        self.bias = nn.Parameter(torch.zeros(num_heads * out_features if concat else out_features))
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.feat_dropout = nn.Dropout(dropout)
        
        # Hard Concatenation: TPP 静态拼接投影层 (节点级)
        if self.hard_concat_tpp:
            self.tpp_node_projection = nn.Sequential(
                nn.Linear(out_features + PHARMACOPHORE_FEATURE_DIM, out_features),
                nn.ReLU(),
            )
        
        self.to(device)
    
    def compute_pharmacophore_match_score(
        self, pharm_features, target_tpp=None
    ):
        """
        计算原子药效团特征与 TPP 的加权匹配分数
        
        Args:
            pharm_features: [batch, num_atoms, 6] 原始药效团二值特征
        
        Returns:
            match_scores: [batch, num_atoms] 匹配分数 (0-1)
        """
        if pharm_features.size(-1) != PHARMACOPHORE_FEATURE_DIM:
            raise ValueError(
                "pharm_features must contain the six original pharmacophore "
                f"indicators, got shape {tuple(pharm_features.shape)}"
            )
        
        batch_size = pharm_features.shape[0]
        if target_tpp is None:
            target_tpp = self.tpp_vector
        tpp = prepare_tpp_batch(
            target_tpp,
            batch_size=batch_size,
            dtype=pharm_features.dtype,
            target_device=pharm_features.device,
        )[..., :PHARMACOPHORE_FEATURE_DIM].unsqueeze(1)
        
        # 应用可学习的重要性权重
        importance = F.softmax(self.pharm_importance, dim=0).view(1, 1, -1)  # [1, 1, 6]
        
        # 加权点积匹配: 原子有该特征 AND TPP 也需要该特征，乘以重要性权重
        numerator = (pharm_features * tpp * importance).sum(dim=-1)
        
        # 归一化匹配分数
        denominator = (tpp * importance).sum(dim=-1).clamp_min(1e-8)
        match_scores = numerator / denominator  # [batch, num_atoms]
        
        return match_scores
    
    def compute_edge_pharmacophore_interaction(self, src_pharm, dst_pharm):
        """
        计算边级别的药效团交互分数
        
        Args:
            src_pharm: [batch, num_atoms, max_degree, 6] 源原子药效团特征
            dst_pharm: [batch, num_atoms, max_degree, 6] 目标原子药效团特征
        
        Returns:
            interact_scores: [batch, num_atoms, max_degree, num_heads] 交互分数
        """
        batch_size, num_atoms, max_degree, _ = src_pharm.shape
        
        # 拼接源和目标的药效团特征
        combined = torch.cat([src_pharm, dst_pharm], dim=-1)  # [batch, num_atoms, max_degree, 12]
        
        # 通过交互网络计算分数
        # 需要 reshape 以适应 Linear 层
        combined_flat = combined.view(-1, PHARMACOPHORE_FEATURE_DIM * 2)
        interact_flat = self.pharm_interact_net(combined_flat)  # [batch*num_atoms*max_degree, num_heads]
        interact_scores = interact_flat.view(batch_size, num_atoms, max_degree, self.num_heads)
        
        return interact_scores
    
    def forward(
        self, atoms, edges, raw_pharm_features, target_tpp=None
    ):
        """
        前向传播
        
        Args:
            atoms: [batch, num_atoms, in_features] 原子特征
            edges: [batch, num_atoms, max_degree] 邻接边索引 (-1 表示无边)
            raw_pharm_features: [batch, num_atoms, 6] 原始药效团二值特征
        
        Returns:
            new_atoms: [batch, num_atoms, out_features * num_heads] 或 [batch, num_atoms, out_features]
        """
        batch_size, num_atoms, _ = atoms.shape
        _, _, max_degree = edges.shape
        
        atoms = atoms.to(device)
        edges = edges.to(device)
        raw_pharm_features = raw_pharm_features.to(
            device=device, dtype=atoms.dtype
        )
        if raw_pharm_features.shape[:2] != atoms.shape[:2]:
            raise ValueError(
                "raw_pharm_features must align with atoms on batch and atom "
                f"dimensions: atoms={tuple(atoms.shape)}, "
                f"raw_pharm_features={tuple(raw_pharm_features.shape)}"
            )

        if target_tpp is None:
            target_tpp = self.tpp_vector
        tpp_batch = prepare_tpp_batch(
            target_tpp,
            batch_size=batch_size,
            dtype=atoms.dtype,
            target_device=atoms.device,
        )
        attention_tpp = tpp_batch[..., :PHARMACOPHORE_FEATURE_DIM]
        
        # 1. 特征变换: [batch, num_atoms, num_heads * out_features]
        h = torch.matmul(atoms, self.W)
        h = h.view(batch_size, num_atoms, self.num_heads, self.out_features)
        # h: [batch, num_atoms, num_heads, out_features]
        
        # Hard Concatenation: 将 TPP 静态拼接到变换后的节点特征
        if self.hard_concat_tpp:
            tpp_expanded = attention_tpp.view(
                batch_size, 1, 1, PHARMACOPHORE_FEATURE_DIM
            ).expand(
                batch_size, num_atoms, self.num_heads, -1
            )  # [batch, num_atoms, num_heads, 6]
            h_concat = torch.cat([h, tpp_expanded], dim=-1)  # [..., out_features + 6]
            h = self.tpp_node_projection(h_concat)  # [..., out_features]
        
        # 2. 使用输入图中保留的原始药效团特征。
        # 不从隐藏表示末六维重新切片，以免第二层以后丢失药效团语义。
        pharm_features = raw_pharm_features
        
        # 3. 计算每个原子与 TPP 的匹配分数
        pharm_scores = self.compute_pharmacophore_match_score(
            pharm_features, attention_tpp
        )  # [batch, num_atoms]
        
        # 4. 计算标准 GAT 注意力分数
        # 源节点注意力: a_src^T * h_i
        attn_src = (h * self.a_src.view(1, 1, self.num_heads, self.out_features)).sum(dim=-1)
        # attn_src: [batch, num_atoms, num_heads]
        
        # 目标节点注意力: a_dst^T * h_j
        attn_dst = (h * self.a_dst.view(1, 1, self.num_heads, self.out_features)).sum(dim=-1)
        # attn_dst: [batch, num_atoms, num_heads]
        
        # 5. 聚合邻居信息
        # 创建有效边掩码
        edge_mask = (edges != -1).float()  # [batch, num_atoms, max_degree]
        
        # 将 -1 替换为 0 用于索引 (后面会用 mask 过滤)
        safe_edges = edges.clamp(min=0)  # [batch, num_atoms, max_degree]
        
        # 获取邻居特征
        batch_idx = torch.arange(batch_size, device=device).view(-1, 1, 1).expand(-1, num_atoms, max_degree)
        neighbor_h = h[batch_idx, safe_edges]  # [batch, num_atoms, max_degree, num_heads, out_features]
        
        # 获取邻居的注意力分数 (目标节点)
        neighbor_attn_dst = attn_dst[batch_idx, safe_edges]  # [batch, num_atoms, max_degree, num_heads]
        
        # 获取邻居的药效团匹配分数
        neighbor_pharm_scores = pharm_scores[batch_idx, safe_edges]  # [batch, num_atoms, max_degree]
        
        # 获取邻居的药效团特征
        neighbor_pharm_features = pharm_features[batch_idx, safe_edges]  # [batch, num_atoms, max_degree, 6]
        
        # 6. 计算完整的注意力系数 (核心创新部分)
        # ============================================================
        
        # 6.1 标准 GAT 注意力
        attn_src_expanded = attn_src.unsqueeze(2).expand(-1, -1, max_degree, -1)
        e = F.leaky_relu(attn_src_expanded + neighbor_attn_dst, negative_slope=self.negative_slope)
        # e: [batch, num_atoms, max_degree, num_heads]
        
        # 6.2 ~ 6.4 药效团偏置 (消融实验: disable_tpp_bias=True 时跳过)
        if not self.disable_tpp_bias:
            # 6.2 邻居节点与 TPP 的匹配偏置。
            # 该项随邻居 j 变化，因此不会在 softmax_j 中相消。
            edge_bias = self.beta_edge * neighbor_pharm_scores.unsqueeze(-1)
            edge_bias = edge_bias.expand(-1, -1, -1, self.num_heads)
            # edge_bias: [batch, num_atoms, max_degree, num_heads]
            
            # 6.3 中心-邻居原子的局部药效团配对偏置
            src_pharm_expanded = pharm_features.unsqueeze(2).expand(-1, -1, max_degree, -1)
            # src_pharm_expanded: [batch, num_atoms, max_degree, 6]
            
            # 计算交互分数
            interact_scores = self.compute_edge_pharmacophore_interaction(
                src_pharm_expanded, neighbor_pharm_features
            )  # [batch, num_atoms, max_degree, num_heads]
            
            interact_bias = self.beta_interact * interact_scores
            
            # 6.4 组合有效偏置
            e = e + edge_bias + interact_bias
        
        # ============================================================
        
        # 7. Softmax 归一化 (只在有效邻居上)
        e = e.masked_fill(edge_mask.unsqueeze(-1) == 0, float('-inf'))
        
        # Softmax
        alpha = F.softmax(e, dim=2)  # [batch, num_atoms, max_degree, num_heads]
        
        # 处理全是 -inf 的情况 (孤立节点)
        alpha = torch.nan_to_num(alpha, nan=0.0)
        
        # Dropout
        alpha = self.attn_dropout(alpha)
        
        # 8. 聚合邻居特征
        alpha_expanded = alpha.unsqueeze(-1)  # [batch, num_atoms, max_degree, num_heads, 1]
        
        # 加权求和
        out = (alpha_expanded * neighbor_h).sum(dim=2)  # [batch, num_atoms, num_heads, out_features]
        
        # 9. 输出处理
        if self.concat:
            out = out.view(batch_size, num_atoms, -1)
        else:
            out = out.mean(dim=2)
        
        out = out + self.bias
        
        return out


# ============================================================
# 原有的辅助模块 (保持兼容性)
# ============================================================
class GraphLookup(nn.Module):
    def __init__(self):
        super().__init__()
        self.to(device)

    def temporal_padding(self, x, paddings=(1, 0), pad_value=0):
        if not isinstance(paddings, (tuple, list, np.ndarray)):
            paddings = (paddings, paddings)
        output = torch.zeros(x.size(0), x.size(1) + sum(paddings), x.size(2), device=device)
        output[:, :paddings[0], :] = pad_value
        output[:, paddings[1]:, :] = pad_value
        output[:, paddings[0]: paddings[0]+x.size(1), :] = x
        return output
    
    def lookup_neighbors(self, atoms, edges, maskvalue=0, include_self=False):
        masked_edges = edges + 1
        masked_atoms = self.temporal_padding(atoms, (1, 0), pad_value=maskvalue)

        batch_n, lookup_size, n_atom_features = masked_atoms.size()
        _, max_atoms, max_degree = masked_edges.size()

        expanded_atoms = masked_atoms.unsqueeze(2).expand(batch_n, lookup_size, max_degree, n_atom_features)
        expanded_edges = masked_edges.unsqueeze(3).expand(batch_n, max_atoms, max_degree, n_atom_features)
        output = torch.gather(expanded_atoms, 1, expanded_edges)

        if include_self:
            return torch.cat([(atoms.view(batch_n, max_atoms, 1, n_atom_features)), output], dim=2)
        return output

    def forward(self, atoms, edges, maskvalue=0, include_self=True):
        atoms, edges = atoms.to(device), edges.to(device)
        return self.lookup_neighbors(atoms, edges, maskvalue, include_self)


class nfpConv(nn.Module):
    """原有的 NFP 卷积层 (保留用于对比实验)"""
    def __init__(self, ishape, oshape):
        super(nfpConv, self).__init__()
        self.ishape = ishape
        self.oshape = oshape

        self.w = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty((self.ishape, self.oshape))), requires_grad=True).to(device)
        self.b = nn.Parameter(torch.nn.init.constant_(torch.empty((1, self.oshape)), 0.01), requires_grad=True).to(device)
        
        self.degArr = nn.ParameterList([nn.Parameter(torch.nn.init.xavier_normal_(torch.empty((self.ishape + 6, self.oshape))), requires_grad=True).to(device) for _ in range(6)])
       
        self.graphLookup = GraphLookup()  
        self.to(device)

    def forward(self, input):
        atoms, bonds, edges = input
        atoms, bonds, edges = atoms.to(device), bonds.to(device), edges.to(device)
        atom_degrees = (edges != -1).sum(-1, keepdim=True)
        neighbor_atom_features = self.graphLookup(atoms, edges, include_self=True)
        summed_atom_features = neighbor_atom_features.sum(-2)
        summed_bond_features = bonds.sum(-2)
        summed_features = torch.cat([summed_atom_features, summed_bond_features], dim=-1)

        new_features = None
        for degree in range(6):
            atom_masks_this_degree = (atom_degrees == degree).float()
            new_unmasked_features = F.sigmoid(torch.matmul(summed_features, self.degArr[degree]) + self.b)
            new_masked_features = new_unmasked_features * atom_masks_this_degree
            new_features = new_masked_features if degree == 0 else new_features + new_masked_features

        return new_features


class nfpOutput(nn.Module):
    def __init__(self, layer, fpl):
        super(nfpOutput, self).__init__()
        self.fpl = fpl
        self.layer = layer

        w = torch.empty((self.layer + num_bond_features(), self.fpl), device=device)
        self.w = nn.Parameter(w)
        b = torch.empty((1, self.fpl), device=device)
        self.b = nn.Parameter(b)

        torch.nn.init.xavier_normal_(self.w)
        torch.nn.init.constant_(self.b, 0.01)

        self.to(device)

    def forward(self, a, b, e):
        a, b, e = a.to(device), b.to(device), e.to(device)
        atom_degrees = (e != -1).sum(-1, keepdim=True)
        general_atom_mask = (atom_degrees != 0).float()
        summed_bond_features = b.sum(-2)
        summed_features = torch.cat([a, summed_bond_features], dim=-1)
        fingerprint_out_unmasked = F.sigmoid(torch.matmul(summed_features, self.w) + self.b)
        fingerprint_out_masked = fingerprint_out_unmasked * general_atom_mask

        return fingerprint_out_masked.sum(dim=-2), fingerprint_out_masked
    

class GraphPool(nn.Module):
    def __init__(self):
        super(GraphPool, self).__init__()
        self.graphLookup = GraphLookup()
        self.to(device)

    def forward(self, atoms, edges):
        atoms, edges = atoms.to(device), edges.to(device)
        neighbor_atom_features = self.graphLookup(atoms, edges, maskvalue=-np.inf, include_self=True)
        max_features = neighbor_atom_features.max(dim=2)[0]
        atom_degrees = (edges != -1).sum(dim=-1, keepdim=True)
        general_atom_mask = (atom_degrees != 0).float()
        return max_features * general_atom_mask


# ============================================================
# 多尺度特征聚合模块 (Multi-Scale Feature Aggregation)
# ============================================================
class MultiScaleAggregator(nn.Module):
    """
    多尺度特征聚合模块
    
    创新点：
    1. 可学习的层级权重 - 自动学习每层指纹的重要性
    2. 注意力聚合 - 基于指纹内容动态调整权重
    3. 门控机制 - 控制每层信息的流动
    
    聚合公式:
        fingerprint = Σ (α_i * gate_i * fp_i)
        其中 α_i = softmax(learnable_weights) 或 attention(fp_i)
    """
    
    def __init__(self, num_layers, fpl, aggregation_type='attention'):
        """
        Args:
            num_layers: 层数 (包括最终输出层)
            fpl: 指纹长度
            aggregation_type: 聚合类型
                - 'learned': 可学习的固定权重
                - 'attention': 基于内容的注意力权重
                - 'gated': 门控聚合
                - 'hierarchical': 层级注意力 (考虑层级关系)
        """
        super(MultiScaleAggregator, self).__init__()
        
        self.num_layers = num_layers
        self.fpl = fpl
        self.aggregation_type = aggregation_type
        
        # ============================================================
        # 方法 1: 可学习的固定权重
        # ============================================================
        # 初始化为均匀分布，让模型自己学习
        self.layer_weights = nn.Parameter(torch.ones(num_layers))
        
        # ============================================================
        # 方法 2: 基于内容的注意力权重
        # ============================================================
        # 将每层指纹映射到注意力分数
        self.attention_net = nn.Sequential(
            nn.Linear(fpl, fpl // 2),
            nn.Tanh(),
            nn.Linear(fpl // 2, 1, bias=False)
        )
        
        # ============================================================
        # 方法 3: 门控机制
        # ============================================================
        # 每层一个门控网络，决定该层信息的流动量
        self.gate_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fpl, fpl // 4),
                nn.ReLU(),
                nn.Linear(fpl // 4, fpl),
                nn.Sigmoid()
            ) for _ in range(num_layers)
        ])
        
        # ============================================================
        # 方法 4: 层级注意力 (考虑层级位置)
        # ============================================================
        # 层级位置编码
        self.layer_position_embedding = nn.Parameter(torch.randn(num_layers, fpl // 4))
        self.hierarchical_attention = nn.Sequential(
            nn.Linear(fpl + fpl // 4, fpl // 2),
            nn.Tanh(),
            nn.Linear(fpl // 2, 1, bias=False)
        )
        
        # ============================================================
        # 最终融合层 (可选)
        # ============================================================
        self.fusion_layer = nn.Sequential(
            nn.Linear(fpl, fpl),
            nn.LayerNorm(fpl),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.to(device)
    
    def forward(self, fingerprints):
        """
        聚合多层指纹
        
        Args:
            fingerprints: list of [batch, fpl] 各层的指纹
        
        Returns:
            aggregated: [batch, fpl] 聚合后的指纹
            weights: [batch, num_layers] 各层权重 (用于可视化)
        """
        batch_size = fingerprints[0].shape[0]
        
        # 堆叠所有层的指纹: [batch, num_layers, fpl]
        fp_stack = torch.stack(fingerprints, dim=1)
        
        if self.aggregation_type == 'learned':
            # 方法 1: 可学习的固定权重
            weights = F.softmax(self.layer_weights, dim=0)  # [num_layers]
            weights = weights.view(1, -1, 1).expand(batch_size, -1, self.fpl)
            aggregated = (fp_stack * weights).sum(dim=1)  # [batch, fpl]
            weight_output = weights[:, :, 0]  # [batch, num_layers]
            
        elif self.aggregation_type == 'attention':
            # 方法 2: 基于内容的注意力权重
            # 计算每层指纹的注意力分数
            attn_scores = self.attention_net(fp_stack)  # [batch, num_layers, 1]
            attn_scores = attn_scores.squeeze(-1)  # [batch, num_layers]
            weights = F.softmax(attn_scores, dim=1)  # [batch, num_layers]
            weights_expanded = weights.unsqueeze(-1)  # [batch, num_layers, 1]
            aggregated = (fp_stack * weights_expanded).sum(dim=1)  # [batch, fpl]
            weight_output = weights
            
        elif self.aggregation_type == 'gated':
            # 方法 3: 门控聚合
            gated_fps = []
            for i, (fp, gate_net) in enumerate(zip(fingerprints, self.gate_nets)):
                gate = gate_net(fp)  # [batch, fpl]
                gated_fps.append(fp * gate)
            
            # 使用可学习权重聚合门控后的指纹
            gated_stack = torch.stack(gated_fps, dim=1)  # [batch, num_layers, fpl]
            weights = F.softmax(self.layer_weights, dim=0)
            weights = weights.view(1, -1, 1).expand(batch_size, -1, self.fpl)
            aggregated = (gated_stack * weights).sum(dim=1)
            weight_output = weights[:, :, 0]
            
        elif self.aggregation_type == 'hierarchical':
            # 方法 4: 层级注意力
            # 为每层指纹添加位置编码
            pos_emb = self.layer_position_embedding.unsqueeze(0).expand(batch_size, -1, -1)
            # pos_emb: [batch, num_layers, fpl//4]
            
            fp_with_pos = torch.cat([fp_stack, pos_emb], dim=-1)
            # fp_with_pos: [batch, num_layers, fpl + fpl//4]
            
            attn_scores = self.hierarchical_attention(fp_with_pos).squeeze(-1)
            # attn_scores: [batch, num_layers]
            
            weights = F.softmax(attn_scores, dim=1)
            weights_expanded = weights.unsqueeze(-1)
            aggregated = (fp_stack * weights_expanded).sum(dim=1)
            weight_output = weights
            
        else:
            raise ValueError(f"Unknown aggregation type: {self.aggregation_type}")
        
        # 可选：通过融合层进一步处理
        aggregated = self.fusion_layer(aggregated)
        
        return aggregated, weight_output


# ============================================================
# GIN 卷积层 (Graph Isomorphism Network Convolution)
# ============================================================
class GINConv(nn.Module):
    """
    Graph Isomorphism Network 卷积层 (拓扑通道)
    
    核心公式:
        h_i^(k+1) = MLP( (1 + ε) * h_i^(k) + Σ_{j∈N(i)} h_j^(k) )
    
    其中 ε 是可学习参数，MLP 比简单线性变换有更强的表达力。
    GIN 被证明在图同构测试上达到 WL-test 的上界，
    能捕捉 GAT 难以建模的子图结构信息。
    """
    
    def __init__(self, in_features, out_features, hidden_dim=None, dropout=0.1):
        """
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            hidden_dim: MLP 中间层维度 (默认与 out_features 相同)
            dropout: Dropout 比率
        """
        super(GINConv, self).__init__()
        
        if hidden_dim is None:
            hidden_dim = out_features
        
        # 可学习参数 ε (初始为 0，让模型学习自环的重要性)
        self.epsilon = nn.Parameter(torch.zeros(1))
        
        # 两层 MLP + BatchNorm (增强表达力)
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_features),
        )
        
        self.graphLookup = GraphLookup()
        self.to(device)
    
    def forward(self, atoms, edges):
        """
        前向传播
        
        Args:
            atoms: [batch, num_atoms, in_features] 原子特征
            edges: [batch, num_atoms, max_degree] 邻接边索引
        
        Returns:
            new_atoms: [batch, num_atoms, out_features]
        """
        atoms = atoms.to(device)
        edges = edges.to(device)
        
        batch_size, num_atoms, in_features = atoms.shape
        
        # 获取邻居特征 (不含自身)
        neighbor_features = self.graphLookup(atoms, edges, maskvalue=0, include_self=False)
        # neighbor_features: [batch, num_atoms, max_degree, in_features]
        
        # 聚合邻居: 求和
        neighbor_sum = neighbor_features.sum(dim=2)  # [batch, num_atoms, in_features]
        
        # GIN 更新: (1 + ε) * h_self + Σ h_neighbor
        out = (1 + self.epsilon) * atoms + neighbor_sum  # [batch, num_atoms, in_features]
        
        # 通过 MLP (需要 reshape 为 2D 以兼容 BatchNorm1d)
        out_flat = out.view(-1, in_features)            # [batch*num_atoms, in_features]
        out_flat = self.mlp(out_flat)                    # [batch*num_atoms, out_features]
        out = out_flat.view(batch_size, num_atoms, -1)   # [batch, num_atoms, out_features]
        
        return out


# ============================================================
# 跨通道注意力融合 (Cross-Channel Attention)
# ============================================================
class CrossChannelAttention(nn.Module):
    """
    跨通道注意力融合模块
    
    让药效团通道和拓扑通道的原子表示互相增强:
      h_pharm' = h_pharm + α * CrossAttn(Q=h_pharm, K=h_topo, V=h_topo)
      h_topo'  = h_topo  + α * CrossAttn(Q=h_topo,  K=h_pharm, V=h_pharm)
    
    其中 α 是可学习的融合强度系数。
    """
    
    def __init__(self, dim, num_heads=4, dropout=0.1):
        """
        Args:
            dim: 特征维度 (两个通道必须维度一致)
            num_heads: 注意力头数
            dropout: Dropout 比率
        """
        super(CrossChannelAttention, self).__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        # Pharm → Topo 的跨注意力 (Query 来自 pharm, Key/Value 来自 topo)
        self.W_q_p2t = nn.Linear(dim, dim)
        self.W_k_p2t = nn.Linear(dim, dim)
        self.W_v_p2t = nn.Linear(dim, dim)
        
        # Topo → Pharm 的跨注意力 (Query 来自 topo, Key/Value 来自 pharm)
        self.W_q_t2p = nn.Linear(dim, dim)
        self.W_k_t2p = nn.Linear(dim, dim)
        self.W_v_t2p = nn.Linear(dim, dim)
        
        # 输出投影
        self.out_proj_pharm = nn.Linear(dim, dim)
        self.out_proj_topo = nn.Linear(dim, dim)
        
        # 可学习的融合强度 (初始较小，防止破坏各通道独立学习)
        self.alpha_pharm = nn.Parameter(torch.tensor(0.1))
        self.alpha_topo = nn.Parameter(torch.tensor(0.1))
        
        # 归一化和 Dropout
        self.norm_pharm = nn.LayerNorm(dim)
        self.norm_topo = nn.LayerNorm(dim)
        self.attn_dropout = nn.Dropout(dropout)
        
        self.scale = self.head_dim ** -0.5
        self.to(device)
    
    def _cross_attn(self, query, key, value, W_q, W_k, W_v, out_proj, mask=None):
        """
        计算跨通道注意力
        
        Args:
            query: [batch, num_atoms, dim] 查询通道
            key:   [batch, num_atoms, dim] 键值通道
            value: [batch, num_atoms, dim] 键值通道
            mask:  [batch, num_atoms] 有效原子掩码
        
        Returns:
            output: [batch, num_atoms, dim]
        """
        batch_size, num_atoms, _ = query.shape
        
        # 线性投影并 reshape 为多头
        Q = W_q(query).view(batch_size, num_atoms, self.num_heads, self.head_dim).transpose(1, 2)
        K = W_k(key).view(batch_size, num_atoms, self.num_heads, self.head_dim).transpose(1, 2)
        V = W_v(value).view(batch_size, num_atoms, self.num_heads, self.head_dim).transpose(1, 2)
        # Q, K, V: [batch, num_heads, num_atoms, head_dim]
        
        # 注意力分数
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # attn: [batch, num_heads, num_atoms, num_atoms]
        
        # 掩码 (忽略填充原子)
        if mask is not None:
            mask_2d = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, num_atoms]
            attn = attn.masked_fill(mask_2d == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.attn_dropout(attn)
        
        # 加权聚合
        out = torch.matmul(attn, V)  # [batch, num_heads, num_atoms, head_dim]
        out = out.transpose(1, 2).contiguous().view(batch_size, num_atoms, -1)
        out = out_proj(out)
        
        return out
    
    def forward(self, h_pharm, h_topo, atom_mask=None):
        """
        双向跨通道注意力融合
        
        Args:
            h_pharm: [batch, num_atoms, dim] 药效团通道表示
            h_topo:  [batch, num_atoms, dim] 拓扑通道表示
            atom_mask: [batch, num_atoms] 有效原子掩码 (可选)
        
        Returns:
            h_pharm_new: [batch, num_atoms, dim] 融合后的药效团通道
            h_topo_new:  [batch, num_atoms, dim] 融合后的拓扑通道
        """
        # Pharm 从 Topo 获取信息
        cross_p = self._cross_attn(
            h_pharm, h_topo, h_topo,
            self.W_q_p2t, self.W_k_p2t, self.W_v_p2t,
            self.out_proj_pharm, atom_mask
        )
        
        # Topo 从 Pharm 获取信息
        cross_t = self._cross_attn(
            h_topo, h_pharm, h_pharm,
            self.W_q_t2p, self.W_k_t2p, self.W_v_t2p,
            self.out_proj_topo, atom_mask
        )
        
        # 残差融合 + LayerNorm
        h_pharm_new = self.norm_pharm(h_pharm + self.alpha_pharm * cross_p)
        h_topo_new = self.norm_topo(h_topo + self.alpha_topo * cross_t)
        
        return h_pharm_new, h_topo_new


class TPPEncoder(nn.Module):
    """Encode a binary or binary+count TPP into a dense condition vector."""

    def __init__(self, input_dim, hidden_dim=64, dropout=0.1):
        super(TPPEncoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

    def forward(self, tpp):
        return self.network(tpp)


class ConditionalFiLM(nn.Module):
    """Apply identity-initialized target conditioning to node features."""

    def __init__(self, condition_dim, feature_dim, scale=0.1):
        super(ConditionalFiLM, self).__init__()
        self.scale = float(scale)
        self.affine = nn.Linear(condition_dim, feature_dim * 2)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, features, condition):
        gamma, beta = self.affine(condition).chunk(2, dim=-1)
        gamma = self.scale * torch.tanh(gamma)
        beta = self.scale * beta
        return (
            features * (1.0 + gamma.unsqueeze(1))
            + beta.unsqueeze(1)
        )


# ============================================================
# 双通道异构消息传递图神经网络 (DC-PharmGAT)
# ============================================================
class DualChannelPharmGAT(nn.Module):
    """
    双通道异构消息传递图神经网络 (Dual-Channel Heterogeneous Message Passing)
    
    创新点:
      通道A (药效团通道): PharmacophoreGATConv - 利用 TPP 引导注意力
      通道B (拓扑通道):   GINConv - 捕捉子图同构结构信息
      跨通道融合: CrossChannelAttention - 让两个通道互相增强
    
    每层的计算流程:
      1. 通道 A: h_pharm = PharmGATConv(atoms, edges)
      2. 通道 B: h_topo  = GINConv(atoms, edges)
      3. 跨通道:  h_pharm, h_topo = CrossAttn(h_pharm, h_topo)
      4. 残差 + 归一化 + 池化
      5. 收集指纹: fp_pharm, fp_topo
    
    最终输出: concat(agg(fp_pharm), agg(fp_topo)) → 投影到 fpl
    """
    
    def __init__(
        self, layers, tpp_vector, return_activations=False, fpl=32,
        num_heads=4, dropout=0.1, aggregation_type='attention',
        disable_tpp_bias=False, hard_concat_tpp=False, fixed_bias=False,
        enable_tpp_conditioning=False, tpp_hidden_dim=64,
        tpp_film_scale=0.1,
    ):
        """
        Args:
            layers: 各层特征维度列表，如 [68, 64, 64, 64, 64]
            tpp_vector: Target Pharmacophore Profile 向量
            return_activations: 是否返回中间激活
            fpl: 指纹长度
            num_heads: 注意力头数
            dropout: Dropout 比率
            aggregation_type: 多尺度聚合类型
        """
        super(DualChannelPharmGAT, self).__init__()
        
        self.layers = layers
        self.fpl = fpl
        self.num_heads = num_heads
        self.return_activations = return_activations
        self.atom_activations = None
        self.aggregation_type = aggregation_type
        self.disable_tpp_bias = disable_tpp_bias
        self.hard_concat_tpp = hard_concat_tpp
        self.fixed_bias = fixed_bias
        self.enable_tpp_conditioning = enable_tpp_conditioning
        self.tpp_hidden_dim = int(tpp_hidden_dim)
        
        # 注册 TPP
        self.register_buffer('tpp_vector', tpp_vector)

        if self.enable_tpp_conditioning:
            self.tpp_encoder = TPPEncoder(
                input_dim=tpp_vector.numel(),
                hidden_dim=self.tpp_hidden_dim,
                dropout=dropout,
            )
        
        # 构建网络层
        self.throughShape = list(zip(layers[:-1], layers[1:]))
        
        # ===== 通道 A: 药效团引导的 GAT =====
        self.pharm_gat_layers = nn.ModuleList()
        self.pharm_output_layers = nn.ModuleList()
        self.pharm_layer_norms = nn.ModuleList()
        
        # ===== 通道 B: GIN 拓扑编码 =====
        self.topo_gin_layers = nn.ModuleList()
        self.topo_output_layers = nn.ModuleList()
        self.topo_layer_norms = nn.ModuleList()

        # ===== TPP 条件调制 =====
        self.pharm_film_layers = nn.ModuleList()
        self.topo_film_layers = nn.ModuleList()
        
        # ===== 跨通道注意力 =====
        self.cross_attentions = nn.ModuleList()
        
        for idx, (in_dim, out_dim) in enumerate(self.throughShape):
            # 通道 A: PharmGATConv
            self.pharm_gat_layers.append(PharmacophoreGATConv(
                in_features=in_dim, out_features=out_dim,
                tpp_vector=self.tpp_vector[:PHARMACOPHORE_FEATURE_DIM],
                num_heads=num_heads,
                dropout=dropout, concat=False, disable_tpp_bias=self.disable_tpp_bias,
                hard_concat_tpp=self.hard_concat_tpp, fixed_bias=self.fixed_bias
            ))
            self.pharm_output_layers.append(nfpOutput(in_dim, self.fpl))
            self.pharm_layer_norms.append(nn.LayerNorm(out_dim))
            
            # 通道 B: GINConv
            self.topo_gin_layers.append(GINConv(
                in_features=in_dim, out_features=out_dim,
                hidden_dim=out_dim, dropout=dropout
            ))
            self.topo_output_layers.append(nfpOutput(in_dim, self.fpl))
            self.topo_layer_norms.append(nn.LayerNorm(out_dim))

            if self.enable_tpp_conditioning:
                self.pharm_film_layers.append(ConditionalFiLM(
                    condition_dim=self.tpp_hidden_dim,
                    feature_dim=out_dim,
                    scale=tpp_film_scale,
                ))
                self.topo_film_layers.append(ConditionalFiLM(
                    condition_dim=self.tpp_hidden_dim,
                    feature_dim=out_dim,
                    scale=tpp_film_scale,
                ))
            
            # 跨通道注意力 (在 out_dim 维度上融合)
            self.cross_attentions.append(CrossChannelAttention(
                dim=out_dim, num_heads=num_heads, dropout=dropout
            ))
        
        # 最终输出层
        self.pharm_final_output = nfpOutput(self.layers[-1], self.fpl)
        self.topo_final_output = nfpOutput(self.layers[-1], self.fpl)
        
        # 图池化
        self.pool = GraphPool()
        
        # 多尺度聚合器 (两个通道各一个)
        num_fp_layers = len(self.throughShape) + 1
        self.pharm_aggregator = MultiScaleAggregator(
            num_layers=num_fp_layers, fpl=fpl, aggregation_type=aggregation_type
        )
        self.topo_aggregator = MultiScaleAggregator(
            num_layers=num_fp_layers, fpl=fpl, aggregation_type=aggregation_type
        )
        
        # 双通道指纹融合投影: 2*fpl → fpl
        self.channel_fusion = nn.Sequential(
            nn.Linear(fpl * 2, fpl),
            nn.LayerNorm(fpl),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        if self.enable_tpp_conditioning:
            self.molecule_condition_projection = nn.Sequential(
                nn.Linear(fpl, self.tpp_hidden_dim),
                nn.LayerNorm(self.tpp_hidden_dim),
                nn.SiLU(),
            )
            self.graph_condition_fusion = nn.Sequential(
                nn.Linear(self.tpp_hidden_dim * 4, fpl),
                nn.LayerNorm(fpl),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        
        # Hard Concatenation: 图级 TPP 拼接投影
        if self.hard_concat_tpp:
            self.graph_tpp_projection = nn.Sequential(
                nn.Linear(fpl + PHARMACOPHORE_FEATURE_DIM, fpl),
                nn.LayerNorm(fpl),
                nn.ReLU(),
            )
        
        # 保存监控数据
        self.layer_weights_history = None
        self.cross_attn_alpha_history = None
        
        self.to(device)
    
    def forward(self, input):
        """
        前向传播
        
        Args:
            input: (atoms, bonds, edges) 元组
        
        Returns:
            fingerprint: [batch, fpl] 分子指纹
        """
        self.atom_activations = []
        if len(input) == 4:
            a, b, e, target_tpp = input
        else:
            a, b, e = input
            target_tpp = self.tpp_vector
        a, b, e = a.to(device), b.to(device), e.to(device)
        tpp_batch = prepare_tpp_batch(
            target_tpp,
            batch_size=a.shape[0],
            dtype=a.dtype,
            target_device=a.device,
        )
        tpp_condition = (
            self.tpp_encoder(tpp_batch)
            if self.enable_tpp_conditioning
            else None
        )
        
        # 缓存原始输入特征 (用于 SP-CL 对比学习)
        self._cached_input_features = a

        # 在所有 PharmGAT 层中复用原始六维药效团二值特征。
        # GraphPool 不改变节点数量或排列，因此该张量始终与节点对齐。
        raw_pharm_features = a[..., -PHARMACOPHORE_FEATURE_DIM:].contiguous()
        
        # 两个通道共享初始原子表示
        a_pharm = a  # 药效团通道
        a_topo = a   # 拓扑通道
        
        # 收集各层的指纹
        pharm_fingerprints = []
        topo_fingerprints = []
        
        for i in range(len(self.throughShape)):
            # 1. 收集当前层指纹 (在卷积前)
            fp_pharm, act_pharm = self.pharm_output_layers[i](a_pharm, b, e)
            fp_topo, act_topo = self.topo_output_layers[i](a_topo, b, e)
            pharm_fingerprints.append(fp_pharm)
            topo_fingerprints.append(fp_topo)
            
            if self.return_activations:
                self.atom_activations.append(act_pharm)
            
            # 2. 通道 A: PharmGATConv
            a_pharm_new = self.pharm_gat_layers[i](
                a_pharm, e, raw_pharm_features, tpp_batch
            )
            
            # 3. 通道 B: GINConv
            a_topo_new = self.topo_gin_layers[i](a_topo, e)
            
            # 4. 跨通道注意力融合
            # 计算原子掩码: 有邻边的原子才是有效原子
            atom_mask = (e != -1).any(dim=-1).float()  # [batch, num_atoms]
            a_pharm_new, a_topo_new = self.cross_attentions[i](
                a_pharm_new, a_topo_new, atom_mask
            )
            
            # 5. 残差连接 (如果维度匹配)
            if a_pharm.shape[-1] == a_pharm_new.shape[-1]:
                a_pharm_new = a_pharm_new + a_pharm
                a_topo_new = a_topo_new + a_topo
            
            # 6. 层归一化 + 激活
            a_pharm_new = self.pharm_layer_norms[i](a_pharm_new)
            a_pharm_new = F.elu(a_pharm_new)
            a_topo_new = self.topo_layer_norms[i](a_topo_new)
            a_topo_new = F.elu(a_topo_new)

            if self.enable_tpp_conditioning:
                a_pharm_new = self.pharm_film_layers[i](
                    a_pharm_new, tpp_condition
                )
                a_topo_new = self.topo_film_layers[i](
                    a_topo_new, tpp_condition
                )
            
            # 7. 图池化
            a_pharm = self.pool(a_pharm_new, e)
            a_topo = self.pool(a_topo_new, e)
        
        # 缓存最终原子表示 - 合并两通道 (用于 SP-CL 对比学习)
        self._cached_atom_repr = (a_pharm + a_topo) / 2
        
        # 最终输出层指纹
        fp_pharm_final, act_pharm_final = self.pharm_final_output(a_pharm, b, e)
        fp_topo_final, act_topo_final = self.topo_final_output(a_topo, b, e)
        pharm_fingerprints.append(fp_pharm_final)
        topo_fingerprints.append(fp_topo_final)
        
        if self.return_activations:
            self.atom_activations.append(act_pharm_final)
        
        # 多尺度聚合
        fp_pharm_agg, pharm_weights = self.pharm_aggregator(pharm_fingerprints)
        fp_topo_agg, topo_weights = self.topo_aggregator(topo_fingerprints)
        
        # 保存监控数据
        self.layer_weights_history = pharm_weights.detach()
        self.cross_attn_alpha_history = [
            (ca.alpha_pharm.item(), ca.alpha_topo.item())
            for ca in self.cross_attentions
        ]
        
        # 双通道指纹融合
        fp_concat = torch.cat([fp_pharm_agg, fp_topo_agg], dim=-1)  # [batch, 2*fpl]
        fingerprint = self.channel_fusion(fp_concat)                  # [batch, fpl]

        if self.enable_tpp_conditioning:
            molecule_condition = self.molecule_condition_projection(
                fingerprint
            )
            fused_condition = torch.cat([
                molecule_condition,
                tpp_condition,
                molecule_condition * tpp_condition,
                torch.abs(molecule_condition - tpp_condition),
            ], dim=-1)
            fingerprint = self.graph_condition_fusion(fused_condition)
        
        # Hard Concatenation: 将 TPP 拼接到图级表示
        if self.hard_concat_tpp:
            batch_size_fp = fingerprint.shape[0]
            tpp_graph = tpp_batch[
                ..., :PHARMACOPHORE_FEATURE_DIM
            ].reshape(batch_size_fp, PHARMACOPHORE_FEATURE_DIM)
            fp_with_tpp = torch.cat([fingerprint, tpp_graph], dim=-1)  # [batch, fpl + 6]
            fingerprint = self.graph_tpp_projection(fp_with_tpp)  # [batch, fpl]
        
        return fingerprint
    
    def get_layer_weights(self):
        """获取药效团通道的层级权重"""
        if self.layer_weights_history is not None:
            return self.layer_weights_history.mean(dim=0).cpu().numpy()
        if self.aggregation_type == 'learned':
            weights = F.softmax(self.pharm_aggregator.layer_weights, dim=0)
            return weights.detach().cpu().numpy()
        return None
    
    def get_cross_attn_alphas(self):
        """获取跨通道注意力的融合强度 α (用于监控训练)"""
        return self.cross_attn_alpha_history


# ============================================================
# 药效团引导的图神经网络 (PharmacophoreGAT) - 增强版
# ============================================================
class PharmacophoreGAT(nn.Module):
    """
    药效团引导的图注意力网络 (增强版)
    
    新增特性：
    - 多尺度特征聚合：可学习的层级权重替代简单相加
    - 支持多种聚合策略：learned, attention, gated, hierarchical
    """
    
    def __init__(
        self, layers, tpp_vector, return_activations=False, fpl=32,
        num_heads=4, dropout=0.1, aggregation_type='attention',
        disable_tpp_bias=False, hard_concat_tpp=False, fixed_bias=False,
        enable_tpp_conditioning=False, tpp_hidden_dim=64,
        tpp_film_scale=0.1,
    ):
        """
        Args:
            layers: 各层特征维度列表，如 [68, 64, 64, 64, 64]
            tpp_vector: Target Pharmacophore Profile 向量
            return_activations: 是否返回中间激活
            fpl: 指纹长度
            num_heads: 注意力头数
            dropout: Dropout 比率
            aggregation_type: 多尺度聚合类型
                - 'learned': 可学习的固定权重
                - 'attention': 基于内容的注意力权重 (推荐)
                - 'gated': 门控聚合
                - 'hierarchical': 层级注意力
        """
        super(PharmacophoreGAT, self).__init__()
        
        self.layers = layers
        self.fpl = fpl
        self.num_heads = num_heads
        self.return_activations = return_activations
        self.atom_activations = None
        self.aggregation_type = aggregation_type
        self.disable_tpp_bias = disable_tpp_bias
        self.hard_concat_tpp = hard_concat_tpp
        self.fixed_bias = fixed_bias
        self.enable_tpp_conditioning = enable_tpp_conditioning
        self.tpp_hidden_dim = int(tpp_hidden_dim)
        
        # 注册 TPP
        self.register_buffer('tpp_vector', tpp_vector)

        if self.enable_tpp_conditioning:
            self.tpp_encoder = TPPEncoder(
                input_dim=tpp_vector.numel(),
                hidden_dim=self.tpp_hidden_dim,
                dropout=dropout,
            )
        
        # 构建网络层
        self.throughShape = list(zip(layers[:-1], layers[1:]))
        self.gat_layers, self.output_layers = self._init_layers()
        
        # 最终输出层
        self.final_output = nfpOutput(self.layers[-1], self.fpl)
        
        # 图池化
        self.pool = GraphPool()
        
        # 层归一化 (稳定训练)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(out_dim) for _, out_dim in self.throughShape
        ])
        self.film_layers = nn.ModuleList([
            ConditionalFiLM(
                condition_dim=self.tpp_hidden_dim,
                feature_dim=out_dim,
                scale=tpp_film_scale,
            )
            for _, out_dim in self.throughShape
        ]) if self.enable_tpp_conditioning else nn.ModuleList()
        
        # ============================================================
        # 多尺度特征聚合器 (核心创新)
        # ============================================================
        num_fp_layers = len(self.throughShape) + 1  # 中间层 + 最终输出层
        self.multi_scale_aggregator = MultiScaleAggregator(
            num_layers=num_fp_layers,
            fpl=fpl,
            aggregation_type=aggregation_type
        )

        if self.enable_tpp_conditioning:
            self.molecule_condition_projection = nn.Sequential(
                nn.Linear(fpl, self.tpp_hidden_dim),
                nn.LayerNorm(self.tpp_hidden_dim),
                nn.SiLU(),
            )
            self.graph_condition_fusion = nn.Sequential(
                nn.Linear(self.tpp_hidden_dim * 4, fpl),
                nn.LayerNorm(fpl),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        
        # 保存层级权重用于监控
        self.layer_weights_history = None
        
        # Hard Concatenation: 图级 TPP 拼接投影
        if self.hard_concat_tpp:
            self.graph_tpp_projection = nn.Sequential(
                nn.Linear(fpl + PHARMACOPHORE_FEATURE_DIM, fpl),
                nn.LayerNorm(fpl),
                nn.ReLU(),
            )
        
        self.to(device)
    
    def _init_layers(self):
        gat_layers = nn.ModuleList()
        output_layers = nn.ModuleList()
        
        for idx, (in_dim, out_dim) in enumerate(self.throughShape):
            # 药效团引导的 GAT 层
            gat_layer = PharmacophoreGATConv(
                in_features=in_dim,
                out_features=out_dim,
                tpp_vector=self.tpp_vector[:PHARMACOPHORE_FEATURE_DIM],
                num_heads=self.num_heads,
                dropout=0.1,
                concat=False,  # 取平均，保持维度一致
                disable_tpp_bias=self.disable_tpp_bias,
                hard_concat_tpp=self.hard_concat_tpp,
                fixed_bias=self.fixed_bias
            )
            gat_layers.append(gat_layer)
            
            # 输出层 (用于生成指纹)
            output_layers.append(nfpOutput(in_dim, self.fpl))
        
        # 最后一层的输出
        output_layers.append(nfpOutput(self.layers[-1], self.fpl))
        
        return gat_layers, output_layers
    
    def forward(self, input):
        """
        前向传播
        
        Args:
            input: (atoms, bonds, edges) 元组
                - atoms: [batch, num_atoms, num_features]
                - bonds: [batch, num_atoms, max_degree, bond_features]
                - edges: [batch, num_atoms, max_degree]
        
        Returns:
            fingerprint: [batch, fpl] 分子指纹
        """
        self.atom_activations = []
        if len(input) == 4:
            a, b, e, target_tpp = input
        else:
            a, b, e = input
            target_tpp = self.tpp_vector
        a, b, e = a.to(device), b.to(device), e.to(device)
        tpp_batch = prepare_tpp_batch(
            target_tpp,
            batch_size=a.shape[0],
            dtype=a.dtype,
            target_device=a.device,
        )
        tpp_condition = (
            self.tpp_encoder(tpp_batch)
            if self.enable_tpp_conditioning
            else None
        )
        
        # 缓存原始输入特征 (用于 SP-CL 对比学习)
        self._cached_input_features = a

        # 在所有 PharmGAT 层中复用原始六维药效团二值特征。
        raw_pharm_features = a[..., -PHARMACOPHORE_FEATURE_DIM:].contiguous()
        
        # 收集各层的指纹贡献 (不再简单相加)
        layer_fingerprints = []
        
        for i, (gat_layer, output_layer, layer_norm) in enumerate(
            zip(self.gat_layers, self.output_layers[:-1], self.layer_norms)
        ):
            # 生成当前层的指纹贡献
            fp_contrib, atom_act = output_layer(a, b, e)
            layer_fingerprints.append(fp_contrib)
            
            if self.return_activations:
                self.atom_activations.append(atom_act)
            
            # GAT 卷积
            a_new = gat_layer(
                a, e, raw_pharm_features, tpp_batch
            )
            
            # 残差连接 (如果维度匹配)
            if a.shape[-1] == a_new.shape[-1]:
                a_new = a_new + a
            
            # 层归一化 + 激活
            a_new = layer_norm(a_new)
            a_new = F.elu(a_new)

            if self.enable_tpp_conditioning:
                a_new = self.film_layers[i](a_new, tpp_condition)
            
            # 图池化
            a = self.pool(a_new, e)
        
        # 缓存最终原子表示 (用于 SP-CL 对比学习)
        self._cached_atom_repr = a
        
        # 最终输出层的指纹
        fp_final, atom_act_final = self.final_output(a, b, e)
        layer_fingerprints.append(fp_final)
        
        if self.return_activations:
            self.atom_activations.append(atom_act_final)
        
        # ============================================================
        # 多尺度特征聚合 (替代简单相加)
        # ============================================================
        fingerprint, layer_weights = self.multi_scale_aggregator(layer_fingerprints)

        if self.enable_tpp_conditioning:
            molecule_condition = self.molecule_condition_projection(
                fingerprint
            )
            fused_condition = torch.cat([
                molecule_condition,
                tpp_condition,
                molecule_condition * tpp_condition,
                torch.abs(molecule_condition - tpp_condition),
            ], dim=-1)
            fingerprint = self.graph_condition_fusion(fused_condition)
        
        # Hard Concatenation: 将 TPP 拼接到图级表示
        if self.hard_concat_tpp:
            batch_size_fp = fingerprint.shape[0]
            tpp_graph = tpp_batch[
                ..., :PHARMACOPHORE_FEATURE_DIM
            ].reshape(batch_size_fp, PHARMACOPHORE_FEATURE_DIM)
            fp_with_tpp = torch.cat([fingerprint, tpp_graph], dim=-1)
            fingerprint = self.graph_tpp_projection(fp_with_tpp)
        
        # 保存权重用于监控
        self.layer_weights_history = layer_weights.detach()
        
        return fingerprint
    
    def get_layer_weights(self):
        """获取各层指纹的聚合权重 (用于可解释性分析)"""
        if self.layer_weights_history is not None:
            # 返回平均权重
            return self.layer_weights_history.mean(dim=0).cpu().numpy()
        
        # 如果使用 'learned' 类型，直接返回可学习权重
        if self.aggregation_type == 'learned':
            weights = F.softmax(self.multi_scale_aggregator.layer_weights, dim=0)
            return weights.detach().cpu().numpy()
        
        return None


# ============================================================
# 创新: 子图药效团对比学习正则化 (SP-CL)
# Subgraph Pharmacophore Contrastive Learning Regularization
# ============================================================
class SubgraphPharmCL(nn.Module):
    """
    子图药效团对比学习正则化模块
    
    核心创新: 将原子级药效团匹配提升到功能团/子图级，通过对比学习
    辅助损失增强 GNN 的分子表示学习能力。
    
    工作流程:
    1. 从原子特征最后6位提取药效团类型标签
    2. 对每种药效团类型，注意力池化属于该类型的原子 GNN 表示
       得到 motif embedding (子图级别)
    3. 与6个可学习的 "药效团类型原型" 计算相似度
    4. InfoNCE 对比损失:
       - 拉近: 匹配 TPP 重要特征的 motif ↔ 对应原型
       - 推远: motif ↔ 非对应原型
    
    注意力公式:
        motif_i = Σ_j (α_j * h_j)   其中 j ∈ {原子属于药效团类型 i}
        α_j = softmax(query_i^T · h_j)
    
    对比损失:
        L_CL = -log( exp(sim(motif_i, proto_i)/τ) / Σ_k exp(sim(motif_i, proto_k)/τ) )
    """
    
    def __init__(self, hidden_dim, tpp_vector, proj_dim=64, temperature=0.07):
        """
        Args:
            hidden_dim: GNN 输出的原子特征维度
            tpp_vector: Target Pharmacophore Profile 向量 [6,]
            proj_dim: 对比学习投影空间维度
            temperature: InfoNCE 温度参数 (越小越尖锐)
        """
        super(SubgraphPharmCL, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.temperature = temperature
        
        # 注册 TPP 为 buffer
        self.register_buffer('tpp_vector', tpp_vector)
        
        # 投影头: 将 GNN 原子表示映射到低维对比学习空间
        # 两层 MLP + ReLU (SimCLR 风格)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )
        
        # 6 个可学习的药效团类型原型向量
        # 每个原型代表一种理想的药效团子图表示
        self.prototypes = nn.Parameter(torch.randn(PHARMACOPHORE_FEATURE_DIM, proj_dim))
        nn.init.xavier_uniform_(self.prototypes)
        
        # 注意力池化的查询向量 (每种药效团类型一个)
        # 用于加权聚合属于该类型的原子表示
        self.attention_query = nn.Parameter(torch.randn(PHARMACOPHORE_FEATURE_DIM, hidden_dim))
        nn.init.xavier_uniform_(self.attention_query)
        
        self.to(device)
    
    def compute_motif_embeddings(self, atom_representations, atom_features):
        """
        提取每种药效团类型的 motif embedding
        
        Args:
            atom_representations: [batch, num_atoms, hidden_dim] GNN 最终层输出
            atom_features: [batch, num_atoms, in_features] 原始原子特征
        
        Returns:
            motif_embeddings: [batch, 6, proj_dim] 6种药效团类型的 motif embedding
            motif_masks: [batch, 6] 哪些药效团类型在该分子中有对应原子
        """
        batch_size, num_atoms, _ = atom_representations.shape
        
        # 提取药效团标签 [batch, num_atoms, 6] (原子特征最后6位)
        pharm_labels = atom_features[..., -PHARMACOPHORE_FEATURE_DIM:]
        
        # 原子有效掩码 (排除 padding 原子, padding 原子特征全为 0)
        atom_mask = atom_features.abs().sum(-1) > 0  # [batch, num_atoms]
        
        motif_embeddings = []
        motif_masks = []
        
        for p in range(PHARMACOPHORE_FEATURE_DIM):
            # 属于该药效团类型的原子掩码
            type_mask = (pharm_labels[..., p] > 0.5) & atom_mask  # [batch, num_atoms]
            
            # 该类型是否在分子中存在
            has_type = type_mask.any(dim=1)  # [batch]
            motif_masks.append(has_type)
            
            # 注意力池化: query_p^T · h_j
            query = self.attention_query[p]  # [hidden_dim]
            attn_scores = (atom_representations * query.unsqueeze(0).unsqueeze(0)).sum(-1)
            # attn_scores: [batch, num_atoms]
            
            # 仅在属于该类型的原子上做 softmax
            attn_scores = attn_scores.masked_fill(~type_mask, float('-inf'))
            attn_weights = F.softmax(attn_scores, dim=1)  # [batch, num_atoms]
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            
            # 加权池化得到 motif 表示
            pooled = (atom_representations * attn_weights.unsqueeze(-1)).sum(dim=1)
            # pooled: [batch, hidden_dim]
            
            # 投影到对比学习空间
            projected = self.projector(pooled)  # [batch, proj_dim]
            motif_embeddings.append(projected)
        
        motif_embeddings = torch.stack(motif_embeddings, dim=1)  # [batch, 6, proj_dim]
        motif_masks = torch.stack(motif_masks, dim=1)  # [batch, 6]
        
        return motif_embeddings, motif_masks
    
    def forward(self, atom_representations, atom_features):
        """
        计算子图药效团对比损失
        
        Args:
            atom_representations: [batch, num_atoms, hidden_dim] GNN 最终层输出
            atom_features: [batch, num_atoms, in_features] 原始原子特征
        
        Returns:
            loss: scalar, InfoNCE 对比损失值
        """
        motif_embeddings, motif_masks = self.compute_motif_embeddings(
            atom_representations, atom_features
        )
        # motif_embeddings: [batch, 6, proj_dim]
        # motif_masks: [batch, 6] 该分子是否含有对应药效团类型
        
        batch_size = motif_embeddings.shape[0]
        
        # L2 归一化 (对比学习标准做法)
        motif_normed = F.normalize(motif_embeddings, dim=-1)
        proto_normed = F.normalize(self.prototypes, dim=-1)  # [6, proj_dim]
        
        # 相似度矩阵: [batch, 6, 6]
        # sim[b, i, j] = cosine_sim(motif_i of molecule b, prototype_j) / τ
        sim_matrix = torch.matmul(motif_normed, proto_normed.T) / self.temperature
        
        # TPP 重要特征掩码: 只对 TPP 标记为重要的药效团类型计算对比损失
        tpp_important = (self.tpp_vector > 0.5)  # [6]
        
        # 有效掩码: 药效团类型存在于分子中 AND TPP认为该类型重要
        valid_mask = motif_masks & tpp_important.unsqueeze(0)  # [batch, 6]
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=atom_representations.device, requires_grad=True)
        
        # InfoNCE: 对角线为正样本, 其余为负样本
        # 对于有效的 (molecule, pharm_type_i) 对:
        #   L = -log(exp(sim(motif_i, proto_i)) / Σ_j exp(sim(motif_i, proto_j)))
        # 这等价于 cross_entropy(sim[b, i, :], target=i)
        labels = torch.arange(PHARMACOPHORE_FEATURE_DIM, device=sim_matrix.device)
        labels = labels.unsqueeze(0).expand(batch_size, -1)  # [batch, 6]
        
        # 展平然后只取有效位置
        sim_flat = sim_matrix.reshape(-1, PHARMACOPHORE_FEATURE_DIM)  # [batch*6, 6]
        labels_flat = labels.reshape(-1)  # [batch*6]
        valid_flat = valid_mask.reshape(-1)  # [batch*6]
        
        # 只对有效位置计算交叉熵损失
        loss = F.cross_entropy(
            sim_flat[valid_flat], 
            labels_flat[valid_flat], 
            reduction='mean'
        )
        
        return loss


# ============================================================
# 原有的 nfpDocking (保留用于对比实验)
# ============================================================
class nfpDocking(nn.Module):
    """原有的 NFP Docking 网络 (保留用于对比实验)"""
    
    def __init__(self, layers, return_activations, fpl=32, hf=20):
        super(nfpDocking, self).__init__()
        self.layers = layers
        self.fpl = fpl
        self.throughShape = list(zip(layers[:-1], layers[1:]))
        self.layersArr, self.outputArr = self.init_layers()
        self.op = nfpOutput(self.layers[-1], self.fpl)
        self.pool = GraphPool()
        self.return_activations = return_activations
        self.atom_activations = None
        if self.return_activations:
            self.atom_activations = []
        self.to(device)

    def init_layers(self):
        layersArr, outputArr = [], []
        for idx, (i, o) in enumerate(self.throughShape):
            outputArr.append(nfpOutput(self.layers[idx], self.fpl))
            layersArr.append(nfpConv(i, o))
        outputArr.append(nfpOutput(self.layers[-1], self.fpl))
        return nn.ModuleList(layersArr), nn.ModuleList(outputArr)
    
    def forward(self, input):
        self.atom_activations = []
        a, b, e = input
        a, b, e = a.to(device), b.to(device), e.to(device)
        ffp = torch.zeros(a.shape[0], self.fpl).to(device)
        for i in range(len(self.layers[1:])):
            lfp, aact = self.outputArr[i](a, b, e)
            ffp += lfp
            if self.return_activations: 
                self.atom_activations.append(aact)
            a = self.layersArr[i]((a, b, e))
            a = self.pool(a, e)
        ffp, aact = self.op(a, b, e)
        if self.return_activations: 
            self.atom_activations.append(aact)
        return ffp


# ============================================================
# 全连接分类网络 (保持不变)
# ============================================================
class dockingANN(nn.Module):
    def __init__(self, fpl, ba, layers, dropout):
        super(dockingANN, self).__init__()
        self.inputSize = fpl
        self.ba = ba
        self.arch = list(zip(ba[:-1], ba[1:]))
        self.layers = layers
        self.dropout = dropout

        self.ann = nn.Sequential()
        self.buildModel()
        self.to(device)

    def buildModel(self):
        for j, (i, o) in enumerate(self.arch):
            self.ann.add_module(f'linear {j}', nn.Linear(i, o))
            self.ann[-1].bias = torch.nn.init.constant_(torch.nn.Parameter(torch.empty(o, device=device)), 0.01)
            if o != 1:
                self.ann.add_module(f'batch norm {j}', nn.BatchNorm1d(o))
                self.ann.add_module(f'relu act {j}', nn.ReLU())
                self.ann.add_module(f'dropout {j}', nn.Dropout(self.dropout))

    def forward(self, input):
        return self.ann(input)


# ============================================================
# 主模型: 药效团引导的对接协议 (PharmacophoreGuidedDocking)
# ============================================================
class dockingProtocol(nn.Module):
    """
    药效团引导的分子对接预测模型
    
    结构:
        PharmacophoreGAT / DualChannelPharmGAT (图卷积) -> dockingANN (全连接分类)
    
    创新点:
        - 使用 PharmacophoreGATConv 替代原有卷积层
        - 注意力机制中引入药效团匹配偏置
        - β 参数可学习，自动调节"老师信号"的权重
        - [NEW] 双通道异构消息传递: 药效团 GAT + 拓扑 GIN + 跨通道注意力
    """
    
    def __init__(self, params, tpp_path=None):
        """
        Args:
            params: 模型参数字典
            tpp_path: TPP 文件路径 (如 'cdk2/cdk2_tpp.pt')
        """
        super(dockingProtocol, self).__init__()
        
        self.tpp_input_mode = params.get("tpp_input_mode", "binary")
        self.tpp_count_clip = float(params.get("tpp_count_clip", 20.0))
        self.register_buffer(
            'tpp_vector',
            load_tpp_vector(
                tpp_path,
                mode=self.tpp_input_mode,
                count_clip=self.tpp_count_clip,
            ),
        )
        
        # 是否使用药效团引导的 GAT
        self.use_pharmacophore_gat = params.get("use_pharmacophore_gat", True)
        
        # 是否使用双通道异构消息传递
        self.use_dual_channel = params.get("use_dual_channel", False)
        
        # 多尺度聚合类型: 'learned', 'attention', 'gated', 'hierarchical'
        self.aggregation_type = params.get("aggregation_type", "attention")
        
        # ============================================================
        # SP-CL: 子图药效团对比学习正则化
        # ============================================================
        self.enable_spcl = params.get("enable_spcl", False)
        self.spcl_weight = params.get("spcl_weight", 0.1)  # 对比损失权重 α
        
        # 消融实验: 是否禁用 TPP Bias
        self.disable_tpp_bias = params.get("disable_tpp_bias", False)
        
        # 消融实验: Hard Concatenation (TPP 静态拼接)
        self.hard_concat_tpp = params.get("hard_concat_tpp", False)
        
        # 消融实验: Fixed Bias (β 固定为常数，不参与训练)
        self.fixed_bias = params.get("fixed_bias", False)

        self.enable_tpp_conditioning = params.get(
            "enable_tpp_conditioning", False
        )
        self.tpp_hidden_dim = int(params.get("tpp_hidden_dim", 64))
        self.tpp_film_scale = float(params.get("tpp_film_scale", 0.1))
        if self.enable_tpp_conditioning and self.hard_concat_tpp:
            raise ValueError(
                "enable_tpp_conditioning and hard_concat_tpp are separate "
                "experimental variants and cannot be enabled together"
            )
        
        if self.use_dual_channel:
            # ===== 双通道异构消息传递 (创新点 1) =====
            gnn = DualChannelPharmGAT(
                layers=params["conv"]["layers"],
                tpp_vector=self.tpp_vector,
                return_activations=params["conv"]["activations"],
                fpl=params["fpl"],
                num_heads=params.get("num_heads", 4),
                dropout=params["ann"]["dropout"],
                aggregation_type=self.aggregation_type,
                disable_tpp_bias=self.disable_tpp_bias,
                hard_concat_tpp=self.hard_concat_tpp,
                fixed_bias=self.fixed_bias,
                enable_tpp_conditioning=self.enable_tpp_conditioning,
                tpp_hidden_dim=self.tpp_hidden_dim,
                tpp_film_scale=self.tpp_film_scale,
            )
            if self.hard_concat_tpp:
                ablation_tag = " [ABLATION: Hard Concatenation]"
            elif self.disable_tpp_bias:
                ablation_tag = " [ABLATION: TPP Bias DISABLED]"
            elif self.fixed_bias:
                ablation_tag = " [ABLATION: Fixed Bias (β frozen)]"
            else:
                ablation_tag = ""
            if self.enable_tpp_conditioning:
                ablation_tag += " [TPP: FiLM + graph fusion]"
            print(f"[DC-PharmGAT] Dual-channel enabled{ablation_tag}")
        elif self.use_pharmacophore_gat:
            # 使用原有的 PharmacophoreGAT (带多尺度聚合)
            gnn = PharmacophoreGAT(
                layers=params["conv"]["layers"],
                tpp_vector=self.tpp_vector,
                return_activations=params["conv"]["activations"],
                fpl=params["fpl"],
                num_heads=params.get("num_heads", 4),
                dropout=params["ann"]["dropout"],
                aggregation_type=self.aggregation_type,
                disable_tpp_bias=self.disable_tpp_bias,
                hard_concat_tpp=self.hard_concat_tpp,
                fixed_bias=self.fixed_bias,
                enable_tpp_conditioning=self.enable_tpp_conditioning,
                tpp_hidden_dim=self.tpp_hidden_dim,
                tpp_film_scale=self.tpp_film_scale,
            )
        else:
            # 使用原有的 nfpDocking (用于对比实验)
            gnn = nfpDocking(
                layers=params["conv"]["layers"],
                fpl=params["fpl"],
                return_activations=params["conv"]["activations"]
            )
        
        self.model = nn.Sequential(
            gnn,
            dockingANN(
                fpl=params["fpl"],
                ba=params["ann"]["ba"],
                dropout=params["ann"]["dropout"],
                layers=params["ann"]["layers"]
            )
        )
        
        # 初始化 SP-CL 模块 (仅在启用时)
        if self.enable_spcl and (self.use_pharmacophore_gat or self.use_dual_channel):
            hidden_dim = params["conv"]["layers"][-1]  # GNN 最终层维度
            self.spcl_module = SubgraphPharmCL(
                hidden_dim=hidden_dim,
                tpp_vector=self.tpp_vector[:PHARMACOPHORE_FEATURE_DIM],
                proj_dim=params.get("spcl_proj_dim", 64),
                temperature=params.get("spcl_temperature", 0.07),
            )
            print(f"[SP-CL] 子图药效团对比学习已启用 (weight={self.spcl_weight})")
        else:
            self.spcl_module = None
        
        self.return_activations = params["conv"]["activations"]
        self.to(device)

    @torch.no_grad()
    def set_tpp_vector(self, new_tpp):
        """Replace the fallback TPP used for inference-only ablations."""
        new_tpp = torch.as_tensor(
            new_tpp, dtype=torch.float32, device=device
        ).reshape(-1)
        expected = self.tpp_vector.numel()
        if new_tpp.numel() != expected:
            raise ValueError(
                f"TPP must contain {expected} values for mode "
                f"{self.tpp_input_mode}, got {new_tpp.numel()}"
            )

        for module in self.modules():
            if "tpp_vector" not in module._buffers:
                continue
            target = module._buffers["tpp_vector"]
            if target.numel() == new_tpp.numel():
                source = new_tpp
            elif target.numel() == PHARMACOPHORE_FEATURE_DIM:
                source = new_tpp[:PHARMACOPHORE_FEATURE_DIM]
            else:
                raise RuntimeError(
                    f"Cannot map TPP of length {new_tpp.numel()} to "
                    f"buffer of length {target.numel()}"
                )
            target.copy_(source.to(
                device=target.device, dtype=target.dtype
            ))
    
    def get_beta_values(self):
        """获取所有 GAT 层的 β 值 (用于监控训练)"""
        beta_info = []
        gnn = self.model[0]
        
        # 确定 GAT 层列表
        gat_layers = None
        if self.use_dual_channel and hasattr(gnn, 'pharm_gat_layers'):
            gat_layers = gnn.pharm_gat_layers
        elif self.use_pharmacophore_gat and hasattr(gnn, 'gat_layers'):
            gat_layers = gnn.gat_layers
        
        if gat_layers is not None:
            for i, layer in enumerate(gat_layers):
                layer_betas = {}
                if hasattr(layer, 'beta_edge'):
                    layer_betas['edge'] = layer.beta_edge.item()
                if hasattr(layer, 'beta_interact'):
                    layer_betas['interact'] = layer.beta_interact.item()
                # 兼容旧版本的 beta 参数
                if hasattr(layer, 'beta') and not layer_betas:
                    layer_betas['beta'] = layer.beta.item()
                if layer_betas:
                    beta_info.append(layer_betas)
        return beta_info
    
    def get_pharm_importance(self):
        """获取药效团特征重要性权重 (用于可解释性分析)"""
        importance_info = []
        gnn = self.model[0]
        
        gat_layers = None
        if self.use_dual_channel and hasattr(gnn, 'pharm_gat_layers'):
            gat_layers = gnn.pharm_gat_layers
        elif self.use_pharmacophore_gat and hasattr(gnn, 'gat_layers'):
            gat_layers = gnn.gat_layers
        
        if gat_layers is not None:
            for i, layer in enumerate(gat_layers):
                if hasattr(layer, 'pharm_importance'):
                    weights = F.softmax(layer.pharm_importance, dim=0).detach().cpu().numpy()
                    importance_info.append({
                        'layer': i,
                        'donor': weights[0],
                        'acceptor': weights[1],
                        'aromatic': weights[2],
                        'positive': weights[3],
                        'negative': weights[4],
                        'hydrophobic': weights[5],
                    })
        return importance_info
    
    def get_layer_weights(self):
        """获取多尺度聚合的层级权重 (用于可解释性分析)"""
        gnn = self.model[0]
        if hasattr(gnn, 'get_layer_weights'):
            return gnn.get_layer_weights()
        return None
    
    def get_cross_attn_alphas(self):
        """获取跨通道注意力融合强度 α (用于监控双通道训练)"""
        if self.use_dual_channel:
            gnn = self.model[0]
            if hasattr(gnn, 'get_cross_attn_alphas'):
                return gnn.get_cross_attn_alphas()
        return None
    
    def compute_contrastive_loss(self, input):
        """
        计算 SP-CL 子图药效团对比损失
        
        注意: 必须在 forward() 之后调用，因为需要缓存的中间表示。
        
        Args:
            input: (atoms, bonds, edges) 元组 (与 forward 相同的输入)
        
        Returns:
            cl_loss: scalar, 对比损失值 (已乘以 spcl_weight)
                     如果 SP-CL 未启用则返回 0
        """
        if self.spcl_module is None:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        gnn = self.model[0]  # PharmacophoreGAT 或 DualChannelPharmGAT
        
        # 从缓存获取 GNN 最终层原子表示和原始输入特征
        if not hasattr(gnn, '_cached_atom_repr') or gnn._cached_atom_repr is None:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        atom_repr = gnn._cached_atom_repr      # [batch, num_atoms, hidden_dim]
        atom_features = gnn._cached_input_features  # [batch, num_atoms, in_features]
        
        # 计算对比损失
        cl_loss = self.spcl_module(atom_repr, atom_features)
        
        return self.spcl_weight * cl_loss

    def forward(self, input):
        if self.return_activations:
            return torch.squeeze(self.model(input)), self.model[0].atom_activations
        return torch.squeeze(self.model(input))


# ============================================================
# 数据集类 (保持不变)
# ============================================================
class dockingDataset(Dataset):
    def __init__(self, train, labels, maxa=200, maxd=6, name='unknown'):
        self.train = train
        self.labels = torch.from_numpy(np.array(labels)).float()
        self.maxA = maxa
        self.maxD = maxd
        self.a, self.b, self.e = build_molecular_graph_tensors(
            [x[1] for x in self.train],
            max_neighbors=self.maxD,
            max_atoms=self.maxA,
            dataset_name=name,
        )

    def __len__(self):
        return self.a.shape[0]

    def __getitem__(self, idx):
        return self.a[idx], self.b[idx], self.e[idx], (self.labels[idx], self.train[idx][0])
