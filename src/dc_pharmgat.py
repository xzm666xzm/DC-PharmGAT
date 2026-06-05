"""
dc_pharmgat.py - DC-PharmGAT model architecture.

core innovation point: 
- PharmacophoreGATConv: introduce pharmacophore-matching bias when computing attention coefficients
- attention formula: α_ij = LeakyReLU(a^T [Wh_i || Wh_j]) + β * MatchScore(h_i, TPP)
- β text learnable parameter, the model automatically learns"how strongly to follow the prior"
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
# pharmacophore feature configuration
# ============================================================
# position of pharmacophore features in the atom feature vector (text last six dimensions)
PHARMACOPHORE_FEATURE_DIM = 6
PHARMACOPHORE_FEATURE_NAMES = [
    'is_h_bond_donor',
    'is_h_bond_acceptor',
    'is_aromatic',
    'is_positive',
    'is_negative',
    'is_hydrophobic',
]


def load_tpp_vector(tpp_path):
    """
    load Target Pharmacophore Profile (TPP) vector
    
    Args:
        tpp_path:.pt file path
    
    Returns:
        tpp_tensor: pharmacophore Boolean feature vector [6,] (has_donor, has_acceptor,...)
    """
    if tpp_path is None or not os.path.exists(tpp_path):
        print(f"Warning: TPP file not found at {tpp_path}, using default TPP")
        # details TPP: assume all pharmacophore features are important
        return torch.ones(PHARMACOPHORE_FEATURE_DIM, device=device)
    
    tpp_data = torch.load(tpp_path, map_location=device)
    tpp_dict = tpp_data.get('tpp_dict', {})
    
    # extract Boolean features (has_xxx)
    tpp_bool = torch.tensor([
        float(tpp_dict.get('has_hbond_donor', 1)),
        float(tpp_dict.get('has_hbond_acceptor', 1)),
        float(tpp_dict.get('has_aromatic', 1)),
        float(tpp_dict.get('has_positive_charge', 0)),
        float(tpp_dict.get('has_negative_charge', 0)),
        float(tpp_dict.get('has_hydrophobic', 1)),
], device=device, dtype=torch.float32)
    
    print(f"Loaded TPP from {tpp_path}: {tpp_bool.tolist()}")
    return tpp_bool


# ============================================================
# pharmacophore-guided graph attention convolution layer (PharmacophoreGATConv)
# ============================================================
class PharmacophoreGATConv(nn.Module):
    """
    pharmacophore-guided graph attention convolution layer (enhanced version)
    
    innovation: information GAT attention calculate intermediate introduce details other of pharmacophore summary
    
    attention formula:
        α_ij = softmax(LeakyReLU(a^T [Wh_i || Wh_j]) + β_node * MatchScore(h_i) 
                       + β_edge * EdgePharmScore(h_i, h_j) + β_interact * InteractScore(h_i, h_j))
    
    where:
        - W: feature transformation matrix
        - a: attention vector
        - β_node: text point text pharmacophore summary (learnable)
        - β_edge: details pharmacophore summary (learnable)
        - β_interact: pharmacophore details (learnable)
        - MatchScore: atomic pharmacophore features and TPP of information
        - EdgePharmScore: details atom of pharmacophore information
        - InteractScore: source-target atom of pharmacophore details
    """
    
    def __init__(self, in_features, out_features, tpp_vector, num_heads=1, 
                 negative_slope=0.2, dropout=0.0, concat=True, disable_tpp_bias=False, hard_concat_tpp=False, fixed_bias=False):
        """
        Args:
            in_features: details features dimension
            out_features: details features dimension
            tpp_vector: Target Pharmacophore Profile vector [6,]
            num_heads: attention data
            negative_slope: LeakyReLU negative rate
            dropout: Dropout compare rate
            concat: details details (True) get details (False)
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
        
        # text copy: hard_concat_tpp mode text disable summary
        if self.hard_concat_tpp:
            self.disable_tpp_bias = True
        
        # attention TPP as buffer (details and details update, information model save/load)
        self.register_buffer('tpp_vector', tpp_vector)
        
        # feature transformation matrix W: [in_features, num_heads * out_features]
        self.W = nn.Parameter(torch.empty(in_features, num_heads * out_features))
        nn.init.xavier_uniform_(self.W)
        
        # attention vector a: [num_heads, 2 * out_features]
        # use calculate a^T [Wh_i || Wh_j]
        self.a_src = nn.Parameter(torch.empty(num_heads, out_features))
        self.a_dst = nn.Parameter(torch.empty(num_heads, out_features))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        
        # ============================================================
        # details pharmacophore details argument (core innovation)
        # ============================================================
        if self.fixed_bias:
            # Fixed Bias details: β fixed as constants, do not participate in training
            self.register_buffer('beta_node', torch.tensor(0.3))
            self.register_buffer('beta_edge', torch.tensor(0.3))
            self.register_buffer('beta_interact', torch.tensor(0.4))
        else:
            # β_node: text point information - before atom details TPP of details
            self.beta_node = nn.Parameter(torch.tensor(0.3))
            
            # β_edge: summary - details atom details TPP of details
            self.beta_edge = nn.Parameter(torch.tensor(0.3))
            
            # β_interact: summary - source-target atom of pharmacophore details effect text
            # result information atom information TPP of needed features, text update high of attention
            self.beta_interact = nn.Parameter(torch.tensor(0.4))
        
        # ============================================================
        # pharmacophore features summary (details pharmacophore details of summary)
        # ============================================================
        # details: source atomic pharmacophore features [6] + target atomic pharmacophore features [6] = [12]
        # details: split data [num_heads]
        self.pharm_interact_net = nn.Sequential(
            nn.Linear(PHARMACOPHORE_FEATURE_DIM * 2, 16),
            nn.ReLU(),
            nn.Linear(16, num_heads),
)
        
        # pharmacophore features needed property details (summary pharmacophore features update needed)
        # 6 pharmacophore features: donor, acceptor, aromatic, positive, negative, hydrophobic
        self.pharm_importance = nn.Parameter(torch.ones(PHARMACOPHORE_FEATURE_DIM))
        
        # details
        self.bias = nn.Parameter(torch.zeros(num_heads * out_features if concat else out_features))
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.feat_dropout = nn.Dropout(dropout)
        
        # Hard Concatenation: TPP static concatenation information (text point text)
        if self.hard_concat_tpp:
            self.tpp_node_projection = nn.Sequential(
                nn.Linear(out_features + PHARMACOPHORE_FEATURE_DIM, out_features),
                nn.ReLU(),
)
        
        self.to(device)
    
    def compute_pharmacophore_match_score(self, atom_features):
        """
        calculation atomic pharmacophore features and TPP of details matching score
        
        Args:
            atom_features: [batch, num_atoms, in_features]
        
        Returns:
            match_scores: [batch, num_atoms] matching score (0-1)
        """
        # extract atom of pharmacophore features (after 6 text)
        pharm_features = atom_features[..., -PHARMACOPHORE_FEATURE_DIM:]  # [batch, num_atoms, 6]
        
        # calculation and TPP of details
        # TPP: [6,] -> [1, 1, 6]
        tpp = self.tpp_vector.view(1, 1, -1)
        
        # use learnable of needed property details
        importance = F.softmax(self.pharm_importance, dim=0).view(1, 1, -1)  # [1, 1, 6]
        
        # details point information: atom have features AND TPP needed need features, information need property details
        match = pharm_features * tpp * importance  # [batch, num_atoms, 6]
        
        # text single matching score
        tpp_weighted_sum = (tpp * importance).sum() + 1e-8
        match_scores = match.sum(dim=-1) / tpp_weighted_sum  # [batch, num_atoms]
        
        return match_scores
    
    def compute_edge_pharmacophore_interaction(self, src_pharm, dst_pharm):
        """
        calculation details other of pharmacophore split data
        
        Args:
            src_pharm: [batch, num_atoms, max_degree, 6] source atomic pharmacophore features
            dst_pharm: [batch, num_atoms, max_degree, 6] target atomic pharmacophore features
        
        Returns:
            interact_scores: [batch, num_atoms, max_degree, num_heads] split data
        """
        batch_size, num_atoms, max_degree, _ = src_pharm.shape
        
        # details source and target of pharmacophore features
        combined = torch.cat([src_pharm, dst_pharm], dim=-1)  # [batch, num_atoms, max_degree, 12]
        
        # details calculation split data
        # need reshape information Linear text
        combined_flat = combined.view(-1, PHARMACOPHORE_FEATURE_DIM * 2)
        interact_flat = self.pharm_interact_net(combined_flat)  # [batch*num_atoms*max_degree, num_heads]
        interact_scores = interact_flat.view(batch_size, num_atoms, max_degree, self.num_heads)
        
        return interact_scores
    
    def forward(self, atoms, edges):
        """
        before information
        
        Args:
            atoms: [batch, num_atoms, in_features] atom features
            edges: [batch, num_atoms, max_degree] details (-1 table text unable to)
        
        Returns:
            new_atoms: [batch, num_atoms, out_features * num_heads] text [batch, num_atoms, out_features]
        """
        batch_size, num_atoms, _ = atoms.shape
        _, _, max_degree = edges.shape
        
        atoms = atoms.to(device)
        edges = edges.to(device)
        
        # 1. features details: [batch, num_atoms, num_heads * out_features]
        h = torch.matmul(atoms, self.W)
        h = h.view(batch_size, num_atoms, self.num_heads, self.out_features)
        # h: [batch, num_atoms, num_heads, out_features]
        
        # Hard Concatenation: will TPP static concatenation information after of node features
        if self.hard_concat_tpp:
            tpp_expanded = self.tpp_vector.view(1, 1, 1, -1).expand(
                batch_size, num_atoms, self.num_heads, -1
)  # [batch, num_atoms, num_heads, 6]
            h_concat = torch.cat([h, tpp_expanded], dim=-1)  # [..., out_features + 6]
            h = self.tpp_node_projection(h_concat)  # [..., out_features]
        
        # 2. extract pharmacophore features
        pharm_features = atoms[..., -PHARMACOPHORE_FEATURE_DIM:]  # [batch, num_atoms, 6]
        
        # 3. calculation text point text pharmacophore matching score
        pharm_scores = self.compute_pharmacophore_match_score(atoms)  # [batch, num_atoms]
        
        # 4. calculation details GAT attention split data
        # source text point attention: a_src^T * h_i
        attn_src = (h * self.a_src.view(1, 1, self.num_heads, self.out_features)).sum(dim=-1)
        # attn_src: [batch, num_atoms, num_heads]
        
        # target text point attention: a_dst^T * h_j
        attn_dst = (h * self.a_dst.view(1, 1, self.num_heads, self.out_features)).sum(dim=-1)
        # attn_dst: [batch, num_atoms, num_heads]
        
        # 5. details
        # create have effect information
        edge_mask = (edges!= -1).float()  # [batch, num_atoms, max_degree]
        
        # will -1 details as 0 use information (after details use mask details)
        safe_edges = edges.clamp(min=0)  # [batch, num_atoms, max_degree]
        
        # get details features
        batch_idx = torch.arange(batch_size, device=device).view(-1, 1, 1).expand(-1, num_atoms, max_degree)
        neighbor_h = h[batch_idx, safe_edges]  # [batch, num_atoms, max_degree, num_heads, out_features]
        
        # get details of attention split data (target text point)
        neighbor_attn_dst = attn_dst[batch_idx, safe_edges]  # [batch, num_atoms, max_degree, num_heads]
        
        # get details of pharmacophore matching score
        neighbor_pharm_scores = pharm_scores[batch_idx, safe_edges]  # [batch, num_atoms, max_degree]
        
        # get details of pharmacophore features
        neighbor_pharm_features = pharm_features[batch_idx, safe_edges]  # [batch, num_atoms, max_degree, 6]
        
        # 6. calculation details of attention data (core innovation text split)
        # ============================================================
        
        # 6.1 details GAT attention
        attn_src_expanded = attn_src.unsqueeze(2).expand(-1, -1, max_degree, -1)
        e = F.leaky_relu(attn_src_expanded + neighbor_attn_dst, negative_slope=self.negative_slope)
        # e: [batch, num_atoms, max_degree, num_heads]
        
        # 6.2 ~ 6.5 pharmacophore details (ablation experiment: disable_tpp_bias=True information)
        if not self.disable_tpp_bias:
            # 6.2 text point text pharmacophore details (source text point details TPP)
            node_bias = self.beta_node * pharm_scores.unsqueeze(-1).unsqueeze(-1)
            node_bias = node_bias.expand(-1, -1, max_degree, self.num_heads)
            # node_bias: [batch, num_atoms, max_degree, num_heads]
            
            # 6.3 details pharmacophore details (information point details TPP)
            edge_bias = self.beta_edge * neighbor_pharm_scores.unsqueeze(-1)
            edge_bias = edge_bias.expand(-1, -1, -1, self.num_heads)
            # edge_bias: [batch, num_atoms, max_degree, num_heads]
            
            # 6.4 information pharmacophore details (source-target details effect text)
            # prepare source atom of pharmacophore features (information max_degree dimension)
            src_pharm_expanded = pharm_features.unsqueeze(2).expand(-1, -1, max_degree, -1)
            # src_pharm_expanded: [batch, num_atoms, max_degree, 6]
            
            # calculation split data
            interact_scores = self.compute_edge_pharmacophore_interaction(
                src_pharm_expanded, neighbor_pharm_features
)  # [batch, num_atoms, max_degree, num_heads]
            
            interact_bias = self.beta_interact * interact_scores
            
            # 6.5 information have details
            e = e + node_bias + edge_bias + interact_bias
        
        # ============================================================
        
        # 7. Softmax text single (details have effect information)
        e = e.masked_fill(edge_mask.unsqueeze(-1) == 0, float('-inf'))
        
        # Softmax
        alpha = F.softmax(e, dim=2)  # [batch, num_atoms, max_degree, num_heads]
        
        # process details -inf of details (information point)
        alpha = torch.nan_to_num(alpha, nan=0.0)
        
        # Dropout
        alpha = self.attn_dropout(alpha)
        
        # 8. summary features
        alpha_expanded = alpha.unsqueeze(-1)  # [batch, num_atoms, max_degree, num_heads, 1]
        
        # information and
        out = (alpha_expanded * neighbor_h).sum(dim=2)  # [batch, num_atoms, num_heads, out_features]
        
        # 9. details process
        if self.concat:
            out = out.view(batch_size, num_atoms, -1)
        else:
            out = out.mean(dim=2)
        
        out = out + self.bias
        
        return out


# ============================================================
# has of details module (save information property)
# ============================================================
class GraphLookup(nn.Module):
    def __init__(self):
        super().__init__()
        self.to(device)

    def temporal_padding(self, x, paddings=(1, 0), pad_value=0):
        if not isinstance(paddings, (tuple, list, np.ndarray)):
            paddings = (paddings, paddings)
        output = torch.zeros(x.size(0), x.size(1) + sum(paddings), x.size(2), device=device)
        output[:,:paddings[0],:] = pad_value
        output[:, paddings[1]:,:] = pad_value
        output[:, paddings[0]: paddings[0]+x.size(1),:] = x
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


class GraphMessagePassingLayer(nn.Module):
    """Graph message-passing layer used by the molecular encoder."""
    def __init__(self, ishape, oshape):
        super(GraphMessagePassingLayer, self).__init__()
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
        atom_degrees = (edges!= -1).sum(-1, keepdim=True)
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


class FingerprintReadoutLayer(nn.Module):
    def __init__(self, layer, fpl):
        super(FingerprintReadoutLayer, self).__init__()
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
        atom_degrees = (e!= -1).sum(-1, keepdim=True)
        general_atom_mask = (atom_degrees!= 0).float()
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
        atom_degrees = (edges!= -1).sum(dim=-1, keepdim=True)
        general_atom_mask = (atom_degrees!= 0).float()
        return max_features * general_atom_mask


# ============================================================
# information features details module (Multi-Scale Feature Aggregation)
# ============================================================
class MultiScaleAggregator(nn.Module):
    """
    information features details module
    
    innovation: 
    1. learnable of summary - details details of needed property
    2. attention details - base details details
    3. information copy - text copy summary of details
    
    summary:
        fingerprint = Σ (α_i * gate_i * fp_i)
        where α_i = softmax(learnable_weights) text attention(fp_i)
    """
    
    def __init__(self, num_layers, fpl, aggregation_type='attention'):
        """
        Args:
            num_layers: data (details text)
            fpl: summary
            aggregation_type: details type
                - 'learned': learnable of summary
                - 'attention': base information of attention details
                - 'gated': summary
                - 'hierarchical': details attention (details)
        """
        super(MultiScaleAggregator, self).__init__()
        
        self.num_layers = num_layers
        self.fpl = fpl
        self.aggregation_type = aggregation_type
        
        # ============================================================
        # details 1: learnable of summary
        # ============================================================
        # information as split text, model summary
        self.layer_weights = nn.Parameter(torch.ones(num_layers))
        
        # ============================================================
        # details 2: base information of attention details
        # ============================================================
        # will details text attention split data
        self.attention_net = nn.Sequential(
            nn.Linear(fpl, fpl // 2),
            nn.Tanh(),
            nn.Linear(fpl // 2, 1, bias=False)
)
        
        # ============================================================
        # details 3: information copy
        # ============================================================
        # details one summary, details of details amount
        self.gate_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fpl, fpl // 4),
                nn.ReLU(),
                nn.Linear(fpl // 4, fpl),
                nn.Sigmoid()
) for _ in range(num_layers)
])
        
        # ============================================================
        # details 4: details attention (details)
        # ============================================================
        # summary encoding
        self.layer_position_embedding = nn.Parameter(torch.randn(num_layers, fpl // 4))
        self.hierarchical_attention = nn.Sequential(
            nn.Linear(fpl + fpl // 4, fpl // 2),
            nn.Tanh(),
            nn.Linear(fpl // 2, 1, bias=False)
)
        
        # ============================================================
        # details (can)
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
        details
        
        Args:
            fingerprints: list of [batch, fpl] details of details
        
        Returns:
            aggregated: [batch, fpl] details after of details
            weights: [batch, num_layers] summary (use can details)
        """
        batch_size = fingerprints[0].shape[0]
        
        # information has of details: [batch, num_layers, fpl]
        fp_stack = torch.stack(fingerprints, dim=1)
        
        if self.aggregation_type == 'learned':
            # details 1: learnable of summary
            weights = F.softmax(self.layer_weights, dim=0)  # [num_layers]
            weights = weights.view(1, -1, 1).expand(batch_size, -1, self.fpl)
            aggregated = (fp_stack * weights).sum(dim=1)  # [batch, fpl]
            weight_output = weights[:,:, 0]  # [batch, num_layers]
            
        elif self.aggregation_type == 'attention':
            # details 2: base information of attention details
            # calculation summary of attention split data
            attn_scores = self.attention_net(fp_stack)  # [batch, num_layers, 1]
            attn_scores = attn_scores.squeeze(-1)  # [batch, num_layers]
            weights = F.softmax(attn_scores, dim=1)  # [batch, num_layers]
            weights_expanded = weights.unsqueeze(-1)  # [batch, num_layers, 1]
            aggregated = (fp_stack * weights_expanded).sum(dim=1)  # [batch, fpl]
            weight_output = weights
            
        elif self.aggregation_type == 'gated':
            # details 3: summary
            gated_fps = []
            for i, (fp, gate_net) in enumerate(zip(fingerprints, self.gate_nets)):
                gate = gate_net(fp)  # [batch, fpl]
                gated_fps.append(fp * gate)
            
            # use learnable details after of details
            gated_stack = torch.stack(gated_fps, dim=1)  # [batch, num_layers, fpl]
            weights = F.softmax(self.layer_weights, dim=0)
            weights = weights.view(1, -1, 1).expand(batch_size, -1, self.fpl)
            aggregated = (gated_stack * weights).sum(dim=1)
            weight_output = weights[:,:, 0]
            
        elif self.aggregation_type == 'hierarchical':
            # details 4: details attention
            # as details details encoding
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
        
        # can: details one process
        aggregated = self.fusion_layer(aggregated)
        
        return aggregated, weight_output


# ============================================================
# GIN information (Graph Isomorphism Network Convolution)
# ============================================================
class GINConv(nn.Module):
    """
    Graph Isomorphism Network information (details channel)
    
    summary:
        h_i^(k+1) = MLP((1 + ε) * h_i^(k) + Σ_{j∈N(i)} h_j^(k))
    
    where ε text learnable parameter, MLP compare information property details have update text of table details. 
    GIN text validate details figure details test information WL-test of details, 
    information GAT details build model of molecule figure result information. 
    """
    
    def __init__(self, in_features, out_features, hidden_dim=None, dropout=0.1):
        """
        Args:
            in_features: details features dimension
            out_features: details features dimension
            hidden_dim: MLP middle details dimension (details and out_features details)
            dropout: Dropout compare rate
        """
        super(GINConv, self).__init__()
        
        if hidden_dim is None:
            hidden_dim = out_features
        
        # learnable parameter ε (details as 0, model information ring of needed property)
        self.epsilon = nn.Parameter(torch.zeros(1))
        
        # details MLP + BatchNorm (details table details)
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
        before information
        
        Args:
            atoms: [batch, num_atoms, in_features] atom features
            edges: [batch, num_atoms, max_degree] details
        
        Returns:
            new_atoms: [batch, num_atoms, out_features]
        """
        atoms = atoms.to(device)
        edges = edges.to(device)
        
        batch_size, num_atoms, in_features = atoms.shape
        
        # get details features (summary)
        neighbor_features = self.graphLookup(atoms, edges, maskvalue=0, include_self=False)
        # neighbor_features: [batch, num_atoms, max_degree, in_features]
        
        # summary: and
        neighbor_sum = neighbor_features.sum(dim=2)  # [batch, num_atoms, in_features]
        
        # GIN update: (1 + ε) * h_self + Σ h_neighbor
        out = (1 + self.epsilon) * atoms + neighbor_sum  # [batch, num_atoms, in_features]
        
        # details MLP (need reshape as 2D information BatchNorm1d)
        out_flat = out.view(-1, in_features)            # [batch*num_atoms, in_features]
        out_flat = self.mlp(out_flat)                    # [batch*num_atoms, out_features]
        out = out_flat.view(batch_size, num_atoms, -1)   # [batch, num_atoms, out_features]
        
        return out


# ============================================================
# text channel attention details (Cross-Channel Attention)
# ============================================================
class CrossChannelAttention(nn.Module):
    """
    text channel attention details module
    
    text pharmacophore channel and details channel of atom table details:
      h_pharm' = h_pharm + α * CrossAttn(Q=h_pharm, K=h_topo, V=h_topo)
      h_topo'  = h_topo  + α * CrossAttn(Q=h_topo,  K=h_pharm, V=h_pharm)
    
    where α text learnable of details data. 
    """
    
    def __init__(self, dim, num_heads=4, dropout=0.1):
        """
        Args:
            dim: features dimension (text channel details dimension single)
            num_heads: attention data
            dropout: Dropout compare rate
        """
        super(CrossChannelAttention, self).__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        # Pharm -> Topo of text attention (Query details pharm, Key/Value details topo)
        self.W_q_p2t = nn.Linear(dim, dim)
        self.W_k_p2t = nn.Linear(dim, dim)
        self.W_v_p2t = nn.Linear(dim, dim)
        
        # Topo -> Pharm of text attention (Query details topo, Key/Value details pharm)
        self.W_q_t2p = nn.Linear(dim, dim)
        self.W_k_t2p = nn.Linear(dim, dim)
        self.W_v_t2p = nn.Linear(dim, dim)
        
        # summary
        self.out_proj_pharm = nn.Linear(dim, dim)
        self.out_proj_topo = nn.Linear(dim, dim)
        
        # learnable of summary (details compare text, details channel summary)
        self.alpha_pharm = nn.Parameter(torch.tensor(0.1))
        self.alpha_topo = nn.Parameter(torch.tensor(0.1))
        
        # one and Dropout
        self.norm_pharm = nn.LayerNorm(dim)
        self.norm_topo = nn.LayerNorm(dim)
        self.attn_dropout = nn.Dropout(dropout)
        
        self.scale = self.head_dim ** -0.5
        self.to(device)
    
    def _cross_attn(self, query, key, value, W_q, W_k, W_v, out_proj, mask=None):
        """
        calculation text channel attention
        
        Args:
            query: [batch, num_atoms, dim] details channel
            key:   [batch, num_atoms, dim] bond value channel
            value: [batch, num_atoms, dim] bond value channel
            mask:  [batch, num_atoms] have effect atom details
        
        Returns:
            output: [batch, num_atoms, dim]
        """
        batch_size, num_atoms, _ = query.shape
        
        # text property information reshape as details
        Q = W_q(query).view(batch_size, num_atoms, self.num_heads, self.head_dim).transpose(1, 2)
        K = W_k(key).view(batch_size, num_atoms, self.num_heads, self.head_dim).transpose(1, 2)
        V = W_v(value).view(batch_size, num_atoms, self.num_heads, self.head_dim).transpose(1, 2)
        # Q, K, V: [batch, num_heads, num_atoms, head_dim]
        
        # attention split data
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # attn: [batch, num_heads, num_atoms, num_atoms]
        
        # details (summary atom)
        if mask is not None:
            mask_2d = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, num_atoms]
            attn = attn.masked_fill(mask_2d == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.attn_dropout(attn)
        
        # summary
        out = torch.matmul(attn, V)  # [batch, num_heads, num_atoms, head_dim]
        out = out.transpose(1, 2).contiguous().view(batch_size, num_atoms, -1)
        out = out_proj(out)
        
        return out
    
    def forward(self, h_pharm, h_topo, atom_mask=None):
        """
        information channel attention details
        
        Args:
            h_pharm: [batch, num_atoms, dim] pharmacophore channel table text
            h_topo:  [batch, num_atoms, dim] details channel table text
            atom_mask: [batch, num_atoms] have effect atom details (can)
        
        Returns:
            h_pharm_new: [batch, num_atoms, dim] details after of pharmacophore channel
            h_topo_new:  [batch, num_atoms, dim] details after of details channel
        """
        # Pharm text Topo get details
        cross_p = self._cross_attn(
            h_pharm, h_topo, h_topo,
            self.W_q_p2t, self.W_k_p2t, self.W_v_p2t,
            self.out_proj_pharm, atom_mask
)
        
        # Topo text Pharm get details
        cross_t = self._cross_attn(
            h_topo, h_pharm, h_pharm,
            self.W_q_t2p, self.W_k_t2p, self.W_v_t2p,
            self.out_proj_topo, atom_mask
)
        
        # residual fusion + LayerNorm
        h_pharm_new = self.norm_pharm(h_pharm + self.alpha_pharm * cross_p)
        h_topo_new = self.norm_topo(h_topo + self.alpha_topo * cross_t)
        
        return h_pharm_new, h_topo_new


# ============================================================
# dual-channel details figure summary (DC-PharmGAT)
# ============================================================
class DualChannelPharmGAT(nn.Module):
    """
    dual-channel details figure summary (Dual-Channel Heterogeneous Message Passing)
    
    innovation:
      channelA (pharmacophore channel): PharmacophoreGATConv - use TPP details attention
      channelB (details channel):   GINConv - details molecule figure details result information
      text channel details: CrossChannelAttention - details channel summary
    
    details of calculation details:
      1. channel A: h_pharm = PharmGATConv(atoms, edges)
      2. channel B: h_topo  = GINConv(atoms, edges)
      3. text channel:  h_pharm, h_topo = CrossAttn(h_pharm, h_topo)
      4. details + text single + details
      5. summary: fp_pharm, fp_topo
    
    summary: concat(agg(fp_pharm), agg(fp_topo)) -> information fpl
    """
    
    def __init__(self, layers, tpp_vector, return_activations=False, fpl=32,
                 num_heads=4, dropout=0.1, aggregation_type='attention', disable_tpp_bias=False, hard_concat_tpp=False, fixed_bias=False):
        """
        Args:
            layers: details features dimension text table, text [68, 64, 64, 64, 64]
            tpp_vector: Target Pharmacophore Profile vector
            return_activations: details return middle details active
            fpl: summary
            num_heads: attention data
            dropout: Dropout compare rate
            aggregation_type: details type
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
        
        # attention TPP
        self.register_buffer('tpp_vector', tpp_vector)
        
        # build information
        self.throughShape = list(zip(layers[:-1], layers[1:]))
        
        # ===== channel A: pharmacophore details of GAT =====
        self.pharm_gat_layers = nn.ModuleList()
        self.pharm_output_layers = nn.ModuleList()
        self.pharm_layer_norms = nn.ModuleList()
        
        # ===== channel B: GIN details encoding =====
        self.topo_gin_layers = nn.ModuleList()
        self.topo_output_layers = nn.ModuleList()
        self.topo_layer_norms = nn.ModuleList()
        
        # ===== text channel attention =====
        self.cross_attentions = nn.ModuleList()
        
        for idx, (in_dim, out_dim) in enumerate(self.throughShape):
            # channel A: PharmGATConv
            self.pharm_gat_layers.append(PharmacophoreGATConv(
                in_features=in_dim, out_features=out_dim,
                tpp_vector=self.tpp_vector, num_heads=num_heads,
                dropout=dropout, concat=False, disable_tpp_bias=self.disable_tpp_bias,
                hard_concat_tpp=self.hard_concat_tpp, fixed_bias=self.fixed_bias
))
            self.pharm_output_layers.append(FingerprintReadoutLayer(in_dim, self.fpl))
            self.pharm_layer_norms.append(nn.LayerNorm(out_dim))
            
            # channel B: GINConv
            self.topo_gin_layers.append(GINConv(
                in_features=in_dim, out_features=out_dim,
                hidden_dim=out_dim, dropout=dropout
))
            self.topo_output_layers.append(FingerprintReadoutLayer(in_dim, self.fpl))
            self.topo_layer_norms.append(nn.LayerNorm(out_dim))
            
            # text channel attention (text out_dim dimension information)
            self.cross_attentions.append(CrossChannelAttention(
                dim=out_dim, num_heads=num_heads, dropout=dropout
))
        
        # details
        self.pharm_final_output = FingerprintReadoutLayer(self.layers[-1], self.fpl)
        self.topo_final_output = FingerprintReadoutLayer(self.layers[-1], self.fpl)
        
        # figure details
        self.pool = GraphPool()
        
        # details (text channel one)
        num_fp_layers = len(self.throughShape) + 1
        self.pharm_aggregator = MultiScaleAggregator(
            num_layers=num_fp_layers, fpl=fpl, aggregation_type=aggregation_type
)
        self.topo_aggregator = MultiScaleAggregator(
            num_layers=num_fp_layers, fpl=fpl, aggregation_type=aggregation_type
)
        
        # dual-channel details: 2*fpl -> fpl
        self.channel_fusion = nn.Sequential(
            nn.Linear(fpl * 2, fpl),
            nn.LayerNorm(fpl),
            nn.ReLU(),
            nn.Dropout(dropout)
)
        
        # Hard Concatenation: figure text TPP summary
        if self.hard_concat_tpp:
            self.graph_tpp_projection = nn.Sequential(
                nn.Linear(fpl + PHARMACOPHORE_FEATURE_DIM, fpl),
                nn.LayerNorm(fpl),
                nn.ReLU(),
)
        
        # save details data
        self.layer_weights_history = None
        self.cross_attn_alpha_history = None
        
        self.to(device)
    
    def forward(self, input):
        """
        before information
        
        Args:
            input: (atoms, bonds, edges) details
        
        Returns:
            fingerprint: [batch, fpl] molecule details
        """
        self.atom_activations = []
        a, b, e = input
        a, b, e = a.to(device), b.to(device), e.to(device)
        
        # cache original details features (use SP-CL compare details)
        self._cached_input_features = a
        
        # text channel total information atom table text
        a_pharm = a  # pharmacophore channel
        a_topo = a   # details channel
        
        # summary of details
        pharm_fingerprints = []
        topo_fingerprints = []
        
        for i in range(len(self.throughShape)):
            # 1. information before information (information before)
            fp_pharm, act_pharm = self.pharm_output_layers[i](a_pharm, b, e)
            fp_topo, act_topo = self.topo_output_layers[i](a_topo, b, e)
            pharm_fingerprints.append(fp_pharm)
            topo_fingerprints.append(fp_topo)
            
            if self.return_activations:
                self.atom_activations.append(act_pharm)
            
            # 2. channel A: PharmGATConv
            a_pharm_new = self.pharm_gat_layers[i](a_pharm, e)
            
            # 3. channel B: GINConv
            a_topo_new = self.topo_gin_layers[i](a_topo, e)
            
            # 4. text channel attention details
            # calculation atom details: have details of atom details have effect atom
            atom_mask = (e!= -1).any(dim=-1).float()  # [batch, num_atoms]
            a_pharm_new, a_topo_new = self.cross_attentions[i](
                a_pharm_new, a_topo_new, atom_mask
)
            
            # 5. summary (result dimension details)
            if a_pharm.shape[-1] == a_pharm_new.shape[-1]:
                a_pharm_new = a_pharm_new + a_pharm
                a_topo_new = a_topo_new + a_topo
            
            # 6. details single + text active
            a_pharm_new = self.pharm_layer_norms[i](a_pharm_new)
            a_pharm_new = F.elu(a_pharm_new)
            a_topo_new = self.topo_layer_norms[i](a_topo_new)
            a_topo_new = F.elu(a_topo_new)
            
            # 7. figure details
            a_pharm = self.pool(a_pharm_new, e)
            a_topo = self.pool(a_topo_new, e)
        
        # cache details atom table text - information channel (use SP-CL compare details)
        self._cached_atom_repr = (a_pharm + a_topo) / 2
        
        # details text
        fp_pharm_final, act_pharm_final = self.pharm_final_output(a_pharm, b, e)
        fp_topo_final, act_topo_final = self.topo_final_output(a_topo, b, e)
        pharm_fingerprints.append(fp_pharm_final)
        topo_fingerprints.append(fp_topo_final)
        
        if self.return_activations:
            self.atom_activations.append(act_pharm_final)
        
        # details
        fp_pharm_agg, pharm_weights = self.pharm_aggregator(pharm_fingerprints)
        fp_topo_agg, topo_weights = self.topo_aggregator(topo_fingerprints)
        
        # save details data
        self.layer_weights_history = pharm_weights.detach()
        self.cross_attn_alpha_history = [
            (ca.alpha_pharm.item(), ca.alpha_topo.item())
            for ca in self.cross_attentions
]
        
        # dual-channel summary
        fp_concat = torch.cat([fp_pharm_agg, fp_topo_agg], dim=-1)  # [batch, 2*fpl]
        fingerprint = self.channel_fusion(fp_concat)                  # [batch, fpl]
        
        # Hard Concatenation: will TPP information figure text table text
        if self.hard_concat_tpp:
            batch_size_fp = fingerprint.shape[0]
            tpp_graph = self.tpp_vector.unsqueeze(0).expand(batch_size_fp, -1)  # [batch, 6]
            fp_with_tpp = torch.cat([fingerprint, tpp_graph], dim=-1)  # [batch, fpl + 6]
            fingerprint = self.graph_tpp_projection(fp_with_tpp)  # [batch, fpl]
        
        return fingerprint
    
    def get_layer_weights(self):
        """get pharmacophore channel of summary"""
        if self.layer_weights_history is not None:
            return self.layer_weights_history.mean(dim=0).cpu().numpy()
        if self.aggregation_type == 'learned':
            weights = F.softmax(self.pharm_aggregator.layer_weights, dim=0)
            return weights.detach().cpu().numpy()
        return None
    
    def get_cross_attn_alphas(self):
        """get text channel attention of summary α (use information training)"""
        return self.cross_attn_alpha_history


# ============================================================
# pharmacophore details of figure summary (PharmacophoreGAT) - enhanced version
# ============================================================
class PharmacophoreGAT(nn.Module):
    """
    pharmacophore details of graph attention network (enhanced version)
    
    new feature property: 
    - information features details: learnable of details summary
    - details details: learned, attention, gated, hierarchical
    """
    
    def __init__(self, layers, tpp_vector, return_activations=False, fpl=32, 
                 num_heads=4, dropout=0.1, aggregation_type='attention', disable_tpp_bias=False, hard_concat_tpp=False, fixed_bias=False):
        """
        Args:
            layers: details features dimension text table, text [68, 64, 64, 64, 64]
            tpp_vector: Target Pharmacophore Profile vector
            return_activations: details return middle details active
            fpl: summary
            num_heads: attention data
            dropout: Dropout compare rate
            aggregation_type: details type
                - 'learned': learnable of summary
                - 'attention': base information of attention details (details)
                - 'gated': summary
                - 'hierarchical': details attention
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
        
        # attention TPP
        self.register_buffer('tpp_vector', tpp_vector)
        
        # build information
        self.throughShape = list(zip(layers[:-1], layers[1:]))
        self.gat_layers, self.output_layers = self._init_layers()
        
        # details
        self.final_output = FingerprintReadoutLayer(self.layers[-1], self.fpl)
        
        # figure details
        self.pool = GraphPool()
        
        # details single (details training)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(out_dim) for _, out_dim in self.throughShape
])
        
        # ============================================================
        # information features information (core innovation)
        # ============================================================
        num_fp_layers = len(self.throughShape) + 1  # middle details + details
        self.multi_scale_aggregator = MultiScaleAggregator(
            num_layers=num_fp_layers,
            fpl=fpl,
            aggregation_type=aggregation_type
)
        
        # save summary use information
        self.layer_weights_history = None
        
        # Hard Concatenation: figure text TPP summary
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
            # pharmacophore details of GAT text
            gat_layer = PharmacophoreGATConv(
                in_features=in_dim,
                out_features=out_dim,
                tpp_vector=self.tpp_vector,
                num_heads=self.num_heads,
                dropout=0.1,
                concat=False,  # get details, save to dimension single
                disable_tpp_bias=self.disable_tpp_bias,
                hard_concat_tpp=self.hard_concat_tpp,
                fixed_bias=self.fixed_bias
)
            gat_layers.append(gat_layer)
            
            # information (use generate details)
            output_layers.append(FingerprintReadoutLayer(in_dim, self.fpl))
        
        # after single of details
        output_layers.append(FingerprintReadoutLayer(self.layers[-1], self.fpl))
        
        return gat_layers, output_layers
    
    def forward(self, input):
        """
        before information
        
        Args:
            input: (atoms, bonds, edges) details
                - atoms: [batch, num_atoms, num_features]
                - bonds: [batch, num_atoms, max_degree, bond_features]
                - edges: [batch, num_atoms, max_degree]
        
        Returns:
            fingerprint: [batch, fpl] molecule details
        """
        self.atom_activations = []
        a, b, e = input
        a, b, e = a.to(device), b.to(device), e.to(device)
        
        # cache original details features (use SP-CL compare details)
        self._cached_input_features = a
        
        # summary of summary (details)
        layer_fingerprints = []
        
        for i, (gat_layer, output_layer, layer_norm) in enumerate(
            zip(self.gat_layers, self.output_layers[:-1], self.layer_norms)
):
            # generate before text of summary
            fp_contrib, atom_act = output_layer(a, b, e)
            layer_fingerprints.append(fp_contrib)
            
            if self.return_activations:
                self.atom_activations.append(atom_act)
            
            # GAT details
            a_new = gat_layer(a, e)
            
            # summary (result dimension details)
            if a.shape[-1] == a_new.shape[-1]:
                a_new = a_new + a
            
            # details single + text active
            a_new = layer_norm(a_new)
            a_new = F.elu(a_new)
            
            # figure details
            a = self.pool(a_new, e)
        
        # cache details atom table text (use SP-CL compare details)
        self._cached_atom_repr = a
        
        # details of details
        fp_final, atom_act_final = self.final_output(a, b, e)
        layer_fingerprints.append(fp_final)
        
        if self.return_activations:
            self.atom_activations.append(atom_act_final)
        
        # ============================================================
        # information features details (details)
        # ============================================================
        fingerprint, layer_weights = self.multi_scale_aggregator(layer_fingerprints)
        
        # Hard Concatenation: will TPP information figure text table text
        if self.hard_concat_tpp:
            batch_size_fp = fingerprint.shape[0]
            tpp_graph = self.tpp_vector.unsqueeze(0).expand(batch_size_fp, -1)
            fp_with_tpp = torch.cat([fingerprint, tpp_graph], dim=-1)
            fingerprint = self.graph_tpp_projection(fp_with_tpp)
        
        # save details use information
        self.layer_weights_history = layer_weights.detach()
        
        return fingerprint
    
    def get_layer_weights(self):
        """get summary of summary (use can details property split text)"""
        if self.layer_weights_history is not None:
            # return summary
            return self.layer_weights_history.mean(dim=0).cpu().numpy()
        
        # result use 'learned' type, details return learnable details
        if self.aggregation_type == 'learned':
            weights = F.softmax(self.multi_scale_aggregator.layer_weights, dim=0)
            return weights.detach().cpu().numpy()
        
        return None


# ============================================================
# create new: molecule figure pharmacophore compare details positive details (SP-CL)
# Subgraph Pharmacophore Contrastive Learning Regularization
# ============================================================
class SubgraphPharmCL(nn.Module):
    """
    molecule figure pharmacophore compare details positive details module
    
    core innovation: will atom pharmacophore details extract details success text group/molecule figure text, information compare details
    information fail details GNN of molecular representation summary. 
    
    summary:
    1. atom features text last six dimensions extract pharmacophore type details
    2. information pharmacophore type, attention details type of atom GNN table text
       details motif embedding (molecule figure additional)
    3. and6learnable of "pharmacophore type text type" calculation information
    4. InfoNCE compare loss:
       - details: details TPP needed features of motif ↔ information type
       - details: motif ↔ summary type
    
    attention formula:
        motif_i = Σ_j (α_j * h_j)   where j ∈ {atom details pharmacophore type i}
        α_j = softmax(query_i^T · h_j)
    
    compare loss:
        L_CL = -log(exp(sim(motif_i, proto_i)/τ) / Σ_k exp(sim(motif_i, proto_k)/τ))
    """
    
    def __init__(self, hidden_dim, tpp_vector, proj_dim=64, temperature=0.07):
        """
        Args:
            hidden_dim: GNN details of atom features dimension
            tpp_vector: Target Pharmacophore Profile vector [6,]
            proj_dim: compare details dimension
            temperature: InfoNCE details argument (details)
        """
        super(SubgraphPharmCL, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.temperature = temperature
        
        # attention TPP as buffer
        self.register_buffer('tpp_vector', tpp_vector)
        
        # information: will GNN atom table summary low dimensions compare summary
        # details MLP + ReLU (SimCLR details)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
)
        
        # 6 learnable of pharmacophore type text type vector
        # details type text table single ideal of pharmacophore molecule figure table text
        self.prototypes = nn.Parameter(torch.randn(PHARMACOPHORE_FEATURE_DIM, proj_dim))
        nn.init.xavier_uniform_(self.prototypes)
        
        # attention details of details vector (details pharmacophore type one)
        # use details details type of atom table text
        self.attention_query = nn.Parameter(torch.randn(PHARMACOPHORE_FEATURE_DIM, hidden_dim))
        nn.init.xavier_uniform_(self.attention_query)
        
        self.to(device)
    
    def compute_motif_embeddings(self, atom_representations, atom_features):
        """
        extract details pharmacophore type of motif embedding
        
        Args:
            atom_representations: [batch, num_atoms, hidden_dim] GNN details
            atom_features: [batch, num_atoms, in_features] original atom features
        
        Returns:
            motif_embeddings: [batch, 6, proj_dim] 6text pharmacophore type of motif embedding
            motif_masks: [batch, 6] details pharmacophore type details molecule middle have details atom
        """
        batch_size, num_atoms, _ = atom_representations.shape
        
        # extract pharmacophore details [batch, num_atoms, 6] (atom features text last six dimensions)
        pharm_labels = atom_features[..., -PHARMACOPHORE_FEATURE_DIM:]
        
        # atom have effect details (exclude padding atom, padding atom features as 0)
        atom_mask = atom_features.abs().sum(-1) > 0  # [batch, num_atoms]
        
        motif_embeddings = []
        motif_masks = []
        
        for p in range(PHARMACOPHORE_FEATURE_DIM):
            # information pharmacophore type of atom details
            type_mask = (pharm_labels[..., p] > 0.5) & atom_mask  # [batch, num_atoms]
            
            # type information molecule middle cached
            has_type = type_mask.any(dim=1)  # [batch]
            motif_masks.append(has_type)
            
            # attention details: query_p^T · h_j
            query = self.attention_query[p]  # [hidden_dim]
            attn_scores = (atom_representations * query.unsqueeze(0).unsqueeze(0)).sum(-1)
            # attn_scores: [batch, num_atoms]
            
            # details type of atom details softmax
            attn_scores = attn_scores.masked_fill(~type_mask, float('-inf'))
            attn_weights = F.softmax(attn_scores, dim=1)  # [batch, num_atoms]
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            
            # details motif table text
            pooled = (atom_representations * attn_weights.unsqueeze(-1)).sum(dim=1)
            # pooled: [batch, hidden_dim]
            
            # summary compare summary
            projected = self.projector(pooled)  # [batch, proj_dim]
            motif_embeddings.append(projected)
        
        motif_embeddings = torch.stack(motif_embeddings, dim=1)  # [batch, 6, proj_dim]
        motif_masks = torch.stack(motif_masks, dim=1)  # [batch, 6]
        
        return motif_embeddings, motif_masks
    
    def forward(self, atom_representations, atom_features):
        """
        calculation molecule figure pharmacophore compare loss
        
        Args:
            atom_representations: [batch, num_atoms, hidden_dim] GNN details
            atom_features: [batch, num_atoms, in_features] original atom features
        
        Returns:
            loss: scalar, InfoNCE compare loss value
        """
        motif_embeddings, motif_masks = self.compute_motif_embeddings(
            atom_representations, atom_features
)
        # motif_embeddings: [batch, 6, proj_dim]
        # motif_masks: [batch, 6] molecule information have details pharmacophore type
        
        batch_size = motif_embeddings.shape[0]
        
        # L2 text single (compare details)
        motif_normed = F.normalize(motif_embeddings, dim=-1)
        proto_normed = F.normalize(self.prototypes, dim=-1)  # [6, proj_dim]
        
        # details: [batch, 6, 6]
        # sim[b, i, j] = cosine_sim(motif_i of molecule b, prototype_j) / τ
        sim_matrix = torch.matmul(motif_normed, proto_normed.T) / self.temperature
        
        # TPP needed features details: details TPP details as needed of pharmacophore type calculation compare loss
        tpp_important = (self.tpp_vector > 0.5)  # [6]
        
        # have effect details: pharmacophore type store details molecule middle AND TPPas type needed
        valid_mask = motif_masks & tpp_important.unsqueeze(0)  # [batch, 6]
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=atom_representations.device, requires_grad=True)
        
        # InfoNCE: information as positive details, details as negative details
        # details have effect of (molecule, pharm_type_i) text:
        #   L = -log(exp(sim(motif_i, proto_i)) / Σ_j exp(sim(motif_i, proto_j)))
        # summary cross_entropy(sim[b, i,:], target=i)
        labels = torch.arange(PHARMACOPHORE_FEATURE_DIM, device=sim_matrix.device)
        labels = labels.unsqueeze(0).expand(batch_size, -1)  # [batch, 6]
        
        # information after get have effect details
        sim_flat = sim_matrix.reshape(-1, PHARMACOPHORE_FEATURE_DIM)  # [batch*6, 6]
        labels_flat = labels.reshape(-1)  # [batch*6]
        valid_flat = valid_mask.reshape(-1)  # [batch*6]
        
        # details have effect details calculation summary fail
        loss = F.cross_entropy(
            sim_flat[valid_flat], 
            labels_flat[valid_flat], 
            reduction='mean'
)
        
        return loss


# ============================================================
# baseline graph encoder retained for ablation comparisons
# ============================================================
class GraphFingerprintEncoder(nn.Module):
    """Baseline molecular graph encoder used for ablation comparisons."""
    
    def __init__(self, layers, return_activations, fpl=32, hf=20):
        super(GraphFingerprintEncoder, self).__init__()
        self.layers = layers
        self.fpl = fpl
        self.throughShape = list(zip(layers[:-1], layers[1:]))
        self.layersArr, self.outputArr = self.init_layers()
        self.op = FingerprintReadoutLayer(self.layers[-1], self.fpl)
        self.pool = GraphPool()
        self.return_activations = return_activations
        self.atom_activations = None
        if self.return_activations:
            self.atom_activations = []
        self.to(device)

    def init_layers(self):
        layersArr, outputArr = [], []
        for idx, (i, o) in enumerate(self.throughShape):
            outputArr.append(FingerprintReadoutLayer(self.layers[idx], self.fpl))
            layersArr.append(GraphMessagePassingLayer(i, o))
        outputArr.append(FingerprintReadoutLayer(self.layers[-1], self.fpl))
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
# information split class details (save information)
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
            if o!= 1:
                self.ann.add_module(f'batch norm {j}', nn.BatchNorm1d(o))
                self.ann.add_module(f'relu act {j}', nn.ReLU())
                self.ann.add_module(f'dropout {j}', nn.Dropout(self.dropout))

    def forward(self, input):
        return self.ann(input)


# ============================================================
# model: pharmacophore details of summary (PharmacophoreGuidedDocking)
# ============================================================
class dockingProtocol(nn.Module):
    """
    pharmacophore details of molecule information test model
    
    result text:
        PharmacophoreGAT / DualChannelPharmGAT (figure details) -> dockingANN (information split class)
    
    innovation:
        - use PharmacophoreGATConv information have information
        - attention text copy middle introduce pharmacophore summary
        - β argument learnable, summary"summary"of details
        - [NEW] dual-channel details: pharmacophore GAT + details GIN + text channel attention
    """
    
    def __init__(self, params, tpp_path=None):
        """
        Args:
            params: model parameters details
            tpp_path: TPP file path (text 'cdk2/cdk2_tpp.pt')
        """
        super(dockingProtocol, self).__init__()
        
        # load TPP vector
        self.tpp_vector = load_tpp_vector(tpp_path)
        
        # information use pharmacophore details of GAT
        self.use_pharmacophore_gat = params.get("use_pharmacophore_gat", True)
        
        # information use dual-channel details
        self.use_dual_channel = params.get("use_dual_channel", False)
        
        # details type: 'learned', 'attention', 'gated', 'hierarchical'
        self.aggregation_type = params.get("aggregation_type", "attention")
        
        # ============================================================
        # SP-CL: molecule figure pharmacophore compare details positive details
        # ============================================================
        self.enable_spcl = params.get("enable_spcl", False)
        self.spcl_weight = params.get("spcl_weight", 0.1)  # compare loss details α
        
        # ablation experiment: details disable TPP Bias
        self.disable_tpp_bias = params.get("disable_tpp_bias", False)
        
        # ablation experiment: Hard Concatenation (TPP static concatenation)
        self.hard_concat_tpp = params.get("hard_concat_tpp", False)
        
        # ablation experiment: Fixed Bias (β fixed as constants, do not participate in training)
        self.fixed_bias = params.get("fixed_bias", False)
        
        if self.use_dual_channel:
            # ===== dual-channel details (innovation 1) =====
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
                fixed_bias=self.fixed_bias
)
            if self.hard_concat_tpp:
                ablation_tag = " [ABLATION: Hard Concatenation]"
            elif self.disable_tpp_bias:
                ablation_tag = " [ABLATION: TPP Bias DISABLED]"
            elif self.fixed_bias:
                ablation_tag = " [ABLATION: Fixed Bias (β frozen)]"
            else:
                ablation_tag = ""
            print(f"[DC-PharmGAT] Dual-channel enabled{ablation_tag}")
        elif self.use_pharmacophore_gat:
            # use have of PharmacophoreGAT (details)
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
                fixed_bias=self.fixed_bias
)
        else:
            # baseline graph encoder for ablation studies
            gnn = GraphFingerprintEncoder(
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
        
        # information SP-CL module (information use)
        if self.enable_spcl and (self.use_pharmacophore_gat or self.use_dual_channel):
            hidden_dim = params["conv"]["layers"][-1]  # GNN information dimension
            self.spcl_module = SubgraphPharmCL(
                hidden_dim=hidden_dim,
                tpp_vector=self.tpp_vector,
                proj_dim=params.get("spcl_proj_dim", 64),
                temperature=params.get("spcl_temperature", 0.07),
)
            print(f"[SP-CL] molecule figure pharmacophore compare summary use (weight={self.spcl_weight})")
        else:
            self.spcl_module = None
        
        self.return_activations = params["conv"]["activations"]
        self.to(device)
    
    def get_beta_values(self):
        """get has GAT text of β value (use information training)"""
        beta_info = []
        gnn = self.model[0]
        
        # details GAT details table
        gat_layers = None
        if self.use_dual_channel and hasattr(gnn, 'pharm_gat_layers'):
            gat_layers = gnn.pharm_gat_layers
        elif self.use_pharmacophore_gat and hasattr(gnn, 'gat_layers'):
            gat_layers = gnn.gat_layers
        
        if gat_layers is not None:
            for i, layer in enumerate(gat_layers):
                layer_betas = {}
                if hasattr(layer, 'beta_node'):
                    layer_betas['node'] = layer.beta_node.item()
                if hasattr(layer, 'beta_edge'):
                    layer_betas['edge'] = layer.beta_edge.item()
                if hasattr(layer, 'beta_interact'):
                    layer_betas['interact'] = layer.beta_interact.item()
                # details old details of beta argument
                if hasattr(layer, 'beta') and not layer_betas:
                    layer_betas['beta'] = layer.beta.item()
                if layer_betas:
                    beta_info.append(layer_betas)
        return beta_info
    
    def get_pharm_importance(self):
        """get pharmacophore features needed property details (use can details property split text)"""
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
        """get details of summary (use can details property split text)"""
        gnn = self.model[0]
        if hasattr(gnn, 'get_layer_weights'):
            return gnn.get_layer_weights()
        return None
    
    def get_cross_attn_alphas(self):
        """get text channel attention summary α (use information dual-channel training)"""
        if self.use_dual_channel:
            gnn = self.model[0]
            if hasattr(gnn, 'get_cross_attn_alphas'):
                return gnn.get_cross_attn_alphas()
        return None
    
    def compute_contrastive_loss(self, input):
        """
        calculation SP-CL molecule figure pharmacophore compare loss
        
        note: information forward() after use, as need cache of middle text table text. 
        
        Args:
            input: (atoms, bonds, edges) details (and forward details of details)
        
        Returns:
            cl_loss: scalar, compare loss value (information spcl_weight)
                     result SP-CL details use return 0
        """
        if self.spcl_module is None:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        gnn = self.model[0]  # PharmacophoreGAT text DualChannelPharmGAT
        
        # details store get GNN information atom table and original details features
        if not hasattr(gnn, '_cached_atom_repr') or gnn._cached_atom_repr is None:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        atom_repr = gnn._cached_atom_repr      # [batch, num_atoms, hidden_dim]
        atom_features = gnn._cached_input_features  # [batch, num_atoms, in_features]
        
        # calculation compare loss
        cl_loss = self.spcl_module(atom_repr, atom_features)
        
        return self.spcl_weight * cl_loss

    def forward(self, input):
        if self.return_activations:
            return torch.squeeze(self.model(input)), self.model[0].atom_activations
        return torch.squeeze(self.model(input))


# ============================================================
# dataset class (save information)
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
