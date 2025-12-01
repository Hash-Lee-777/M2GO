import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import torch

def apply_geometric_refinement(
    xyz, 
    pred_labels, 
    unknown_id, 
    knn_k=10, 
    cluster_eps=0.6, 
    min_cluster_size=30,
    fallback_class_id=0
):
    """
    Module B: Spatio-Geometric Refinement (SGR)
    
    Args:
        xyz (np.ndarray or torch.Tensor): (N, 3) 原始点云坐标
        pred_labels (np.ndarray or torch.Tensor): (N, ) 预测标签
        unknown_id (int): Unknown 类别的 ID (通常是 num_classes - 1)
        knn_k (int): 局部平滑的邻居数量
        cluster_eps (float): DBSCAN 聚类半径 (米)
        min_cluster_size (int): 认定为有效物体的最小点数
        fallback_class_id (int): 被判定为噪点的 Unknown 将被重置为此 ID (通常设为背景类或 Ignore)
        
    Returns:
        np.ndarray: Refined labels (N, )
    """
    # 1. 数据格式统一转为 Numpy
    if isinstance(xyz, torch.Tensor):
        xyz = xyz.cpu().numpy()
    if isinstance(pred_labels, torch.Tensor):
        pred_labels = pred_labels.cpu().numpy()
        
    # 复制一份标签，避免修改原数据
    refined_labels = pred_labels.copy()

    # ======================================================
    # Step 1: Micro-level - Local Neighborhood Propagation (k-NN)
    # 作用：利用周围点的投票来平滑边界，找回被误判的 Unknown 边缘
    # ======================================================
    if knn_k > 0:
        # 建立 KD-Tree (algorithm='auto' 会自动选择最快的)
        nbrs = NearestNeighbors(n_neighbors=knn_k, algorithm='auto', n_jobs=-1).fit(xyz)
        _, indices = nbrs.kneighbors(xyz)
        
        # 获取邻居的标签 (N, k)
        neighbor_labels = pred_labels[indices]
        
        # 众数投票 (Mode Voting)
        # mode 返回 (modes, counts)，我们需要 modes
        modes, _ = stats.mode(neighbor_labels, axis=1, keepdims=False)
        refined_labels = modes.squeeze() # 更新标签

    # ======================================================
    # Step 2: Macro-level - Global Density Filtering (DBSCAN)
    # 作用：剔除空间上离散的 Unknown 预测点 (False Positives)
    # ======================================================
    
    # 仅提取被预测为 Unknown 的点进行聚类 (节省计算量)
    unk_mask = (refined_labels == unknown_id)
    unk_xyz = xyz[unk_mask]
    
    # 只有当存在 Unknown 点时才执行
    if len(unk_xyz) > 0:
        # 执行 DBSCAN
        # eps: 两个点被视为邻居的最大距离
        # min_samples: 形成核心点所需的最小邻居数
        clustering = DBSCAN(eps=cluster_eps, min_samples=2, n_jobs=-1).fit(unk_xyz)
        cluster_labels = clustering.labels_ # 噪声点会被标记为 -1
        
        # 统计每个簇的点数
        unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
        
        # 筛选有效簇：
        # 1. 簇 ID 不为 -1 (不是 DBSCAN 自身的噪声)
        # 2. 簇的大小 >= min_cluster_size (具有物理体积)
        valid_clusters = unique_clusters[(counts >= min_cluster_size) & (unique_clusters != -1)]
        
        # 构造有效点的 Mask (在 unk_xyz 这个子集中的 Mask)
        is_valid_point_in_subset = np.isin(cluster_labels, valid_clusters)
        
        # 找出需要被剔除的点在原图中的索引
        full_indices = np.where(unk_mask)[0]
        noise_indices = full_indices[~is_valid_point_in_subset]
        
        # 将噪点重置 (修正误报)
        # 这里我们将原本预测为 Unknown 但实际上是噪点的点，强制改为 fallback_class_id
        if len(noise_indices) > 0:
            refined_labels[noise_indices] = fallback_class_id

    return refined_labels