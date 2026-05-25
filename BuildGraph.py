import torch
import torch.nn.functional as F
import random
from collections import defaultdict
import numpy as np
from sklearn.neighbors import NearestNeighbors

#### 构建基因-细胞图，只要有表达就连边
def build_gene_cell_graph(gene_cell: np.ndarray):
    """
    gene_cell: [n_genes, n_cells]

    Returns a lightweight graph dict.
    """
    gene_cell = np.asarray(gene_cell)
    n_genes, n_cells = gene_cell.shape

    g_idx, c_idx = np.nonzero(gene_cell)

    # adjacency dictionaries
    gene_to_cells = defaultdict(list)
    cell_to_genes = defaultdict(list)

    for g, c in zip(g_idx, c_idx):
        gene_to_cells[int(g)].append(int(c))
        cell_to_genes[int(c)].append(int(g))

    graph = {
        "n_genes": n_genes,
        "n_cells": n_cells,
        "gene_to_cells": gene_to_cells,
        "cell_to_genes": cell_to_genes,
    }
    return graph

#### 子图采样函数
def sample_bipartite_subgraph(
    graph,
    seed_genes,
    sample_depth: int = 2,
    sample_width_gene_to_cell: int = 64,
    sample_width_cell_to_gene: int = 64,
):
    """
    Sample a bipartite gene-cell subgraph from seed genes.

    Returns:
        sub_genes: sorted list of global gene ids
        sub_cells: sorted list of global cell ids
        edges_gc: list of (local_gene_idx, local_cell_idx)
    """
    gene_to_cells = graph["gene_to_cells"]
    cell_to_genes = graph["cell_to_genes"]

    sub_genes = set(int(g) for g in seed_genes)
    sub_cells = set()

    frontier_genes = set(sub_genes)
    frontier_cells = set()

    for _ in range(sample_depth):
        # expand gene -> cell
        new_cells = set()
        for g in frontier_genes:
            neigh_cells = gene_to_cells.get(g, [])
            if len(neigh_cells) > sample_width_gene_to_cell:
                neigh_cells = random.sample(neigh_cells, sample_width_gene_to_cell)
            new_cells.update(neigh_cells)

        frontier_cells = new_cells - sub_cells
        sub_cells.update(new_cells)

        # expand cell -> gene
        new_genes = set()
        for c in frontier_cells:
            neigh_genes = cell_to_genes.get(c, [])
            if len(neigh_genes) > sample_width_cell_to_gene:
                neigh_genes = random.sample(neigh_genes, sample_width_cell_to_gene)
            new_genes.update(neigh_genes)

        frontier_genes = new_genes - sub_genes
        sub_genes.update(new_genes)

    sub_genes = sorted(sub_genes)
    sub_cells = sorted(sub_cells)

    gene_map = {g: i for i, g in enumerate(sub_genes)}
    cell_map = {c: i for i, c in enumerate(sub_cells)}

    edges_gc = []
    for g in sub_genes:
        for c in gene_to_cells.get(g, []):
            if c in cell_map:
                edges_gc.append((gene_map[g], cell_map[c]))

    return sub_genes, sub_cells, edges_gc

def make_subgraph_jobs(
    graph,
    n_batch: int = 50,
    seed_gene_batch_size: int = 128,
    sample_depth: int = 2,
    sample_width_gene_to_cell: int = 64,
    sample_width_cell_to_gene: int = 64,
    seed: int = 0,
):
    random.seed(seed)
    np.random.seed(seed)

    n_genes = graph["n_genes"]
    all_genes = np.arange(n_genes)

    jobs = []
    for _ in range(n_batch):
        seed_genes = np.random.choice(
            all_genes,
            size=min(seed_gene_batch_size, n_genes),
            replace=False
        )
        sub_genes, sub_cells, edges_gc = sample_bipartite_subgraph(
            graph=graph,
            seed_genes=seed_genes,
            sample_depth=sample_depth,
            sample_width_gene_to_cell=sample_width_gene_to_cell,
            sample_width_cell_to_gene=sample_width_cell_to_gene,
        )
        jobs.append({
            "sub_genes": sub_genes,
            "sub_cells": sub_cells,
            "edges_gc": edges_gc,
        })
    return jobs

def prepare_subgraph_tensors(
    gene_cell: np.ndarray,
    job: dict,
    feature_mode: str = "ae",   # "ae" or "raw"
    gene_embedding: np.ndarray = None,
    cell_embedding: np.ndarray = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Convert one sampled subgraph into tensors for GNN/HGT.
    """

    sub_genes = job["sub_genes"]
    sub_cells = job["sub_cells"]
    edges_gc = job["edges_gc"]

    # submatrix target
    adj_sub = gene_cell[np.ix_(sub_genes, sub_cells)].astype(np.float32)

    # node features
    if feature_mode == "ae":
        assert gene_embedding is not None and cell_embedding is not None
        gene_feat = gene_embedding[sub_genes].astype(np.float32)
        cell_feat = cell_embedding[sub_cells].astype(np.float32)

    elif feature_mode == "raw":
        # gene feature = rows of gene_cell -> [sub_genes, all_cells] or [sub_genes, n_cells]
        # cell feature = rows of gene_cell.T -> [sub_cells, all_genes]
        gene_feat = gene_cell[sub_genes, :].astype(np.float32)
        cell_feat = gene_cell[:, sub_cells].T.astype(np.float32)
    else:
        raise ValueError("feature_mode must be 'ae' or 'raw'")

    # combine nodes
    n_g = len(sub_genes)
    n_c = len(sub_cells)

    if feature_mode == "ae":
        node_feature = np.concatenate([gene_feat, cell_feat], axis=0)   # [n_g+n_c, d]
    else:
        # raw mode: dimensions differ, return separately
        node_feature = [gene_feat, cell_feat]

    node_type = np.concatenate([
        np.zeros(n_g, dtype=np.int64),
        np.ones(n_c, dtype=np.int64)
    ])

    # build bidirectional edges
    # local gene ids: 0 .. n_g-1
    # local cell ids: n_g .. n_g+n_c-1 in unified graph
    src_gc = np.array([g for g, c in edges_gc], dtype=np.int64)
    dst_gc = np.array([c + n_g for g, c in edges_gc], dtype=np.int64)

    src_cg = dst_gc.copy()
    dst_cg = src_gc.copy()

    edge_index = np.vstack([
        np.concatenate([src_gc, src_cg]),
        np.concatenate([dst_gc, dst_cg])
    ])

    # edge type: 0 = gene->cell, 1 = cell->gene
    edge_type = np.concatenate([
        np.zeros(len(src_gc), dtype=np.int64),
        np.ones(len(src_cg), dtype=np.int64)
    ])

    # placeholder edge_time, same as original implementation idea
    edge_time = np.zeros(edge_index.shape[1], dtype=np.int64)

    # tensors
    if feature_mode == "ae":
        node_feature = torch.tensor(node_feature, dtype=torch.float32, device=device)
    else:
        node_feature = [
            torch.tensor(node_feature[0], dtype=torch.float32, device=device),
            torch.tensor(node_feature[1], dtype=torch.float32, device=device),
        ]

    node_type = torch.tensor(node_type, dtype=torch.long, device=device)
    edge_index = torch.tensor(edge_index, dtype=torch.long, device=device)
    edge_type = torch.tensor(edge_type, dtype=torch.long, device=device)
    edge_time = torch.tensor(edge_time, dtype=torch.long, device=device)
    adj_sub = torch.tensor(adj_sub, dtype=torch.float32, device=device)

    return {
        "node_feature": node_feature,
        "node_type": node_type,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "edge_time": edge_time,
        "adj_sub": adj_sub,
        "n_g": n_g,
        "n_c": n_c,
        "sub_genes": sub_genes,
        "sub_cells": sub_cells,
    }
    
def build_knn_edge_index(emb, k=10, metric="cosine", include_self=False, mutual=False):
    """
    emb: numpy array or torch tensor, shape [N, d]
    return: edge_index [2, E]
    """
    if isinstance(emb, torch.Tensor):
        emb = emb.detach().cpu().numpy()

    n = emb.shape[0]
    n_neighbors = k + 1 if include_self else k

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    nbrs.fit(emb)
    indices = nbrs.kneighbors(emb, return_distance=False)  # [N, k] or [N, k+1]

    edge_set = set()

    for i in range(n):
        neigh = indices[i].tolist()

        if include_self:
            neigh = [j for j in neigh if j != i]

        for j in neigh[:k]:
            edge_set.add((i, j))

    if mutual:
        edge_set = {(i, j) for (i, j) in edge_set if (j, i) in edge_set}

    # 常见做法：转成无向图
    undirected_edges = set()
    for i, j in edge_set:
        undirected_edges.add((i, j))
        undirected_edges.add((j, i))

    edge_index = torch.tensor(list(undirected_edges), dtype=torch.long).t().contiguous()
    return edge_index

def build_knn_edge_index(x, k=10, metric="cosine"):
    """
    x: torch.Tensor [N, d]
    return: edge_index [2, E]
    """
    x_np = x.detach().cpu().numpy()
    n = x_np.shape[0]

    if n <= 1:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)

    k_eff = min(k, n - 1)
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1, metric=metric)
    nbrs.fit(x_np)
    indices = nbrs.kneighbors(x_np, return_distance=False)

    edges = set()
    for i in range(n):
        neigh = [j for j in indices[i].tolist() if j != i][:k_eff]
        for j in neigh:
            edges.add((i, j))
            edges.add((j, i))  # 无向化

    if len(edges) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)

    edge_index = torch.tensor(list(edges), dtype=torch.long, device=x.device).t().contiguous()
    return edge_index

# 全局 KNN 建图
def build_global_knn_edge_index(emb, k=10, metric="cosine", device="cpu"):
    """
    emb: np.ndarray or torch.Tensor [N, d]
    return: global edge_index [2, E]
    """
    if isinstance(emb, torch.Tensor):
        emb = emb.detach().cpu().numpy()

    n = emb.shape[0]
    if n <= 1:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    k_eff = min(k, n - 1)
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1, metric=metric)
    nbrs.fit(emb)
    indices = nbrs.kneighbors(emb, return_distance=False)

    edges = set()
    for i in range(n):
        neigh = [j for j in indices[i].tolist() if j != i][:k_eff]
        for j in neigh:
            edges.add((i, j))
            edges.add((j, i))  # 无向化

    if len(edges) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    edge_index = torch.tensor(list(edges), dtype=torch.long, device=device).t().contiguous()
    return edge_index

# 全局 KNN 建图，用GPU, 分块计算相似度矩阵以节省内存
@torch.no_grad()
def build_global_knn_edge_index_torch(
    emb,
    k=10,
    device="cuda" if torch.cuda.is_available() else "cpu",
    chunk_size=4096,
    make_undirected=True,
    return_cpu=True,
):
    """
    emb: np.ndarray or torch.Tensor [N, d]
    return: edge_index [2, E]
    cosine similarity + topk
    """
    if not isinstance(emb, torch.Tensor):
        emb = torch.tensor(emb, dtype=torch.float32)
    else:
        emb = emb.detach().to(torch.float32)

    emb = emb.to(device)
    n = emb.shape[0]

    if n <= 1:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        return edge_index.cpu() if return_cpu else edge_index

    k_eff = min(k, n - 1)

    # cosine similarity = normalized dot product
    emb = F.normalize(emb, p=2, dim=1)

    idx_chunks = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)

        sim_chunk = emb[start:end] @ emb.T  # [chunk, N]

        row_idx = torch.arange(start, end, device=device)
        sim_chunk[torch.arange(end - start, device=device), row_idx] = -1e9

        topk_idx = torch.topk(sim_chunk, k=k_eff, dim=1, largest=True).indices
        idx_chunks.append(topk_idx)

        del sim_chunk

    nn_idx = torch.cat(idx_chunks, dim=0)  # [N, k]

    src = torch.arange(n, device=device).repeat_interleave(k_eff)
    dst = nn_idx.reshape(-1)

    edge_index = torch.stack([src, dst], dim=0)

    if make_undirected:
        rev_edge_index = torch.stack([dst, src], dim=0)
        edge_index = torch.cat([edge_index, rev_edge_index], dim=1)
        edge_index = torch.unique(edge_index, dim=1)

    if return_cpu:
        edge_index = edge_index.cpu()

    return edge_index

# 从全局图截取 batch 子图边
def induce_subgraph_edges(global_edge_index, sub_nodes, device="cpu"):
    """
    global_edge_index: [2, E], global node ids
    sub_nodes: 当前 batch 的全局节点编号
    return: local edge_index [2, E_sub]
    """
    if len(sub_nodes) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    if isinstance(global_edge_index, torch.Tensor):
        global_edge_index = global_edge_index.detach().cpu()

    sub_nodes = list(sub_nodes)
    sub_set = set(sub_nodes)
    global_to_local = {g: i for i, g in enumerate(sub_nodes)}

    src = global_edge_index[0].tolist()
    dst = global_edge_index[1].tolist()

    edges = []
    for s, d in zip(src, dst):
        if s in sub_set and d in sub_set:
            edges.append((global_to_local[s], global_to_local[d]))

    if len(edges) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    local_edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    return local_edge_index