import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, uniform
from torch_geometric.utils import softmax
import copy
import random
import copy
import numpy as np
from tqdm.auto import tqdm
import torch.nn.functional as F
from BuildGraph import build_gene_cell_graph, make_subgraph_jobs, prepare_subgraph_tensors, build_global_knn_edge_index_torch, induce_subgraph_edges
from GAT import HomoGAT
from HGT import GNN, GNN_from_raw, GAT_HGT_Wrapper
from Loss import compute_kl_reconstruction_loss, compute_cross_linkpred_loss


def build_hgt_model(
    feature_mode: str,
    gene_embedding_dim: int = None,
    cell_embedding_dim: int = None,
    n_hid: int = 128,
    n_heads: int = 8,
    n_layers: int = 2,
    dropout: float = 0.2,
    AEtype: int = 1,
    use_gat: bool = True,
    gat_hidden_dim: int = 128,
    gat_heads: int = 4,
    gat_dropout: float = 0.2,
    knn_k: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    if feature_mode == "ae":
        assert gene_embedding_dim is not None
        hgt_model = GNN(
            in_dim=gene_embedding_dim,
            n_hid=n_hid,
            num_types=2,
            num_relations=2,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            conv_name='hgt',
            prev_norm=True,
            last_norm=True,
            use_RTE=False,
        ).to(device)

        if use_gat:
            model = GAT_HGT_Wrapper(
                hgt_model=hgt_model,
                in_dim=gene_embedding_dim,
                gat_hidden_dim=gat_hidden_dim,
                gat_heads=gat_heads,
                gat_dropout=gat_dropout,
            ).to(device)
        else:
            model = hgt_model

    elif feature_mode == "raw":
        assert gene_embedding_dim is not None and cell_embedding_dim is not None
        model = GNN_from_raw(
            in_dim=[gene_embedding_dim, cell_embedding_dim],
            n_hid=n_hid,
            num_types=2,
            num_relations=2,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            conv_name='hgt',
            prev_norm=True,
            last_norm=True,
            use_RTE=False,
            AEtype=AEtype,
        ).to(device)
    else:
        raise ValueError("feature_mode must be 'ae' or 'raw'")

    return model

def forward_one_subgraph(
    model,
    batch_data,
    feature_mode: str = "ae",
    loss_type: str = "kl",
):
    """
    batch_data from prepare_subgraph_tensors(...)
    """

    node_feature = batch_data["node_feature"]
    node_type = batch_data["node_type"]
    edge_index = batch_data["edge_index"]
    edge_type = batch_data["edge_type"]
    edge_time = batch_data["edge_time"]
    adj_sub = batch_data["adj_sub"]
    n_g = batch_data["n_g"]

    if feature_mode == "ae":
        node_rep = model(
            node_feature=node_feature,
            node_type=node_type,
            edge_time=edge_time,
            edge_index=edge_index,
            edge_type=edge_type,
            edge_index_gg=batch_data.get("edge_index_gg", None),
            edge_index_cc=batch_data.get("edge_index_cc", None),
        )
    elif feature_mode == "raw":
        # original raw branch returns node_rep, node_decoded_embedding
        node_rep, node_decoded_embedding = model(
            node_feature=node_feature,
            node_type=node_type,
            edge_time=edge_time,
            edge_index=edge_index,
            edge_type=edge_type
        )
    else:
        raise ValueError("feature_mode must be 'ae' or 'raw'")

    gene_z = node_rep[node_type == 0]
    cell_z = node_rep[node_type == 1]

    if loss_type == "kl":
        loss, decoder = compute_kl_reconstruction_loss(gene_z, cell_z, adj_sub)

    elif loss_type == "cross":
        loss = compute_cross_linkpred_loss(node_rep, edge_index)
        decoder = None

    else:
        raise ValueError("loss_type must be 'kl' or 'cross'")

    return {
        "loss": loss,
        "node_rep": node_rep,
        "gene_z": gene_z,
        "cell_z": cell_z,
        "decoder": decoder,
        "attention": getattr(model, "att", None),
    }
    
def train_hgt_with_subgraph_sampling(
    gene_cell: np.ndarray,
    feature_mode: str = "ae",        # "ae" or "raw"
    loss_type: str = "kl",           # "kl" or "cross"
    gene_embedding: np.ndarray = None,
    cell_embedding: np.ndarray = None,
    n_hid: int = 128,
    n_heads: int = 8,
    n_layers: int = 2,
    dropout: float = 0.2,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    max_epochs: int = 100,
    patience: int = 10,
    n_batch: int = 50,
    seed_gene_batch_size: int = 128,
    sample_depth: int = 2,
    sample_width_gene_to_cell: int = 64,
    sample_width_cell_to_gene: int = 64,
    AEtype: int = 1,

    # 新增：全局 KNN 图参数
    use_gat: bool = True,
    knn_k: int = 10,
    knn_metric: str = "cosine", # "euclidean" or "cosine"
    gat_hidden_dim: int = 128,
    gat_heads: int = 4,
    gat_dropout: float = 0.2,

    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
):
    """
    Main training function.
    """
    gene_cell = np.asarray(gene_cell, dtype=np.float32)
    n_genes, n_cells = gene_cell.shape

    # build graph
    graph = build_gene_cell_graph(gene_cell)

    # prepare jobs
    jobs = make_subgraph_jobs(
        graph=graph,
        n_batch=n_batch,
        seed_gene_batch_size=seed_gene_batch_size,
        sample_depth=sample_depth,
        sample_width_gene_to_cell=sample_width_gene_to_cell,
        sample_width_cell_to_gene=sample_width_cell_to_gene,
        seed=0,
    )

    # infer input dims
    if feature_mode == "ae":
        assert gene_embedding is not None and cell_embedding is not None
        gene_embedding_dim = gene_embedding.shape[1]
        cell_embedding_dim = cell_embedding.shape[1]
        assert gene_embedding_dim == cell_embedding_dim
    else:
        gene_embedding_dim = gene_cell.shape[1]   # gene feature dim = n_cells
        cell_embedding_dim = gene_cell.shape[0]   # cell feature dim = n_genes

    # build model
    model = build_hgt_model(
        feature_mode=feature_mode,
        gene_embedding_dim=gene_embedding_dim,
        cell_embedding_dim=cell_embedding_dim,
        n_hid=n_hid,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        AEtype=AEtype,
        use_gat=use_gat,
        gat_hidden_dim=gat_hidden_dim,
        gat_heads=gat_heads,
        gat_dropout=gat_dropout,
        knn_k=knn_k,
        device=device,
    )

    # 关键新增：训练前全局构一次 gg / cc 图
    global_edge_index_gg = None
    global_edge_index_cc = None

    if feature_mode == "ae" and use_gat:
        if verbose:
            print("Building global gene-gene KNN graph...", flush=True)
        global_edge_index_gg = build_global_knn_edge_index_torch(
            gene_embedding,
            k=knn_k,
            device=device,
            chunk_size=4096,
            make_undirected=True,
            return_cpu=True,
        )

        if verbose:
            print("Building global cell-cell KNN graph...", flush=True)
        global_edge_index_cc = build_global_knn_edge_index_torch(
            cell_embedding,
            k=knn_k,
            device=device,
            chunk_size=4096,
            make_undirected=True,
            return_cpu=True,
        )

        if verbose:
            print(
                f"Global gg edges: {global_edge_index_gg.size(1)}, "
                f"Global cc edges: {global_edge_index_cc.size(1)}",
                flush=True
            )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    wait = 0
    history = []

    for epoch in tqdm(range(max_epochs), desc="Epoch", disable=not verbose):
        model.train()
        epoch_loss = 0.0

        if verbose:
            print(f"Start epoch {epoch:03d}", flush=True)

        for bi, job in tqdm(
                enumerate(jobs),
                total=len(jobs),
                desc=f"Batch {epoch:03d}",
                leave=False,
                disable=not verbose,
            ):
            if verbose and bi % 10 == 0:
                print(f"  epoch {epoch:03d} batch {bi}/{len(jobs)}", flush=True)

            batch_data = prepare_subgraph_tensors(
                gene_cell=gene_cell,
                job=job,
                feature_mode=feature_mode,
                gene_embedding=gene_embedding,
                cell_embedding=cell_embedding,
                device=device,
            )

            # 关键新增：从全局 gg / cc 图里截取当前 batch 的局部边
            if feature_mode == "ae" and use_gat:
                sub_genes = job["sub_genes"]
                sub_cells = job["sub_cells"]

                edge_index_gg = induce_subgraph_edges(
                    global_edge_index_gg,
                    sub_genes,
                    device=device,
                )
                edge_index_cc = induce_subgraph_edges(
                    global_edge_index_cc,
                    sub_cells,
                    device=device,
                )

                batch_data["edge_index_gg"] = edge_index_gg
                batch_data["edge_index_cc"] = edge_index_cc

            out = forward_one_subgraph(
                model=model,
                batch_data=batch_data,
                feature_mode=feature_mode,
                loss_type=loss_type,
            )

            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())

        epoch_loss = epoch_loss / max(1, len(jobs))
        history.append(epoch_loss)

        if verbose:
            print(f"Epoch {epoch:03d} | avg_loss={epoch_loss:.6f}", flush=True)

        if epoch_loss < best_loss - 1e-6:
            best_loss = epoch_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch:03d}, best_loss={best_loss:.6f}", flush=True)
                break

    model.load_state_dict(best_state)
    return {
        "model": model,
        "graph": graph,
        "jobs": jobs,
        "history": history,
        "best_loss": best_loss,
        "global_edge_index_gg": global_edge_index_gg,
        "global_edge_index_cc": global_edge_index_cc,
    }
    
@torch.no_grad()
def infer_hgt_embeddings_with_global_graph(
    model,
    gene_cell,
    gene_embedding,
    cell_embedding,
    global_edge_index_gg,
    global_edge_index_cc,
    gene_block_size=1024,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    model.eval()

    gene_cell = np.asarray(gene_cell, dtype=np.float32)
    n_genes, n_cells = gene_cell.shape
    all_cells = list(range(n_cells))

    refined_gene_blocks = []
    final_cell_embedding = None

    for start in range(0, n_genes, gene_block_size):
        end = min(start + gene_block_size, n_genes)
        sub_genes = list(range(start, end))
        sub_cells = all_cells

        adj_sub = gene_cell[np.ix_(sub_genes, sub_cells)]
        g_local, c_local = np.nonzero(adj_sub)
        edges_gc = list(zip(g_local.tolist(), c_local.tolist()))

        job = {
            "sub_genes": sub_genes,
            "sub_cells": sub_cells,
            "edges_gc": edges_gc,
        }

        batch_data = prepare_subgraph_tensors(
            gene_cell=gene_cell,
            job=job,
            feature_mode="ae",
            gene_embedding=gene_embedding,
            cell_embedding=cell_embedding,
            device=device,
        )

        # 加局部 gg / cc 边
        batch_data["edge_index_gg"] = induce_subgraph_edges(
            global_edge_index_gg, sub_genes, device=device
        )
        batch_data["edge_index_cc"] = induce_subgraph_edges(
            global_edge_index_cc, sub_cells, device=device
        )

        node_rep = model(
            node_feature=batch_data["node_feature"],
            node_type=batch_data["node_type"],
            edge_time=batch_data["edge_time"],
            edge_index=batch_data["edge_index"],
            edge_type=batch_data["edge_type"],
            edge_index_gg=batch_data["edge_index_gg"],
            edge_index_cc=batch_data["edge_index_cc"],
        )

        node_type = batch_data["node_type"]
        gene_z = node_rep[node_type == 0].detach().cpu().numpy()
        cell_z = node_rep[node_type == 1].detach().cpu().numpy()

        refined_gene_blocks.append(gene_z)
        final_cell_embedding = cell_z

    refined_gene_embedding = np.vstack(refined_gene_blocks)

    return {
        "refined_gene_embedding": refined_gene_embedding,
        "refined_cell_embedding": final_cell_embedding,
    }