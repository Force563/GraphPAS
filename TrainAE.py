import copy
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
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from AutoEncoder import GraphPAS_AutoEncoder


# =========================
# 1) loader builders
# =========================

def build_matrix_loaders(
    x: torch.Tensor,
    batch_size: int = 1024,
    val_ratio: float = 0.1,
    seed: int = 0,
):
    """
    x shape: [n_samples, n_features]
    Return:
        train_loader, val_loader
    """
    x = x.to(torch.float32)

    dataset = TensorDataset(x)
    n_total = len(dataset)
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val

    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=min(batch_size, len(train_set)),
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=min(batch_size, len(val_set)),
        shuffle=False,
        drop_last=False,
    )
    return train_loader, val_loader


def build_gene_ae_loaders(
    gene_cell: np.ndarray,
    batch_size: int = 1024,
    val_ratio: float = 0.1,
    seed: int = 0,
):
    """
    gene AE input:
        gene_cell shape [n_genes, n_cells]
        sample = gene
        feature = cells
    """
    gene_x = torch.tensor(gene_cell, dtype=torch.float32)
    train_loader, val_loader = build_matrix_loaders(
        gene_x,
        batch_size=batch_size,
        val_ratio=val_ratio,
        seed=seed,
    )
    return train_loader, val_loader, gene_x


def build_cell_ae_loaders(
    gene_cell: np.ndarray,
    batch_size: int = 1024,
    val_ratio: float = 0.1,
    seed: int = 0,
):
    """
    cell AE input:
        gene_cell.T shape [n_cells, n_genes]
        sample = cell
        feature = genes
    """
    cell_x = torch.tensor(gene_cell.T, dtype=torch.float32)
    train_loader, val_loader = build_matrix_loaders(
        cell_x,
        batch_size=batch_size,
        val_ratio=val_ratio,
        seed=seed,
    )
    return train_loader, val_loader, cell_x


# =========================
# 2) 通用训练器
# =========================

def train_autoencoder_from_loaders(
    model,
    train_loader,
    val_loader,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    max_epochs: int = 500,
    patience: int = 30,
    min_delta: float = 1e-5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
):
    """
    Train one autoencoder using provided loaders.
    Assumes each batch is (batch_x,)
    """
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    wait = 0

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in range(max_epochs):
        model.train()
        train_losses = []

        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)

            x_hat, z = model(batch_x)
            loss = criterion(x_hat, batch_x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for (batch_x,) in val_loader:
                batch_x = batch_x.to(device)

                x_hat, z = model(batch_x)
                loss = criterion(x_hat, batch_x)

                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if verbose and (epoch % 20 == 0 or epoch == max_epochs - 1):
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f}"
            )

        if val_loss < best_val - min_delta:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch:03d}, best_val={best_val:.6f}")
                break

    model.load_state_dict(best_state)
    return model, history


# =========================
# 3) 高层接口：gene AE
# =========================

def fit_gene_autoencoder(
    gene_cell: np.ndarray,
    latent_dim: int = 256,
    hidden_dims=(512,),
    dropout: float = 0.0,
    output_activation=None,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 1024,
    max_epochs: int = 500,
    val_ratio: float = 0.1,
    patience: int = 30,
    min_delta: float = 1e-5,
    seed: int = 0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
):
    """
    gene AE:
        input = gene_cell [n_genes, n_cells]
        sample = gene
    """
    gene_cell = gene_cell.astype(np.float32)

    train_loader, val_loader, gene_x = build_gene_ae_loaders(
        gene_cell=gene_cell,
        batch_size=batch_size,
        val_ratio=val_ratio,
        seed=seed,
    )

    model = GraphPAS_AutoEncoder(
        input_dim=gene_x.shape[1],
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        dropout=dropout,
        output_activation=output_activation,
    )

    if verbose:
        print("Training gene AE...")

    model, history = train_autoencoder_from_loaders(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        patience=patience,
        min_delta=min_delta,
        device=device,
        verbose=verbose,
    )
    print(next(model.parameters()).device)

    gene_embedding = model.get_embedding(
        gene_x.to(device),
        batch_size=batch_size
    ).numpy()

    return {
        "model": model,
        "embedding": gene_embedding,   # [n_genes, latent_dim]
        "history": history,
        "input_tensor": gene_x,
        "train_loader": train_loader,
        "val_loader": val_loader,
    }


# =========================
# 4) 高层接口：cell AE
# =========================

def fit_cell_autoencoder(
    gene_cell: np.ndarray,
    latent_dim: int = 256,
    hidden_dims=(512,),
    dropout: float = 0.0,
    output_activation=None,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 1024,
    max_epochs: int = 500,
    val_ratio: float = 0.1,
    patience: int = 30,
    min_delta: float = 1e-5,
    seed: int = 0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
):
    """
    cell AE:
        input = gene_cell.T [n_cells, n_genes]
        sample = cell
    """
    gene_cell = gene_cell.astype(np.float32)

    train_loader, val_loader, cell_x = build_cell_ae_loaders(
        gene_cell=gene_cell,
        batch_size=batch_size,
        val_ratio=val_ratio,
        seed=seed,
    )

    model = GraphPAS_AutoEncoder(
        input_dim=cell_x.shape[1],
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        dropout=dropout,
        output_activation=output_activation,
    )

    if verbose:
        print("Training cell AE...")

    model, history = train_autoencoder_from_loaders(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        patience=patience,
        min_delta=min_delta,
        device=device,
        verbose=verbose,
    )
    print(next(model.parameters()).device)

    cell_embedding = model.get_embedding(
        cell_x.to(device),
        batch_size=batch_size
    ).numpy()

    return {
        "model": model,
        "embedding": cell_embedding,   # [n_cells, latent_dim]
        "history": history,
        "input_tensor": cell_x,
        "train_loader": train_loader,
        "val_loader": val_loader,
    }


# =========================
# 5) 兼容你原来的总接口
# =========================

def fit_gene_cell_autoencoders(
    gene_cell: np.ndarray,
    latent_dim: int = 256,
    hidden_dims=(512,),
    dropout: float = 0.0,
    output_activation=None,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 1024,
    max_epochs: int = 500,
    val_ratio: float = 0.1,
    patience: int = 30,
    min_delta: float = 1e-5,
    seed: int = 0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
):
    """
    gene_cell shape: [n_genes, n_cells]
    """
    gene_result = fit_gene_autoencoder(
        gene_cell=gene_cell,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        output_activation=output_activation,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=max_epochs,
        val_ratio=val_ratio,
        patience=patience,
        min_delta=min_delta,
        seed=seed,
        device=device,
        verbose=verbose,
    )

    cell_result = fit_cell_autoencoder(
        gene_cell=gene_cell,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        output_activation=output_activation,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=max_epochs,
        val_ratio=val_ratio,
        patience=patience,
        min_delta=min_delta,
        seed=seed,
        device=device,
        verbose=verbose,
    )

    return {
        "gene_ae": gene_result["model"],
        "cell_ae": cell_result["model"],
        "gene_embedding": gene_result["embedding"],
        "cell_embedding": cell_result["embedding"],
        "gene_history": gene_result["history"],
        "cell_history": cell_result["history"],
        "gene_train_loader": gene_result["train_loader"],
        "gene_val_loader": gene_result["val_loader"],
        "cell_train_loader": cell_result["train_loader"],
        "cell_val_loader": cell_result["val_loader"],
    }