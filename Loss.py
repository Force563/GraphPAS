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

def compute_kl_reconstruction_loss(gene_z, cell_z, adj_sub):
    """
    gene_z: [n_g, d]
    cell_z: [n_c, d]
    adj_sub: [n_g, n_c]
    """
    decoder = gene_z @ cell_z.T
    loss = F.kl_div(
        decoder.softmax(dim=-1).log(),
        adj_sub.softmax(dim=-1),
        reduction='sum'
    )
    return loss, decoder

def compute_cross_linkpred_loss(node_rep, edge_index):
    """
    node_rep: [n_nodes, d]
    edge_index: [2, n_edges]
    """
    EPS = 1e-15

    # positive edges
    value_pos = (node_rep[edge_index[0]] * node_rep[edge_index[1]]).sum(dim=1)
    pos_loss = -torch.log(torch.sigmoid(value_pos) + EPS).mean()

    # negative edges
    neg_edge_index = negative_sampling(edge_index, num_nodes=node_rep.size(0))
    value_neg = (node_rep[neg_edge_index[0]] * node_rep[neg_edge_index[1]]).sum(dim=1)
    neg_loss = -torch.log(1 - torch.sigmoid(value_neg) + EPS).mean()

    loss = pos_loss + neg_loss
    return loss