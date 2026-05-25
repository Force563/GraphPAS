# GraphPAS

GraphPAS is a graph neural network framework for pathway activity scoring in single-cell RNA sequencing (scRNA-seq) data. The method integrates heterogeneous graph transformer learning and graph representation modeling to infer robust gene–cell associations and quantify pathway activity at single-cell resolution.

## Overview

Single-cell transcriptomic data are highly sparse and noisy, which limits the robustness of conventional gene set enrichment and pathway activity scoring methods. GraphPAS addresses this challenge by constructing a heterogeneous graph containing genes and cells and learning low-dimensional embeddings through graph neural networks.

The framework consists of:

- Dual autoencoder-based representation learning
- Homogeneous graph attention refinement
- Heterogeneous graph transformer (HGT)
- Gene–cell association reconstruction
- Pathway activity scoring from learned representations

GraphPAS improves the robustness and biological interpretability of pathway activity estimation in scRNA-seq data.

---

## Framework

The GraphPAS workflow includes:

1. Construction of gene–gene and cell–cell graphs
2. Heterogeneous gene–cell bipartite graph modeling
3. Graph transformer-based embedding learning
4. Reconstruction of gene–cell association matrices
5. Pathway activity scoring
6. Downstream analysis:
   - Cell clustering
   - Functional heterogeneity analysis
   - Disease-related pathway characterization

---

