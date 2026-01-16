"""Graph Attention Network for CFG-aware aggregation.

Implements the 3-layer GAT from the plan for aggregating basic block
embeddings using control flow graph structure.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from claustrum.model.config import ClaustrumConfig


class GraphAttentionLayer(nn.Module):
    """Single Graph Attention layer.

    Implements multi-head attention over graph neighbors.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat

        # Per-head output dimension
        self.head_dim = out_features // num_heads if concat else out_features

        # Linear transformations for each head
        self.W = nn.Linear(in_features, self.head_dim * num_heads, bias=False)

        # Attention parameters
        self.a_src = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.zeros(num_heads, self.head_dim))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass.

        Args:
            x: Node features (num_nodes, in_features)
            edge_index: Edge indices (2, num_edges)
            return_attention_weights: Return attention weights

        Returns:
            Updated node features
        """
        num_nodes = x.size(0)

        # Linear transformation
        h = self.W(x)  # (num_nodes, head_dim * num_heads)
        h = h.view(num_nodes, self.num_heads, self.head_dim)  # (num_nodes, num_heads, head_dim)

        # Source and target nodes for each edge
        src, dst = edge_index[0], edge_index[1]

        # Compute attention coefficients
        # e_ij = LeakyReLU(a_src * h_i + a_dst * h_j)
        alpha_src = (h * self.a_src).sum(dim=-1)  # (num_nodes, num_heads)
        alpha_dst = (h * self.a_dst).sum(dim=-1)  # (num_nodes, num_heads)

        # For each edge, compute attention score
        e = alpha_src[src] + alpha_dst[dst]  # (num_edges, num_heads)
        e = self.leaky_relu(e)

        # Softmax over neighbors
        # Use scatter to compute softmax normalization per destination node
        e_max = torch.zeros(num_nodes, self.num_heads, device=x.device)
        e_max.scatter_reduce_(
            0, dst.unsqueeze(-1).expand_as(e), e, reduce="amax", include_self=False
        )
        e = e - e_max[dst]

        alpha = torch.exp(e)
        alpha_sum = torch.zeros(num_nodes, self.num_heads, device=x.device)
        alpha_sum.scatter_add_(0, dst.unsqueeze(-1).expand_as(alpha), alpha)
        alpha = alpha / (alpha_sum[dst] + 1e-10)

        alpha = self.dropout(alpha)

        # Aggregate messages
        # out_i = sum_j alpha_ij * h_j
        alpha_expanded = alpha.unsqueeze(-1)  # (num_edges, num_heads, 1)
        h_src = h[src]  # (num_edges, num_heads, head_dim)
        messages = alpha_expanded * h_src  # (num_edges, num_heads, head_dim)

        out = torch.zeros(num_nodes, self.num_heads, self.head_dim, device=x.device)
        out.scatter_add_(0, dst.unsqueeze(-1).unsqueeze(-1).expand_as(messages), messages)

        if self.concat:
            out = out.view(num_nodes, -1)  # (num_nodes, num_heads * head_dim)
        else:
            out = out.mean(dim=1)  # (num_nodes, head_dim)

        if return_attention_weights:
            return out, (edge_index, alpha)

        return (out,)


class CFGAttentionNetwork(nn.Module):
    """3-layer Graph Attention Network for CFG aggregation.

    Takes basic block embeddings and CFG structure, produces
    function-level embeddings that capture both sequential
    and structural properties.
    """

    def __init__(self, config: ClaustrumConfig):
        super().__init__()

        self.config = config

        # Input projection (from transformer hidden to GNN hidden)
        self.input_proj = nn.Linear(config.hidden_size, config.gnn_hidden_size)

        # GAT layers
        self.layers = nn.ModuleList()

        for i in range(config.gnn_num_layers):
            in_dim = config.gnn_hidden_size
            out_dim = config.gnn_hidden_size
            concat = i < config.gnn_num_layers - 1  # Don't concat on last layer

            self.layers.append(
                GraphAttentionLayer(
                    in_features=in_dim,
                    out_features=out_dim,
                    num_heads=config.gnn_num_heads,
                    dropout=config.gnn_dropout,
                    concat=concat,
                )
            )

        # Output projection
        self.output_proj = nn.Linear(config.gnn_hidden_size, config.embedding_size)

        # Layer norm and dropout
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(config.gnn_hidden_size) for _ in range(config.gnn_num_layers)]
        )
        self.dropout = nn.Dropout(config.gnn_dropout)

    def forward(
        self,
        block_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        return_block_embeddings: bool = False,
    ) -> dict:
        """Forward pass through GAT.

        Args:
            block_embeddings: Basic block embeddings (num_blocks, hidden_size)
            edge_index: CFG edges (2, num_edges)
            batch: Batch assignment for each block (for batched graphs)
            return_block_embeddings: Return per-block embeddings

        Returns:
            Dictionary with function embeddings
        """
        # Project input
        h = self.input_proj(block_embeddings)
        h = F.gelu(h)

        # GAT layers with residual connections
        for i, (layer, ln) in enumerate(zip(self.layers, self.layer_norms)):
            h_residual = h
            h = layer(h, edge_index)[0]
            h = ln(h)
            h = F.gelu(h)
            h = self.dropout(h)

            if h.shape == h_residual.shape:
                h = h + h_residual

        block_out = h

        # Graph-level pooling
        if batch is not None:
            # Batched graphs - pool per graph
            num_graphs = batch.max().item() + 1
            graph_embeddings = torch.zeros(num_graphs, h.size(-1), device=h.device)

            # Mean pooling per graph
            counts = torch.zeros(num_graphs, device=h.device)
            graph_embeddings.scatter_add_(0, batch.unsqueeze(-1).expand_as(h), h)
            counts.scatter_add_(0, batch, torch.ones_like(batch, dtype=torch.float))
            graph_embeddings = graph_embeddings / counts.unsqueeze(-1).clamp(min=1)
        else:
            # Single graph - global mean pooling
            graph_embeddings = h.mean(dim=0, keepdim=True)

        # Output projection
        function_embeddings = self.output_proj(graph_embeddings)
        function_embeddings = F.normalize(function_embeddings, p=2, dim=-1)

        result = {
            "function_embeddings": function_embeddings,
        }

        if return_block_embeddings:
            result["block_embeddings"] = block_out

        return result


def build_edge_index(cfg_edges: list[tuple[int, int]], device: torch.device) -> torch.Tensor:
    """Convert CFG edge list to edge_index tensor.

    Args:
        cfg_edges: List of (src, dst) block ID pairs
        device: Target device

    Returns:
        Edge index tensor (2, num_edges)
    """
    if not cfg_edges:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    src = [e[0] for e in cfg_edges]
    dst = [e[1] for e in cfg_edges]

    return torch.tensor([src, dst], dtype=torch.long, device=device)


def batch_graphs(
    block_embeddings_list: list[torch.Tensor],
    edge_index_list: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batch multiple graphs into a single graph.

    Args:
        block_embeddings_list: List of block embedding tensors
        edge_index_list: List of edge index tensors

    Returns:
        (batched_embeddings, batched_edge_index, batch_assignment)
    """
    device = block_embeddings_list[0].device

    batched_embeddings = []
    batched_edges = []
    batch = []

    node_offset = 0

    for i, (embeddings, edges) in enumerate(zip(block_embeddings_list, edge_index_list)):
        num_nodes = embeddings.size(0)

        batched_embeddings.append(embeddings)

        # Offset edge indices
        if edges.numel() > 0:
            batched_edges.append(edges + node_offset)

        batch.extend([i] * num_nodes)
        node_offset += num_nodes

    batched_embeddings = torch.cat(batched_embeddings, dim=0)

    if batched_edges:
        batched_edge_index = torch.cat(batched_edges, dim=1)
    else:
        batched_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

    batch_tensor = torch.tensor(batch, dtype=torch.long, device=device)

    return batched_embeddings, batched_edge_index, batch_tensor
