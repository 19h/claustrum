"""FAISS index management for similarity search.

Implements the scalable ANN search from the plan:
- IVF + Product Quantization for 1-10M embeddings
- HNSW for larger scale
- Binary quantization for memory efficiency
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, List, Tuple, Any

import numpy as np


@dataclass
class IndexConfig:
    """Configuration for FAISS index."""

    index_type: str = "ivf_pq"  # "flat", "ivf_pq", "hnsw"
    embedding_dim: int = 256

    # IVF parameters
    nlist: int = 4096  # Number of clusters
    nprobe: int = 64  # Clusters to search

    # PQ parameters
    m: int = 64  # Number of subquantizers
    nbits: int = 8  # Bits per subquantizer

    # HNSW parameters
    hnsw_m: int = 16  # Connections per node
    ef_construction: int = 200
    ef_search: int = 64

    # Reranking
    use_reranking: bool = False
    rerank_k: int = 100


class FAISSIndex:
    """Wrapper for FAISS index with convenient API.

    Supports multiple index types:
    - Flat: Exact search, slow for large scale
    - IVF+PQ: Approximate search with compression
    - HNSW: Graph-based approximate search
    """

    def __init__(self, config: Optional[IndexConfig] = None):
        self.config = config or IndexConfig()
        self._index: Any = None  # FAISS index
        self._trained = False
        self._metadata: dict[int, dict] = {}

    def build(
        self,
        embeddings: np.ndarray,
        metadata: Optional[List[dict]] = None,
        normalize: bool = True,
    ) -> None:
        """Build the index.

        Args:
            embeddings: (N, D) embeddings to index
            metadata: Optional metadata for each embedding
            normalize: L2 normalize embeddings for cosine similarity
        """
        import faiss

        embeddings = embeddings.astype("float32")
        d = embeddings.shape[1]

        if normalize:
            faiss.normalize_L2(embeddings)

        # Build index based on type
        if self.config.index_type == "flat":
            self._index = faiss.IndexFlatIP(d)

        elif self.config.index_type == "ivf_pq":
            # IVF with Product Quantization
            nlist = min(self.config.nlist, embeddings.shape[0] // 10)
            m = min(self.config.m, d // 4)

            quantizer = faiss.IndexFlatIP(d)
            self._index = faiss.IndexIVFPQ(quantizer, d, nlist, m, self.config.nbits)

            # Train on data
            self._index.train(embeddings)
            self._trained = True

            # Set search parameters
            self._index.nprobe = self.config.nprobe

        elif self.config.index_type == "hnsw":
            self._index = faiss.IndexHNSWFlat(d, self.config.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            self._index.hnsw.efConstruction = self.config.ef_construction
            self._index.hnsw.efSearch = self.config.ef_search

        elif self.config.index_type == "ivf_hnsw":
            # IVF with HNSW quantizer
            nlist = min(self.config.nlist, embeddings.shape[0] // 10)

            quantizer = faiss.IndexHNSWFlat(d, self.config.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            self._index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            self._index.train(embeddings)
            self._trained = True

        else:
            raise ValueError(f"Unknown index type: {self.config.index_type}")

        # Add embeddings
        self._index.add(embeddings)

        # Store metadata
        if metadata:
            self._metadata = {i: m for i, m in enumerate(metadata)}

    def search(
        self,
        queries: np.ndarray,
        k: int = 10,
        normalize: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for nearest neighbors.

        Args:
            queries: (N, D) query embeddings
            k: Number of neighbors to return
            normalize: L2 normalize queries

        Returns:
            (scores, indices) both (N, k)
        """
        import faiss

        if self._index is None:
            raise ValueError("Index not built")

        queries = queries.astype("float32")
        if normalize:
            faiss.normalize_L2(queries)

        scores, indices = self._index.search(queries, k)

        return scores, indices

    def search_with_metadata(
        self,
        queries: np.ndarray,
        k: int = 10,
    ) -> List[List[dict]]:
        """Search and return results with metadata.

        Args:
            queries: (N, D) query embeddings
            k: Number of neighbors

        Returns:
            List of result lists, each result has 'index', 'score', 'metadata'
        """
        scores, indices = self.search(queries, k)

        results = []
        for i in range(len(queries)):
            query_results = []
            for score, idx in zip(scores[i], indices[i]):
                if idx < 0:
                    continue
                result = {
                    "index": int(idx),
                    "score": float(score),
                }
                if idx in self._metadata:
                    result["metadata"] = self._metadata[idx]
                query_results.append(result)
            results.append(query_results)

        return results

    def save(self, path: Union[str, Path]) -> None:
        """Save index to disk.

        Args:
            path: Directory to save index
        """
        import faiss
        import json

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(path / "index.faiss"))

        # Save config
        config_dict = {
            "index_type": self.config.index_type,
            "embedding_dim": self.config.embedding_dim,
            "nlist": self.config.nlist,
            "nprobe": self.config.nprobe,
            "m": self.config.m,
            "nbits": self.config.nbits,
            "hnsw_m": self.config.hnsw_m,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config_dict, f)

        # Save metadata
        if self._metadata:
            with open(path / "metadata.json", "w") as f:
                json.dump({str(k): v for k, v in self._metadata.items()}, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FAISSIndex":
        """Load index from disk.

        Args:
            path: Directory containing saved index

        Returns:
            Loaded FAISSIndex
        """
        import faiss
        import json

        path = Path(path)

        # Load config
        with open(path / "config.json") as f:
            config_dict = json.load(f)
        config = IndexConfig(**config_dict)

        index = cls(config)
        index._index = faiss.read_index(str(path / "index.faiss"))
        index._trained = True

        # Load metadata
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                index._metadata = {int(k): v for k, v in json.load(f).items()}

        return index

    @property
    def ntotal(self) -> int:
        """Number of indexed embeddings."""
        return self._index.ntotal if self._index else 0


def build_ivf_pq_index(
    embeddings: np.ndarray,
    nlist: int = 4096,
    m: int = 64,
    metadata: Optional[List[dict]] = None,
) -> FAISSIndex:
    """Build IVF+PQ index for large-scale search.

    Recommended for 1-10M embeddings.

    Args:
        embeddings: (N, D) embeddings
        nlist: Number of IVF clusters
        m: Number of PQ subquantizers
        metadata: Optional metadata

    Returns:
        Configured FAISSIndex
    """
    config = IndexConfig(
        index_type="ivf_pq",
        embedding_dim=embeddings.shape[1],
        nlist=nlist,
        m=m,
    )

    index = FAISSIndex(config)
    index.build(embeddings, metadata)

    return index


def build_hnsw_index(
    embeddings: np.ndarray,
    M: int = 16,
    ef_construction: int = 200,
    metadata: Optional[List[dict]] = None,
) -> FAISSIndex:
    """Build HNSW index for very large scale.

    Recommended for 10M+ embeddings.

    Args:
        embeddings: (N, D) embeddings
        M: HNSW M parameter
        ef_construction: Construction ef
        metadata: Optional metadata

    Returns:
        Configured FAISSIndex
    """
    config = IndexConfig(
        index_type="hnsw",
        embedding_dim=embeddings.shape[1],
        hnsw_m=M,
        ef_construction=ef_construction,
    )

    index = FAISSIndex(config)
    index.build(embeddings, metadata)

    return index
