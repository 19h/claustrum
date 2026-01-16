"""End-to-end embedder for binary functions.

ClaustrumEmbedder provides a high-level API for:
- Loading pretrained models
- Embedding binary functions
- Similarity search
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, List, cast

import torch
import torch.nn.functional as F

from claustrum.lifting import TieredLifter
from claustrum.normalization import IRNormalizer
from claustrum.tokenization import IRTokenizer
from claustrum.model import ClaustrumEncoder, ClaustrumConfig


class ClaustrumEmbedder:
    """End-to-end function embedding pipeline.

    Provides a simple API for embedding binary functions:
    1. Lift binary to IR
    2. Normalize to architecture-neutral form
    3. Tokenize
    4. Generate embedding

    Example:
        >>> embedder = ClaustrumEmbedder.from_pretrained("claustrum-base")
        >>> embedding = embedder.embed_function("/path/to/binary", 0x1000)
        >>> similar = embedder.search(embedding, k=10)
    """

    def __init__(
        self,
        model: ClaustrumEncoder,
        tokenizer: IRTokenizer,
        lifter: Optional[TieredLifter] = None,
        normalizer: Optional[IRNormalizer] = None,
        device: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.lifter = lifter or TieredLifter()
        self.normalizer = normalizer or IRNormalizer()

        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = self.model.to(self.device)
        self.model.eval()

        # Index for similarity search
        self._index = None
        self._index_metadata = {}

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: Optional[str] = None,
    ) -> "ClaustrumEmbedder":
        """Load pretrained embedder.

        Args:
            model_path: Path to model directory or model name
            device: Target device ("cpu", "cuda", "cuda:0", etc.)

        Returns:
            Configured ClaustrumEmbedder
        """
        path = Path(model_path)

        # Load config
        if (path / "config.json").exists():
            config = ClaustrumConfig.from_pretrained(model_path)
        else:
            config = ClaustrumConfig()

        # Load model
        model = ClaustrumEncoder(config)

        weights_path = path / "pytorch_model.bin"
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)

        # Create tokenizer
        tokenizer = IRTokenizer()

        return cls(model, tokenizer, device=device)

    @torch.no_grad()
    def embed_function(
        self,
        binary_path: Union[str, Path],
        function_addr: int,
        function_size: Optional[int] = None,
        isa: Optional[str] = None,
    ) -> torch.Tensor:
        """Embed a single function.

        Args:
            binary_path: Path to binary file
            function_addr: Address of function start
            function_size: Optional function size hint
            isa: Optional ISA hint

        Returns:
            (embedding_dim,) tensor
        """
        binary_path = Path(binary_path)

        # Lift function to IR
        lifted = self.lifter.lift_function(binary_path, function_addr, function_size, isa)

        # Normalize
        normalized = self.normalizer.normalize(lifted)

        # Tokenize
        tokenized = self.tokenizer.prepare_for_model(normalized, return_tensors="pt")

        # Move to device
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)

        # Generate embedding
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = outputs["pooler_output"].squeeze(0)

        return embedding.cpu()

    @torch.no_grad()
    def embed_functions(
        self,
        binary_path: Union[str, Path],
        function_addrs: List[int],
        batch_size: int = 32,
        isa: Optional[str] = None,
    ) -> torch.Tensor:
        """Embed multiple functions from the same binary.

        Args:
            binary_path: Path to binary file
            function_addrs: List of function addresses
            batch_size: Batch size for inference
            isa: Optional ISA hint

        Returns:
            (num_functions, embedding_dim) tensor
        """
        binary_path = Path(binary_path)
        embeddings = []

        for i in range(0, len(function_addrs), batch_size):
            batch_addrs = function_addrs[i : i + batch_size]
            batch_embeddings = []

            for addr in batch_addrs:
                try:
                    emb = self.embed_function(binary_path, addr, isa=isa)
                    batch_embeddings.append(emb)
                except Exception as e:
                    # Handle lifting failures gracefully
                    print(f"Failed to embed function at {addr:#x}: {e}")
                    # Use zero embedding as placeholder
                    batch_embeddings.append(torch.zeros(self.model.config.embedding_size))

            embeddings.extend(batch_embeddings)

        return torch.stack(embeddings)

    @torch.no_grad()
    def embed_normalized(
        self,
        normalized_functions: List,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Embed pre-normalized functions.

        Useful when you've already done lifting and normalization.

        Args:
            normalized_functions: List of NormalizedFunction
            batch_size: Batch size for inference

        Returns:
            (num_functions, embedding_dim) tensor
        """
        embeddings = []

        for i in range(0, len(normalized_functions), batch_size):
            batch = normalized_functions[i : i + batch_size]

            # Tokenize batch
            tokenized = self.tokenizer.batch_tokenize(batch, return_tensors="pt")

            # Move to device (batch_tokenize returns tensors when return_tensors="pt")
            input_ids = cast(torch.Tensor, tokenized["input_ids"]).to(self.device)
            attention_mask = cast(torch.Tensor, tokenized["attention_mask"]).to(self.device)

            # Generate embeddings
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            batch_embeddings = outputs["pooler_output"]

            embeddings.append(batch_embeddings.cpu())

        return torch.cat(embeddings, dim=0)

    def build_index(
        self,
        embeddings: torch.Tensor,
        metadata: Optional[List[dict]] = None,
        index_type: str = "flat",
    ) -> None:
        """Build FAISS index for similarity search.

        Args:
            embeddings: (N, D) embeddings to index
            metadata: Optional list of metadata dicts for each embedding
            index_type: Type of index ("flat", "ivf", "hnsw")
        """
        import faiss

        embeddings_np = embeddings.numpy().astype("float32")
        d = embeddings_np.shape[1]

        if index_type == "flat":
            self._index = faiss.IndexFlatIP(d)  # Inner product (cosine with normalized)
        elif index_type == "ivf":
            # IVF with Product Quantization
            nlist = min(4096, embeddings_np.shape[0] // 10)
            m = min(64, d // 8)  # Number of subquantizers
            quantizer = faiss.IndexFlatIP(d)
            self._index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
            self._index.train(embeddings_np)
        elif index_type == "hnsw":
            # HNSW index
            M = 16  # Number of connections
            self._index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings_np)
        self._index.add(embeddings_np)

        # Store metadata
        if metadata:
            self._index_metadata = {i: m for i, m in enumerate(metadata)}

    def search(
        self,
        query: torch.Tensor,
        k: int = 10,
        return_metadata: bool = True,
    ) -> List[dict]:
        """Search for similar functions.

        Args:
            query: (D,) or (N, D) query embedding(s)
            k: Number of results to return
            return_metadata: Include metadata in results

        Returns:
            List of result dicts with 'index', 'score', and optionally 'metadata'
        """
        if self._index is None:
            raise ValueError("Index not built. Call build_index first.")

        import faiss

        # Handle single query
        if query.dim() == 1:
            query = query.unsqueeze(0)

        query_np = query.numpy().astype("float32")
        faiss.normalize_L2(query_np)

        # Search
        scores, indices = self._index.search(query_np, k)

        # Format results
        results = []
        for i in range(len(query)):
            query_results = []
            for score, idx in zip(scores[i], indices[i]):
                if idx < 0:  # Invalid index
                    continue
                result = {
                    "index": int(idx),
                    "score": float(score),
                }
                if return_metadata and idx in self._index_metadata:
                    result["metadata"] = self._index_metadata[idx]
                query_results.append(result)
            results.append(query_results)

        # Return single list for single query
        if len(results) == 1:
            return results[0]
        return results

    def save_index(self, path: Union[str, Path]) -> None:
        """Save FAISS index to file.

        Args:
            path: Path to save index
        """
        import faiss
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(path / "index.faiss"))

        if self._index_metadata:
            with open(path / "metadata.json", "w") as f:
                json.dump(self._index_metadata, f)

    def load_index(self, path: Union[str, Path]) -> None:
        """Load FAISS index from file.

        Args:
            path: Path to saved index
        """
        import faiss
        import json

        path = Path(path)

        self._index = faiss.read_index(str(path / "index.faiss"))

        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self._index_metadata = {int(k): v for k, v in json.load(f).items()}
