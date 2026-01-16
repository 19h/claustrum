"""Content hashing and Merkle tree implementation for incremental updates.

This module provides:
- Content-addressable hashing for functions and binaries
- Merkle tree implementation for efficient change detection
- Hash-based caching utilities
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union, Iterator


def compute_content_hash(
    content: Union[bytes, str, dict[str, Any]],
    algorithm: str = "sha256",
) -> str:
    """Compute a content hash for arbitrary data.

    Args:
        content: Data to hash (bytes, string, or JSON-serializable dict)
        algorithm: Hash algorithm (sha256, sha1, md5)

    Returns:
        Hexadecimal hash string
    """
    hasher = hashlib.new(algorithm)

    if isinstance(content, bytes):
        hasher.update(content)
    elif isinstance(content, str):
        hasher.update(content.encode("utf-8"))
    elif isinstance(content, dict):
        # Deterministic JSON serialization
        serialized = json.dumps(content, sort_keys=True, separators=(",", ":"))
        hasher.update(serialized.encode("utf-8"))
    else:
        raise TypeError(f"Unsupported content type: {type(content)}")

    return hasher.hexdigest()


def compute_file_hash(path: Path, algorithm: str = "sha256") -> str:
    """Compute hash of a file with chunked reading for large files.

    Args:
        path: Path to the file
        algorithm: Hash algorithm

    Returns:
        Hexadecimal hash string
    """
    hasher = hashlib.new(algorithm)

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


@dataclass
class MerkleNode:
    """A node in the Merkle tree."""

    hash: str
    left: Optional["MerkleNode"] = None
    right: Optional["MerkleNode"] = None
    data_key: Optional[str] = None  # Original key for leaf nodes

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class MerkleTree:
    """Merkle tree for efficient change detection in large datasets.

    Used to detect which functions have changed when updating embeddings,
    allowing incremental re-embedding rather than full re-computation.

    Example:
        # Build initial tree
        tree1 = MerkleTree()
        for func_id, ir in functions.items():
            tree1.add(func_id, compute_content_hash(ir))
        tree1.build()

        # After changes, build new tree
        tree2 = MerkleTree()
        for func_id, ir in updated_functions.items():
            tree2.add(func_id, compute_content_hash(ir))
        tree2.build()

        # Find changes
        changed = tree1.diff(tree2)
    """

    def __init__(self):
        self._leaves: dict[str, str] = {}  # key -> hash
        self._root: Optional[MerkleNode] = None
        self._built = False

    def add(self, key: str, content_hash: str) -> None:
        """Add a leaf node to the tree.

        Args:
            key: Unique identifier (e.g., function_id)
            content_hash: Hash of the content
        """
        self._leaves[key] = content_hash
        self._built = False

    def build(self) -> str:
        """Build the Merkle tree and return the root hash.

        Returns:
            Root hash of the tree
        """
        if not self._leaves:
            raise ValueError("Cannot build empty Merkle tree")

        # Create leaf nodes (sorted for determinism)
        sorted_keys = sorted(self._leaves.keys())
        nodes = [MerkleNode(hash=self._leaves[key], data_key=key) for key in sorted_keys]

        # Build tree bottom-up
        while len(nodes) > 1:
            new_level = []

            for i in range(0, len(nodes), 2):
                left = nodes[i]
                right = nodes[i + 1] if i + 1 < len(nodes) else left

                # Combine hashes
                combined = left.hash + right.hash
                parent_hash = compute_content_hash(combined)

                new_level.append(
                    MerkleNode(
                        hash=parent_hash,
                        left=left,
                        right=right if right is not left else None,
                    )
                )

            nodes = new_level

        self._root = nodes[0]
        self._built = True

        return self._root.hash

    @property
    def root_hash(self) -> Optional[str]:
        """Return the root hash if tree is built."""
        return self._root.hash if self._root else None

    def diff(self, other: "MerkleTree") -> set[str]:
        """Find keys that differ between this tree and another.

        Args:
            other: Another Merkle tree to compare against

        Returns:
            Set of keys that have changed (added, removed, or modified)
        """
        if not self._built or not other._built:
            raise ValueError("Both trees must be built before diffing")

        # Quick check: if roots match, no changes
        if self.root_hash == other.root_hash:
            return set()

        # Find all different keys
        changed = set()

        all_keys = set(self._leaves.keys()) | set(other._leaves.keys())

        for key in all_keys:
            my_hash = self._leaves.get(key)
            their_hash = other._leaves.get(key)

            if my_hash != their_hash:
                changed.add(key)

        return changed

    def get_changed_keys(self, old_hashes: dict[str, str]) -> set[str]:
        """Find keys that have changed compared to a previous state.

        Args:
            old_hashes: Dictionary of key -> hash from previous state

        Returns:
            Set of keys that have changed
        """
        changed = set()

        all_keys = set(self._leaves.keys()) | set(old_hashes.keys())

        for key in all_keys:
            new_hash = self._leaves.get(key)
            old_hash = old_hashes.get(key)

            if new_hash != old_hash:
                changed.add(key)

        return changed

    def save_state(self) -> dict[str, str]:
        """Save the current leaf hashes for later comparison.

        Returns:
            Dictionary of key -> hash
        """
        return dict(self._leaves)

    @classmethod
    def load_state(cls, state: dict[str, str]) -> "MerkleTree":
        """Load a Merkle tree from saved state.

        Args:
            state: Dictionary of key -> hash

        Returns:
            New MerkleTree instance
        """
        tree = cls()
        for key, content_hash in state.items():
            tree.add(key, content_hash)
        tree.build()
        return tree


class ContentHashCache:
    """Cache for content hashes with persistence.

    Stores hash -> embedding mappings to avoid recomputing embeddings
    for unchanged functions.
    """

    def __init__(self, cache_path: Optional[Path] = None):
        self._cache: dict[str, Any] = {}
        self._cache_path = cache_path

        if cache_path and cache_path.exists():
            self._load()

    def get(self, content_hash: str) -> Optional[Any]:
        """Retrieve cached data by content hash."""
        return self._cache.get(content_hash)

    def put(self, content_hash: str, data: Any) -> None:
        """Store data with content hash key."""
        self._cache[content_hash] = data

    def contains(self, content_hash: str) -> bool:
        """Check if hash exists in cache."""
        return content_hash in self._cache

    def remove(self, content_hash: str) -> None:
        """Remove entry from cache."""
        self._cache.pop(content_hash, None)

    def clear(self) -> None:
        """Clear all cached data."""
        self._cache.clear()

    def save(self) -> None:
        """Persist cache to disk."""
        if self._cache_path:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            # Note: For embeddings, use numpy or pickle for efficiency
            with open(self._cache_path, "w") as f:
                json.dump(
                    {
                        k: v if isinstance(v, (str, int, float, list)) else str(v)
                        for k, v in self._cache.items()
                    },
                    f,
                )

    def _load(self) -> None:
        """Load cache from disk."""
        if self._cache_path and self._cache_path.exists():
            with open(self._cache_path) as f:
                self._cache = json.load(f)

    def __len__(self) -> int:
        return len(self._cache)

    def __iter__(self) -> Iterator[str]:
        return iter(self._cache)
