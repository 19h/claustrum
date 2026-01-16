# CLAUSTRUM

**Cross-ISA Semantic Code Embedding System**

A comprehensive binary analysis platform that creates architecture-neutral embeddings for binary functions, enabling cross-architecture similarity search, semantic clustering, and vulnerability detection.

## Features

- **Multi-ISA Support**: Supports 30+ instruction set architectures including x86, ARM, MIPS, RISC-V, PowerPC, and more
- **Tiered IR Lifting**: Uses VEX, ESIL, and P-Code backends for optimal coverage
- **Architecture-Neutral Embeddings**: Generates embeddings that enable cross-architecture comparison
- **Hierarchical Transformer + GNN**: Advanced model architecture combining transformers with graph neural networks
- **Contrastive Learning**: Pre-trained on cross-compiled code pairs for semantic understanding

## Installation

```bash
# Using uv (recommended)
uv sync

# With development dependencies
uv sync --group dev

# With all optional dependencies
uv sync --all-extras
```

## Quick Start

```python
from claustrum import ClaustrumEmbedder

# Load pretrained model
embedder = ClaustrumEmbedder.from_pretrained("claustrum-base")

# Embed a function
embedding = embedder.embed_function(binary_path, function_addr)

# Search for similar functions
similar = embedder.search(embedding, k=10)
```

## Components

- **lifting**: Multi-backend IR lifting (VEX, ESIL, P-Code)
- **normalization**: Architecture-neutral IR canonicalization  
- **tokenization**: IR tokenization with shared vocabulary (~50K tokens)
- **model**: Hierarchical transformer + GNN architecture
- **training**: Pretraining and contrastive fine-tuning
- **evaluation**: Retrieval and clustering metrics
- **inference**: Production embedding generation
- **tracing**: Execution trace collection and prediction

## Supported Architectures

### Tier 1 (VEX IR - Best Quality)
x86, x86_64, ARM32, ARM64, MIPS32, MIPS64, PPC32, PPC64

### Tier 2 (VEX + ESIL)
RISC-V32, RISC-V64, S390x, SPARC32, SPARC64

### Tier 3 (ESIL/P-Code)
AVR, MSP430, Xtensa, ARC, TriCore, M68K, SH4, Blackfin

### Tier 4 (Limited Support)
Hexagon, LoongArch, VAX, PIC, Z80, 6502, JVM, Dalvik, WebAssembly, eBPF

## Development

```bash
# Run tests
uv run pytest

# Type checking
uv run pyright

# Linting
uv run ruff check src/

# Format code
uv run ruff format src/
```

## License

MIT License
