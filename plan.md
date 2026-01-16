# Cross-ISA Semantic Code Embedding: A Complete Implementation Blueprint

**Functions performing identical computations can be embedded identically across 30+ instruction set architectures by lifting binaries to architecture-neutral intermediate representations, then training transformer-based models with contrastive objectives on paired cross-compilation data.** The approach leverages your existing C→multi-ISA compilation pipeline as the primary supervision signal, treating functions compiled from the same source as positive pairs. State-of-the-art systems like Trex, jTrans, and VexIR2Vec demonstrate that combining IR-based normalization with contrastive learning achieves **7-40% improvements** over syntactic approaches for cross-architecture matching. This report provides specific architectural choices, loss formulations, and deployment strategies for building such a system.

## The most promising architecture combines IR lifting with hierarchical transformers

The research landscape reveals a clear evolutionary trajectory from graph-based approaches (Gemini, 2017) through sequence models (SAFE, 2019) to transformer-based systems achieving current state-of-the-art performance. **Trex** (Pei et al., IEEE TDSC 2023) pioneered execution-semantics learning through micro-traces and VEX IR, achieving +7.8% cross-architecture, +7.2% cross-optimization, and +14.3% cross-obfuscation improvements. **jTrans** (Wang et al., ISSTA 2022) demonstrated that encoding control flow structure into transformer attention yields +30.5% gains over prior work. **VexIR2Vec** achieved the strongest cross-architecture results through compiler-inspired IR normalization, outperforming baselines by **40% in cross-optimization** and **21% in cross-architecture** settings.

For your 30+ ISA requirement, a hierarchical architecture proves optimal. At the instruction level, lift binaries to VEX IR (supporting x86, AMD64, ARM32/64, MIPS32/64, PPC, S390X) or ESIL (supporting 20+ architectures including embedded ISAs like AVR, MSP430). Apply normalization passes—constant propagation, dead code elimination, register abstraction—to canonicalize the representation. Feed normalized IR sequences through a **12-layer BERT-style transformer** to produce basic block embeddings, then aggregate through a **3-layer Graph Attention Network** operating on the control flow graph. This captures both sequential instruction semantics and structural control flow properties.

The recommended model dimensions are **768 hidden units** with **128-256 dimensional final embeddings**. Larger embeddings (512+) provide marginal quality improvements but significantly increase storage and search costs. For your function-level granularity requirement, use attention pooling over basic block embeddings weighted by block centrality in the CFG.

## Intermediate representation lifting solves the cross-ISA vocabulary explosion

Raw assembly tokenization fails for 30+ ISAs because vocabulary size explodes and semantically equivalent operations receive unrelated tokens. Research from NDSS BAR 2025 confirms that **full-instruction tokenization** outperforms subword methods (BPE, WordPiece) for assembly, but this requires architecture-specific tokenizers. The solution is lifting to architecture-neutral IR before tokenization.

**Ghidra P-Code** offers the broadest architecture coverage through SLEIGH processor specifications—covering all 30 ISAs in your pipeline including exotic targets like Blackfin, SuperH, and VAX. P-Code provides medium-level semantics suitable for embedding while maintaining formal correctness. **ESIL** (radare2's stack-based IL) offers excellent ML integration through simple text representation and supports embedded architectures often missing from other lifters. **VEX IR** (from angr/Valgrind) has the strongest track record in academic research but supports fewer architectures (~8 primary ISAs).

For your pipeline, implement a **tiered lifting strategy**: use VEX IR for Tier 1 architectures (x86, ARM, MIPS, RISC-V, PPC) where it excels, fall back to ESIL or P-Code for less common ISAs (AVR, MSP430, Hexagon, LoongArch). Store raw bytes alongside lifted IR to enable hybrid approaches.

Normalize the lifted IR with these transformations:
- **Immediates**: Preserve small constants (0-255), replace larger values with IMM16/IMM32/IMM64 tokens
- **Registers**: Abstract to role-based classes (ARG0-ARG7, RET, SP, FP, GENERAL) rather than architecture-specific names
- **Memory operands**: Canonicalize to MEM[base+offset] form with access size annotations (MEM8, MEM32, MEM64)
- **Function calls**: Replace targets with FUNC token
- **Addresses**: Replace absolute addresses with ADDR token

This normalization reduces vocabulary to approximately **50,000 shared tokens** plus ~1,000 per-ISA extensions for architecture-specific opcodes that resist semantic mapping.

## Contrastive learning with multi-positive formulation exploits your compilation pipeline

Your C→multi-ISA compilation pipeline provides natural supervision through **compilation provenance**: functions compiled from identical source code are semantic equivalents regardless of target ISA. This enables multi-positive contrastive learning where each source function generates 30+ positive pairs.

The **Multi-Positive InfoNCE** loss formulation extends standard contrastive learning to handle multiple positives per anchor:

```
L = -∑ᵢ (1/|P(i)|) × ∑_{p∈P(i)} log[exp(zᵢ·zₚ/τ) / ∑_{a∈A(i)} exp(zᵢ·zₐ/τ)]
```

Where P(i) is the set of all ISA variants from the same source function, A(i) is all samples in the batch, and τ is temperature (**0.07-0.1** works best for fine-grained code similarity). This formulation treats all 30 ISA variants as positives simultaneously, forcing the model to learn ISA-invariant representations.

**Batch construction strategy**: For each batch of N source functions, include all ISA variants, yielding N×30 samples. Use labels to identify positive relationships (same source → same label). With batch size N=256 and 30 ISAs, you get 7,680 samples per batch—sufficient negative diversity without explicit hard negative mining in early training.

**Hard negative mining** becomes critical for later training stages. Sample negatives proportionally to similarity scores using temperature β:

```
q^β(x⁻|x) ∝ exp(β × sim(f(x), f(x⁻)))
```

Start with β=0 (random negatives), progress to β=2.0 over training. This curriculum prevents mode collapse while eventually forcing discrimination between semantically similar but non-equivalent functions.

**Curriculum learning for ISA pairs**: Begin training with architecturally similar pairs (ARM32↔ARM64, x86↔x64), progress to RISC↔RISC pairs (ARM↔MIPS), then CISC↔RISC (x86↔ARM), finally register↔stack machines (all↔JVM, all↔WASM). This ordering reflects genuine difficulty—stack machines like JVM and WebAssembly have fundamentally different execution models requiring the model to learn deeper semantic invariants.

## Self-supervised pretraining accelerates convergence and improves generalization

Before contrastive fine-tuning, pretrain on unlabeled binaries using **Masked Instruction Modeling** adapted from PalmTree (Li et al., CCS 2021). Mask 15% of IR tokens (80% [MASK], 10% random, 10% unchanged) and predict original tokens. Add two auxiliary tasks from PalmTree research:

**Context Window Prediction (CWP)**: Given two instructions, predict whether they appear in the same basic block. This teaches local control flow structure without CFG construction overhead.

**Def-Use Prediction (DUP)**: Given two instructions, predict whether the first defines a value the second uses. This captures data flow dependencies critical for semantic understanding.

**Execution trace prediction** (from Trex) provides the strongest semantic grounding when feasible. Collect micro-traces through forced execution—executing instruction sequences while ignoring control flow constraints. Include dynamic register and memory values in traces. The masked LM task then predicts both instructions and concrete values, teaching execution semantics directly. However, micro-tracing requires architecture-specific emulators and may not scale to all 30 ISAs initially. Prioritize this for Tier 1 architectures where training data abundance justifies the infrastructure investment.

Pretraining configuration: **50-100 epochs** on large binary corpora, **256 batch size**, **3e-5 learning rate** with linear warmup over 10% of training.

## Semantic grounding through I/O behavior provides strongest equivalence signal

For functions where compilation provenance is unavailable or you want verification beyond syntactic alignment, **input-output behavior** provides ground-truth semantic equivalence. Two functions are semantically equivalent if they produce identical outputs for all inputs.

**Symbolic execution** generates I/O pairs by exploring execution paths. Use angr (supporting x86, ARM, MIPS, PPC) to extract path constraints and generate covering inputs. However, path explosion limits scalability—practical only for small functions or focused analysis.

**Fuzzing-based I/O collection** (IMF-SIM approach) scales better: execute functions with random/mutated inputs, collect output traces, compare behavior signatures. System call sequences provide architecture-independent semantic fingerprints. Memory access patterns persist across compilation more reliably than register values.

**Semantic grounding challenges** require explicit handling:
- **Side effects**: Track heap/stack modifications, I/O operations as part of function signature
- **Control flow divergence**: Compiler optimizations create structurally different implementations of identical logic; semantic comparison handles this
- **Floating point variations**: Use tolerance thresholds (1e-6) for FP comparisons due to precision differences across architectures
- **Non-determinism**: Collect multiple traces, compare statistically

For your clustering goal (grouping raycasting, AES, matrix multiplication regardless of ISA), execution traces are particularly valuable. Functions implementing the same algorithm exhibit similar dynamic behavior even when syntactically divergent.

## Evaluation requires multi-dimensional metrics across diverse benchmarks

Standard retrieval metrics apply to your function similarity task:

**Mean Reciprocal Rank (MRR)**: Measures average reciprocal position of first correct match. Critical for "find the equivalent function" queries. Target: **MRR > 0.7** for production quality.

**Recall@K** for K∈{1, 5, 10, 50}: Measures coverage at different result depths. Recall@1 indicates exact match capability; Recall@10 indicates practical retrieval utility. Target: **Recall@1 > 60%**, **Recall@10 > 85%** for cross-architecture matching.

**Mean Average Precision (mAP)**: Considers ranking quality of all relevant results. Important when multiple correct answers exist (e.g., multiple implementations of the same algorithm).

For your clustering evaluation (grouping by semantic category):

**Adjusted Rand Index (ARI)**: Measures clustering agreement with ground truth, corrected for chance. Range [-1, 1], where 1.0 indicates perfect clustering. Resilient to irrelevant features. Target: **ARI > 0.6** for semantic category clustering.

**Normalized Mutual Information (NMI)**: Measures shared information between predicted and ground truth clusters. Range [0, 1]. Less sensitive to cluster count mismatches than ARI.

**Silhouette Score**: Internal metric (no ground truth needed) measuring cluster cohesion vs. separation. Useful for hyperparameter tuning but can be misleading for non-spherical clusters.

**Benchmark datasets** to evaluate against:
- **BinaryCorp-3M** (jTrans): Most diverse large-scale dataset, 3M+ functions from ArchLinux packages
- **Cisco Dataset**: 7 projects compiled across x86, x64, ARM32/64, MIPS32/64 with GCC and Clang
- **Trex Dataset**: 1.47M functions from 13 OSS projects, specifically designed for cross-architecture evaluation
- **POJ-104**: 52,000 functions from 104 programming problems—useful for algorithm-level semantic clustering
- **BinKit** (KAIST): 371,928 binaries across 1,904 compiler option combinations

Evaluate at multiple pool sizes: 1K (development), 10K (research publication), 100K+ (production readiness). Performance drops significantly with pool size—cross-architecture Recall@1 may fall from 80% at 1K to 40% at 100K.

## Scalability architecture enables production deployment

**Inference optimization** starts with model architecture choices. The 12-layer transformer with 768 hidden dimensions provides strong results while remaining tractable. Apply these optimizations:

- **FP16 inference**: 20-50% speedup, ~50% memory reduction, <1% accuracy loss
- **ONNX export with optimization**: Export via optimum library, enable operator fusion for 20-30% latency reduction
- **TensorRT compilation** (NVIDIA GPUs): Significant acceleration for batch inference
- **INT8 quantization** with calibration: 4× memory reduction, 1-3% accuracy loss acceptable for retrieval

**Approximate Nearest Neighbor search** enables sub-linear retrieval:

For 1-10M embeddings: **IVF + Product Quantization** via FAISS
```
index = faiss.index_factory(128, "IVF4096,PQ64")  # 128-dim, 4096 clusters, 64 subquantizers
```

For 10M+ embeddings: Add GPU acceleration and sharding
```
index = faiss.index_factory(128, "IVF16384,PQ64,RFlat")  # More clusters, reranking
```

**Memory vs. accuracy tradeoffs**:
- Binary quantization (1-bit): 32× compression, 90-96% recall retention
- INT8 scalar quantization: 4× compression, 97-99% recall retention
- Hybrid search: Binary coarse search + INT8 rescoring achieves near-full precision at minimal memory cost

**Incremental updates** for large codebases: Implement Merkle tree-based change detection. Compute content hashes for functions; re-embed only changed functions. Store hash→embedding mappings. This enables efficient updates without full re-indexing.

**Distributed processing** via Ray Data: Parallelize embedding generation across GPUs using actor pools matching available GPU count. Process 100 functions per batch for optimal GPU utilization.

## ISA-specific challenges require careful normalization

**Stack machines** (JVM, Dalvik, WebAssembly) present the greatest challenge—fundamentally different execution models from register machines. WebAssembly's structured control flow (no arbitrary gotos) and operand stack semantics require dedicated handling. Solution: Lift to register-based IR that makes stack operations explicit, treating stack slots as virtual registers.

**VLIW architectures** (Hexagon, IA-64) bundle multiple operations per cycle. Hexagon packets contain 1-4 instructions executing simultaneously. Solution: Unpack bundles into sequential IR instructions while preserving parallelism annotations for semantic analysis.

**Predicated instructions** (ARM32): Any instruction can execute conditionally based on status flags. Convert to explicit conditional branches in IR:
```
ARM:     ADDNE r0, r0, r1        ; executes if Z flag clear
IR:      if (!Z) { r0 = r0 + r1 }
```

**Delay slots** (MIPS, SPARC): Instruction after branch always executes. Critical to model correctly—delay slot may contain semantically important computation. Expand in IR:
```
MIPS:    beq $t0, $t1, target
         addi $t2, $t2, 1        ; delay slot - always executes
IR:      temp = $t2 + 1
         if ($t0 == $t1) goto target
         $t2 = temp
```

**Endianness**: Canonicalize to single endianness in IR. Track original endianness as metadata for cases where byte order affects semantics (network protocols, file formats).

**Calling conventions** vary widely—x86-64 System V uses RDI, RSI, RDX, RCX, R8, R9 for arguments while ARM64 AAPCS uses X0-X7. Abstract to role-based tokens (ARG0, ARG1, ..., RET) rather than concrete register names.

## Complete implementation roadmap

**Phase 1 (Months 1-3): Foundation**
- Implement IR lifting pipeline using Ghidra P-Code (broadest coverage) with ESIL fallback
- Build normalization engine with constant propagation, register abstraction, memory canonicalization
- Create tokenizer with ~50K shared vocabulary
- Compile evaluation dataset from POJ-104 and BinKit sources to all 30 target ISAs

**Phase 2 (Months 3-5): Model Development**
- Train 12-layer BERT encoder on masked instruction modeling task
- Add CWP and DUP auxiliary tasks
- Pretrain for 50-100 epochs on binary corpus covering all ISAs

**Phase 3 (Months 5-7): Contrastive Fine-tuning**
- Implement multi-positive InfoNCE loss
- Fine-tune on paired cross-compilation data
- Add curriculum learning (similar ISAs → dissimilar ISAs)
- Implement hard negative mining with progressive difficulty

**Phase 4 (Months 7-9): Integration and Optimization**
- Export to ONNX with INT8 quantization
- Build FAISS index with IVF+PQ
- Implement incremental update system
- Deploy evaluation pipeline across all benchmarks

**Target metrics for production system**:
- Cross-ISA Recall@1: >60% on 100K pool
- Cross-ISA MRR: >0.7 on 10K pool
- Semantic clustering ARI: >0.5 for algorithm categories
- Inference: <10ms per function embedding
- Index: <100GB for 10M function corpus

This blueprint synthesizes findings from Asm2Vec (IEEE S&P 2019), SAFE (DIMVA 2019), Gemini (CCS 2017), InnerEye (NDSS 2019), PalmTree (CCS 2021), Trex (IEEE TDSC 2023), jTrans (ISSTA 2022), UniASM (Neurocomputing 2025), VexIR2Vec (2024), and CLAP (ISSTA 2024), representing the current frontier of cross-ISA binary similarity research.
