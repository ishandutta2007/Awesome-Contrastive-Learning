# The All-Gather Communication and Mini-Batch VRAM Wall

The All-Gather bottleneck arises in distributed setups when negative keys must be collected from all GPUs to compute the InfoNCE denominator, leading to high VRAM consumption and high inter-node communication latency.

## Architectural Diagram
```mermaid
flowchart TD
    A[GPU 0 Embeddings] & B[GPU 1 Embeddings] --> C[Synchronous All-Gather]
    C --> D[Massive Global Similarity Matrix]
    D --> E[Out of Memory / Comm Bottleneck]
```

---
[← Back to main README.md](../README.md)
