# Triplet Loss

Triplet Loss optimizes the relative distance between an anchor sample, a positive sample (same identity), and a negative sample (different identity). It pulls the positive closer while pushing the negative further away by at least a specified margin.

## Architectural Diagram
```mermaid
flowchart TD
    A[Anchor] --> D[Embedding Space]
    B[Positive] --> D
    C[Negative] --> D
    D --> E[L2 Distance Contrast with Margin]
```

---
[← Back to main README.md](../README.md)
