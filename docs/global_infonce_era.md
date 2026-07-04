# The Global InfoNCE Matrix Scale Era

The Global InfoNCE Matrix Scale Era (popularized by SimCLR and MoCo) scaled self-supervised representation learning. By utilizing multi-class contrastive cross-entropy, models learned to distinguish positive pairs from large batches of negative pairs.

## Architectural Diagram
```mermaid
flowchart TD
    A[Anchor Image] --> B[Data Augmentations]
    B --> C[View 1 positive]
    B --> D[View 2 positive]
    E[Batch Images] --> F[Negative Views]
    C & D --> G[InfoNCE Loss]
    F --> G
```

---
[← Back to main README.md](../README.md)
