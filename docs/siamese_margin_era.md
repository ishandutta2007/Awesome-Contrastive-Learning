# The Pairwise Siamese Margin Era

The Pairwise Siamese Margin Era marked the early stages of deep metric learning. Twin neural networks (Siamese towers) shared weights and mapped inputs into an embedding space. Optimization relied on minimizing distance between positive pairs while pushing negative pairs away up to a hardcoded margin.

## Architectural Diagram
```mermaid
flowchart TD
    A[Input 1] --> B[Encoder/Tower 1]
    C[Input 2] --> D[Encoder/Tower 2]
    B --> E[Embedding 1]
    D --> F[Embedding 2]
    E & F --> G[Contrastive Loss with Margin]
```

---
[← Back to main README.md](../README.md)
