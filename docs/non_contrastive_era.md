# The Non-Contrastive Information Maximization Era

Non-Contrastive learning (VICReg, Barlow Twins) eliminated negative samples entirely. They prevent representation collapse using explicit variance, invariance, and covariance regularization constraints on positive views.

## Architectural Diagram
```mermaid
flowchart TD
    A[Image x] --> B[Augmentation t]
    A --> C[Augmentation t']
    B --> D[Encoder f]
    C --> E[Encoder f]
    D --> F[Variance-Covariance Regularization]
    E --> F
```

---
[← Back to main README.md](../README.md)
