# InfoNCE Loss

InfoNCE is a multi-class classification-based loss function that maximizes mutual information between views. It treats contrastive learning as a classification task with one positive target class and many negative classes.

## Architectural Diagram
```mermaid
flowchart TD
    A[Positive Dot Product] --> B[Exp / Temp Scaling]
    C[Negative Dot Products] --> D[Sum Exp / Temp Scaling]
    B & D --> E[Log Categorical Cross-Entropy]
```

---
[← Back to main README.md](../README.md)
