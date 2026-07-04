# Cross-Modal Linear Projections

Cross-Modal Linear Projections map different representation dimensions from independent encoders (like vision and language models) into a unified coordinate space where direct similarity computation can happen.

## Architectural Diagram
```mermaid
flowchart TD
    A[Encoder Output h] --> B[MLP / Linear Projection Head]
    B --> C[Normalized Projection z]
```

---
[← Back to main README.md](../README.md)
