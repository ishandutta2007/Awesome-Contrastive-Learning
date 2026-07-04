# Sigmoid Loss (SigLIP)

Sigmoid Loss (SigLIP) replaces global Softmax normalization with independent binary sigmoid classification per element, dramatically increasing throughput and batch-size scaling capabilities.

## Architectural Diagram
```mermaid
flowchart TD
    A[Query-Key Matrix] --> B[Element-wise Sigmoid Classification]
    B --> C[Binary Cross-Entropy Loss]
```

---
[← Back to main README.md](../README.md)
