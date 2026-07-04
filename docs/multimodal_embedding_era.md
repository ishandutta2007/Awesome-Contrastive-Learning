# The Multi-Modal Joint Embedding Era

The Multi-Modal Joint Embedding Era (CLIP, SigLIP) aligns representations of different modalities (like image and text) in a shared semantic vector space. This allows zero-shot open-vocabulary transfer across vision and language tasks.

## Architectural Diagram
```mermaid
flowchart TD
    A[Image Input] --> B[Image Encoder]
    C[Text Input] --> D[Text Encoder]
    B --> E[Image Embedding]
    D --> F[Text Embedding]
    E & F --> G[Joint Similarity Matrix / Loss]
```

---
[← Back to main README.md](../README.md)
