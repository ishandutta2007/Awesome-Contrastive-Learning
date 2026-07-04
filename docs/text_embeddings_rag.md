# Universal Text Embedding Generation for Enterprise RAG

Enterprise RAG (Retrieval-Augmented Generation) systems use contrastive sentence embedders to represent knowledge documents as dense vectors. Queries retrieve relevant content based on cosine similarity.

## Architectural Diagram
```mermaid
flowchart TD
    A[Corporate Docs] --> B[Contrastive Sentence Encoder]
    B --> C[Dense Vector Index]
    D[User Query] --> E[Vector Search]
    C & E --> F[Retrieved Context]
```

---
[← Back to main README.md](../README.md)
