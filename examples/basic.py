"""Minimal example using embeddings that are already available to retrieval."""

import numpy as np

from mempursuit import select


query_embedding = np.asarray([1.0, 1.0, 0.0])
candidate_embeddings = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.99, -0.05, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)

result = select(query_embedding, candidate_embeddings, budget=2)
print("selected candidate indices:", result.selected_indices.tolist())
print("final composition weights:", result.weights.tolist())
