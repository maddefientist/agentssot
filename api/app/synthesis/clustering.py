import logging

import numpy as np

logger = logging.getLogger("agentssot.synthesis.clustering")


def cluster_items(
    items: list[dict],
    similarity_threshold: float = 0.75,
    min_cluster_size: int = 3,
) -> list[list[dict]]:
    """Cluster items by embedding similarity using greedy agglomerative approach.

    Each item dict must have 'id', 'content', 'embedding' (list[float]).
    Returns list of clusters, each cluster is a list of items.
    Only returns clusters with >= min_cluster_size items.
    """
    if not items:
        return []

    embedded = [it for it in items if it.get("embedding")]
    if len(embedded) < min_cluster_size:
        return []

    clusters: list[list[dict]] = []
    centroids: list[np.ndarray] = []

    for item in embedded:
        emb = np.array(item["embedding"])
        assigned = False

        for i, centroid in enumerate(centroids):
            sim = float(np.dot(emb, centroid) / (np.linalg.norm(emb) * np.linalg.norm(centroid) + 1e-9))
            if sim >= similarity_threshold:
                clusters[i].append(item)
                n = len(clusters[i])
                centroids[i] = centroid * ((n - 1) / n) + emb * (1 / n)
                assigned = True
                break

        if not assigned:
            clusters.append([item])
            centroids.append(emb.copy())

    result = [c for c in clusters if len(c) >= min_cluster_size]
    logger.info(
        "clustering complete",
        extra={"total_items": len(embedded), "clusters_formed": len(result), "threshold": similarity_threshold},
    )
    return result
