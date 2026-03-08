import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SemanticCache:

    def __init__(self, global_threshold=0.85, cluster_threshold=0.75):

        # strict similarity threshold
        self.global_threshold = global_threshold

        # relaxed threshold for same-cluster queries
        self.cluster_threshold = cluster_threshold

        self.cache = []

        self.hit_count = 0
        self.miss_count = 0


    def lookup(self, query_embedding, cluster_id):

        best_score = -1
        best_entry = None

        for entry in self.cache:

            sim = cosine_similarity(
                query_embedding.reshape(1, -1),
                entry["embedding"].reshape(1, -1)
            )[0][0]

            if sim > best_score:
                best_score = sim
                best_entry = entry

        if best_entry is None:
            self.miss_count += 1
            return False, None, best_score

        cached_cluster = best_entry["cluster"]

        # Rule 1 — global strict match
        if best_score >= self.global_threshold:

            self.hit_count += 1
            return True, best_entry, best_score

        # Rule 2 — relaxed match inside same cluster
        if (
            cached_cluster == cluster_id and
            best_score >= self.cluster_threshold
        ):

            self.hit_count += 1
            return True, best_entry, best_score

        # Otherwise miss
        self.miss_count += 1
        return False, None, best_score


    def add(self, query, embedding, result, cluster):

        self.cache.append({
            "query": query,
            "embedding": embedding,
            "result": result,
            "cluster": cluster
        })


    def stats(self):

        total = len(self.cache)

        total_requests = self.hit_count + self.miss_count

        hit_rate = (
            self.hit_count / total_requests
            if total_requests > 0
            else 0
        )

        return {
            "total_entries": total,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate
        }


    def clear(self):

        self.cache = []
        self.hit_count = 0
        self.miss_count = 0