# import numpy as np
# import faiss
# import pickle
# import os

# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# from scipy.special import softmax

# from cache.semantic_cache import SemanticCache


# class QueryEngine:

#     def __init__(self):

#         print("Loading FAISS index...")

#         base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

#         index_path = os.path.join(base_dir, "embeddings", "faiss_index.bin")

#         self.index = faiss.read_index(index_path)

#         self.documents = np.load(
#             "embeddings/documents.npy",
#             allow_pickle=True
#         )

#         self.cluster_centers = np.load(
#             "clustering_results/cluster_centers.npy"
#         )

#         print("Loading embedding model...")

#         self.model = SentenceTransformer(
#             "sentence-transformers/all-MiniLM-L6-v2"
#         )

#         self.cache = SemanticCache(
#             global_threshold=0.85,
#             cluster_threshold=0.75
#         )

#         print("Loading PCA model...")

#         with open("clustering_results/pca_model.pkl", "rb") as f:
#             self.pca = pickle.load(f)

#     def get_cluster(self, embedding):

#         reduced_embedding = self.pca.transform(
#             embedding.reshape(1, -1)
#         )

#         similarity = cosine_similarity(
#             reduced_embedding,
#             self.cluster_centers
#         )

#         probs = softmax(similarity, axis=1)

#         cluster = np.argmax(probs)

#         return cluster

#     def search_documents(self, embedding, k=5):

#         D, I = self.index.search(
#             embedding.reshape(1, -1),
#             k
#         )

#         results = []

#         for idx in I[0]:
#             results.append(self.documents[idx])

#         return results

#     def query(self, text):

#         embedding = self.model.encode(text)

#         cluster = self.get_cluster(embedding)

#         hit, entry, score = self.cache.lookup(
#             embedding,
#             cluster
#         )

#         if hit:

#             return {
#                 "query": text,
#                 "cache_hit": True,
#                 "matched_query": entry["query"],
#                 "similarity_score": float(score),
#                 "result": entry["result"],
#                 "dominant_cluster": int(cluster)
#             }

#         docs = self.search_documents(embedding)

#         result = docs[0]

#         self.cache.add(
#             text,
#             embedding,
#             result,
#             cluster
#         )

#         return {
#             "query": text,
#             "cache_hit": False,
#             "matched_query": None,
#             "similarity_score": None,
#             "result": result,
#             "dominant_cluster": int(cluster)
#         }





import numpy as np
import faiss
import pickle
import os

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax

from cache.semantic_cache import SemanticCache


class QueryEngine:

    def __init__(self):

        # Project root directory
        self.base_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )

        print("Loading FAISS index...")

        index_path = os.path.join(
            self.base_dir,
            "embeddings",
            "faiss_index.bin"
        )

        self.index = faiss.read_index(index_path)

        # Load documents
        documents_path = os.path.join(
            self.base_dir,
            "embeddings",
            "documents.npy"
        )

        self.documents = np.load(
            documents_path,
            allow_pickle=True
        )

        # Load cluster centers
        cluster_centers_path = os.path.join(
            self.base_dir,
            "clustering_results",
            "cluster_centers.npy"
        )

        self.cluster_centers = np.load(cluster_centers_path)

        print("Loading embedding model...")

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize semantic cache
        self.cache = SemanticCache(
            global_threshold=0.85,
            cluster_threshold=0.75
        )

        print("Loading PCA model...")

        pca_path = os.path.join(
            self.base_dir,
            "clustering_results",
            "pca_model.pkl"
        )

        with open(pca_path, "rb") as f:
            self.pca = pickle.load(f)

    def get_cluster(self, embedding):

        reduced_embedding = self.pca.transform(
            embedding.reshape(1, -1)
        )

        similarity = cosine_similarity(
            reduced_embedding,
            self.cluster_centers
        )

        probs = softmax(similarity, axis=1)

        cluster = np.argmax(probs)

        return cluster

    def search_documents(self, embedding, k=5):

        D, I = self.index.search(
            embedding.reshape(1, -1),
            k
        )

        results = []

        for idx in I[0]:
            results.append(self.documents[idx])

        return results

    def query(self, text):

        embedding = self.model.encode(text)

        cluster = self.get_cluster(embedding)

        hit, entry, score = self.cache.lookup(
            embedding,
            cluster
        )

        if hit:

            return {
                "query": text,
                "cache_hit": True,
                "matched_query": entry["query"],
                "similarity_score": float(score),
                "result": entry["result"],
                "dominant_cluster": int(cluster)
            }

        docs = self.search_documents(embedding)

        result = docs[0]

        self.cache.add(
            text,
            embedding,
            result,
            cluster
        )

        return {
            "query": text,
            "cache_hit": False,
            "matched_query": None,
            "similarity_score": None,
            "result": result,
            "dominant_cluster": int(cluster)
        }