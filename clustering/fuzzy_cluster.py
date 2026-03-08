import numpy as np
import os

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# CONFIG
# -----------------------------

CLUSTERS = 15

EMBEDDINGS_PATH = "../embeddings/embeddings.npy"
OUTPUT_DIR = "../clustering_results"

# -----------------------------
# LOAD EMBEDDINGS
# -----------------------------

print("\nLoading embeddings...\n")

embeddings = np.load(EMBEDDINGS_PATH)

print("Original shape:", embeddings.shape)

# -----------------------------
# PCA DIMENSION REDUCTION
# -----------------------------

print("\nRunning PCA...\n")

pca = PCA(n_components=50)
reduced_embeddings = pca.fit_transform(embeddings)

import pickle

with open("../clustering_results/pca_model.pkl", "wb") as f:
    pickle.dump(pca, f)

print("Reduced shape:", reduced_embeddings.shape)

# -----------------------------
# KMEANS CLUSTERING
# -----------------------------

print("\nRunning KMeans clustering...\n")

kmeans = KMeans(n_clusters=CLUSTERS, random_state=42, n_init=10)

kmeans.fit(reduced_embeddings)

cluster_centers = kmeans.cluster_centers_

print("KMeans clustering complete.")

# -----------------------------
# COMPUTE FUZZY MEMBERSHIPS
# -----------------------------

print("\nComputing fuzzy memberships...\n")

from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax

similarity = cosine_similarity(reduced_embeddings, cluster_centers)

membership = softmax(similarity, axis=1)

membership = membership.T

# -----------------------------
# DOMINANT CLUSTER
# -----------------------------

dominant_clusters = np.argmax(membership, axis=0)

# -----------------------------
# SAVE RESULTS
# -----------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

np.save(f"{OUTPUT_DIR}/cluster_centers.npy", cluster_centers)
np.save(f"{OUTPUT_DIR}/membership_matrix.npy", membership)
np.save(f"{OUTPUT_DIR}/dominant_clusters.npy", dominant_clusters)

print("\nSaved clustering results.")