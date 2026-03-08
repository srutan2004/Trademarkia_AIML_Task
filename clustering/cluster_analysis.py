# import numpy as np

# # Load data
# documents = np.load("../embeddings/documents.npy", allow_pickle=True)
# membership = np.load("../clustering_results/membership_matrix.npy")

# clusters = membership.shape[0]
# docs = membership.shape[1]

# print("Clusters:", clusters)
# print("Documents:", docs)

# # Determine dominant cluster
# dominant_clusters = np.argmax(membership, axis=0)

# print("\nCluster distribution:\n")

# for c in range(clusters):

#     indices = np.where(dominant_clusters == c)[0]

#     print(f"Cluster {c} -> {len(indices)} documents")




# print("\nSample documents from clusters:\n")

# for cluster_id in range(clusters):

#     print("\n==============================")
#     print("Cluster:", cluster_id)
#     print("==============================")

#     indices = np.where(dominant_clusters == cluster_id)[0][:5]

#     for i in indices:
#         print(documents[i][:200])
#         print("---")


import numpy as np

# -----------------------------
# LOAD DATA
# -----------------------------

documents = np.load("../embeddings/documents.npy", allow_pickle=True)

membership = np.load("../clustering_results/membership_matrix.npy")

clusters = membership.shape[0]
docs = membership.shape[1]

print("\nClusters:", clusters)
print("Documents:", docs)


# -----------------------------
# DOMINANT CLUSTERS
# -----------------------------

dominant_clusters = np.argmax(membership, axis=0)


# -----------------------------
# CLUSTER SIZE DISTRIBUTION
# -----------------------------

print("\nCluster distribution:\n")

for c in range(clusters):

    indices = np.where(dominant_clusters == c)[0]

    print(f"Cluster {c} -> {len(indices)} documents")


# -----------------------------
# SAMPLE DOCUMENTS FROM CLUSTERS
# -----------------------------

print("\nSample documents per cluster:\n")

for cluster_id in range(clusters):

    print("\n==============================")
    print("Cluster:", cluster_id)
    print("==============================")

    indices = np.where(dominant_clusters == cluster_id)[0][:5]

    for i in indices:

        print(documents[i][:200])
        print("---")


# -----------------------------
# BOUNDARY DOCUMENTS
# -----------------------------

print("\nBoundary documents (uncertain cluster membership):\n")

count = 0

for i in range(docs):

    probs = membership[:, i]

    top2 = np.sort(probs)[-2:]

    # If two clusters are very close
    if top2[1] - top2[0] < 0.05:

        print("Document with uncertain cluster membership:")
        print(documents[i][:200])
        print("Cluster probabilities:", probs)
        print()

        count += 1

    if count == 10:
        break