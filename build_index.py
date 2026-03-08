import os
import json
import numpy as np
import faiss

from tqdm import tqdm
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer

from utils.preprocess import clean_text


# -----------------------------
# CONFIGURATION
# -----------------------------

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = "embeddings"
MIN_DOC_LENGTH = 50


# -----------------------------
# LOAD DATASET
# -----------------------------

print("\nLoading 20 Newsgroups dataset...\n")

dataset = fetch_20newsgroups(
    subset="all",
    remove=("headers", "footers", "quotes")
)

documents = dataset.data
targets = dataset.target
categories = dataset.target_names

print("Total raw documents:", len(documents))


# -----------------------------
# PREPROCESS DOCUMENTS
# -----------------------------

print("\nCleaning documents...\n")

clean_docs = []
doc_ids = []
doc_labels = []

for i, doc in enumerate(documents):

    cleaned = clean_text(doc)

    # Filter extremely short documents
    if len(cleaned) < MIN_DOC_LENGTH:
        continue

    clean_docs.append(cleaned)
    doc_ids.append(i)
    doc_labels.append(categories[targets[i]])

print("Documents after cleaning:", len(clean_docs))


# -----------------------------
# LOAD EMBEDDING MODEL
# -----------------------------

print("\nLoading embedding model...\n")

model = SentenceTransformer(EMBEDDING_MODEL)


# -----------------------------
# GENERATE EMBEDDINGS
# -----------------------------

print("\nGenerating embeddings...\n")

embeddings = model.encode(
    clean_docs,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("Embedding shape:", embeddings.shape)

dimension = embeddings.shape[1]


# -----------------------------
# BUILD FAISS INDEX
# -----------------------------

print("\nBuilding FAISS vector index...\n")

"""
We use IndexFlatIP (Inner Product) with normalized embeddings.

Since embeddings are normalized, inner product is equivalent to
cosine similarity, which is better suited for semantic search.
"""

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print("Vectors stored in FAISS:", index.ntotal)


# -----------------------------
# SAVE OUTPUT
# -----------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\nSaving files...\n")

# Save FAISS index
faiss.write_index(index, f"{OUTPUT_DIR}/faiss_index.bin")

# Save embeddings (needed for clustering)
np.save(f"{OUTPUT_DIR}/embeddings.npy", embeddings)

# Save cleaned documents
np.save(f"{OUTPUT_DIR}/documents.npy", np.array(clean_docs))

# Save metadata
metadata = []

for i in range(len(clean_docs)):
    metadata.append({
        "doc_id": int(doc_ids[i]),
        "category": doc_labels[i]
    })

with open(f"{OUTPUT_DIR}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Saved:")
print("  embeddings/faiss_index.bin")
print("  embeddings/embeddings.npy")
print("  embeddings/documents.npy")
print("  embeddings/metadata.json")

print("\nPart 1 Complete.\n")