import faiss
import numpy as np

# Load stored MITRE embeddings
mitre_embeddings = np.load("../data/mitre_embeddings.npy")  # Shape: (1000, 768)

# Create FAISS index for fast similarity search
index = faiss.IndexFlatL2(mitre_embeddings.shape[1])  # L2 distance (Euclidean)
index.add(mitre_embeddings)  # Add MITRE embeddings to index

# Save the FAISS index for later use
faiss.write_index(index, "../data/mitre_faiss.index")

print("✅ FAISS index created and stored!")
