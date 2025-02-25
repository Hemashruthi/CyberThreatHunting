import faiss
import numpy as np
import pandas as pd
import json

# Load FAISS index
index = faiss.read_index("../data/mitre_faiss.index")

# Load stored log embeddings
df = pd.read_pickle("../data/log_embeddings.pkl")  # Load precomputed log embeddings
log_embeddings = np.stack(df["log_embedding"].values)  # Shape: (1000, 768)

# Load MITRE ATT&CK IDs
with open("../data/mitre_ids.json", "r") as f:
    mitre_ids = json.load(f)

# Search for nearest MITRE ATT&CK technique for each log
D, I = index.search(log_embeddings, k=1)  # k=1 → Find the closest match

# Assign matched techniques
df["Predicted_MITRE_Technique"] = [mitre_ids[idx] for idx in I.flatten()]

# Save results
df.to_csv("../data/threat_hypothesis.csv", index=False)

print("✅ Logs matched with MITRE ATT&CK techniques using FAISS and saved!")
