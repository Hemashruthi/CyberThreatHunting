# import faiss
# import numpy as np
# import pandas as pd
# import json

# # Load FAISS index
# index = faiss.read_index("../data/mitre_faiss.index")

# # Load stored log embeddings
# df = pd.read_pickle("../data/log_embeddings.pkl")  # Load precomputed log embeddings
# log_embeddings = np.stack(df["log_embedding"].values)  # Shape: (1000, 768)

# # Load MITRE ATT&CK IDs
# with open("../data/mitre_ids.json", "r") as f:
#     mitre_ids = json.load(f)

# # Search for nearest MITRE ATT&CK technique for each log
# D, I = index.search(log_embeddings, k=1)  # k=1 → Find the closest match

# # Assign matched techniques
# df["Predicted_MITRE_Technique"] = [mitre_ids[idx] for idx in I.flatten()]

# # Save results
# df.to_csv("../data/threat_hypothesis.csv", index=False)

# print("✅ Logs matched with MITRE ATT&CK techniques using FAISS and saved!")


import numpy as np
import faiss
import json
import pandas as pd
import pickle

# File paths
log_embeddings_path = "../data/log_embeddings.pkl"
log_metadata_path = "../data/preprocessed_logs.csv"
mitre_embeddings_path = "../data/mitre_embeddings.npy"
mitre_ids_path = "../data/mitre_ids.json"
mitre_data_path = "../data/mitre_dataset_1000.json"

try:
    # Load log embeddings and metadata
    with open(log_embeddings_path, "rb") as f:
        log_embeddings = pickle.load(f)
    log_metadata = pd.read_csv(log_metadata_path).head(1000)  # Ensure alignment with embeddings
    mitre_embeddings = np.load(mitre_embeddings_path, allow_pickle=True)

    # Load MITRE ATT&CK IDs and descriptions
    with open(mitre_ids_path, "r") as f:
        mitre_ids = json.load(f)
    with open(mitre_data_path, "r", encoding="utf-8") as f:
        mitre_data = json.load(f)

    # Create FAISS index for MITRE ATT&CK embeddings
    dimension = mitre_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(mitre_embeddings)

    # Search for nearest neighbors
    k = 5  # Number of nearest neighbors to retrieve
    D, I = index.search(log_embeddings, k)

    # Generate hypotheses and confidence scores
    hypotheses = []
    for log_idx, distances, indices in zip(range(len(log_embeddings)), D, I):
        # Extract log metadata
        log_id = log_metadata.iloc[log_idx]["ID"]
        timestamp = log_metadata.iloc[log_idx]["Time"]
        source_ip = log_metadata.iloc[log_idx]["Source"]
        destination_ip = log_metadata.iloc[log_idx]["Destination"]
        protocol = log_metadata.iloc[log_idx]["Protocol"]
        event_type = log_metadata.iloc[log_idx]["Event_Type"]
        raw_log_message = log_metadata.iloc[log_idx]["Info"]
        
        for distance, idx in zip(distances, indices):
            confidence_score = 1 / (1 + distance)  # Example confidence score calculation
            mitre_id = mitre_ids[idx]
            mitre_details = mitre_data[mitre_id].split(" - ", 1)
            mitre_attack_name = mitre_details[0]
            mitre_description = mitre_details[1]
            
            hypothesis = {
                "log_id": log_id,
                "timestamp": timestamp,
                "source_ip": source_ip,
                "destination_ip": destination_ip,
                "protocol": protocol,
                "event_type": event_type,
                "raw_log_message": raw_log_message,
                "log_embedding": log_embeddings[log_idx].tolist(),  # Convert numpy array to list
                "predicted_mitre_technique_id": mitre_id,
                "mitre_attack_name": mitre_attack_name,
                "mitre_description": mitre_description,
                "confidence_score": float(confidence_score),
                "risk_level": "High",  # Example risk level, adjust based on your criteria
                "suggested_mitigation": "Monitor process injection activity, enable security logging, and restrict rundll32 execution."  # Example mitigation, adjust based on your criteria
            }
            
            hypotheses.append(hypothesis)

    # Save hypotheses to JSON
    with open("../data/hypotheses.json", "w") as f:
        json.dump(hypotheses, f, indent=4)

    print(f"✅ Generated and stored hypotheses in '../data/hypotheses.json'!")

except FileNotFoundError as e:
    print(f"❌ File not found: {e.filename}")
except Exception as e:
    print(f"❌ An error occurred: {str(e)}")