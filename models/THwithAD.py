import numpy as np
import faiss
import json
import pandas as pd
import random
from sklearn.ensemble import IsolationForest

# File paths
log_embeddings_path = "../data/log_embeddings.npy"
log_metadata_path = "../data/preprocessed_logs.csv"
mitre_embeddings_path = "../data/mitre_embeddings.npy"
mitre_ids_path = "../data/mitre_ids.json"
mitre_data_path = "../data/mitre_dataset_1000.json"

# Threshold for classifying events as suspicious
confidence_threshold = 0.0055
sample_size = 1000  # Sample size for logs

def convert_to_serializable(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def hypothesis_generation(log_embeddings, log_metadata, mitre_embeddings, mitre_ids, mitre_data, confidence_threshold):
    # Create FAISS index for MITRE ATT&CK embeddings
    dimension = mitre_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(mitre_embeddings)

    # Search for nearest neighbors
    k = 5  # Number of nearest neighbors to retrieve
    D, I = index.search(log_embeddings, k)

    # Generate hypotheses and confidence scores
    hypotheses = []
    detected_log_ids = set()
    for log_idx, distances, indices in zip(range(len(log_embeddings)), D, I):
        # Extract log metadata
        log_id = log_metadata.iloc[log_idx]["No_"]  # Use the "No_" column as the unique identifier
        timestamp = log_metadata.iloc[log_idx]["Time"]
        source_ip = log_metadata.iloc[log_idx]["Source"]
        destination_ip = log_metadata.iloc[log_idx]["Destination"]
        protocol = log_metadata.iloc[log_idx]["Protocol"]
        event_type = log_metadata.iloc[log_idx]["Info"]
        raw_log_message = log_metadata.iloc[log_idx]["Info"]

        for distance, idx in zip(distances, indices):
            confidence_score = 1 / (1 + distance)  # Example confidence score calculation
            if confidence_score >= confidence_threshold:
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
                    "predicted_mitre_technique_id": mitre_id,
                    "mitre_attack_name": mitre_attack_name,
                    "confidence_score": float(confidence_score),
                    "risk_level": "High",  # Since only high-confidence events are included
                    "suggested_mitigation": "Apply general best security practices and monitor for unusual activity."  # Generic mitigation
                }

                hypotheses.append(hypothesis)
                detected_log_ids.add(log_id)

    return hypotheses, detected_log_ids

def anomaly_detection(log_embeddings, log_metadata, detected_log_ids):
    # Filter out known threats
    unknown_log_embeddings = np.array([log_embeddings[i] for i in range(len(log_embeddings)) if log_metadata.iloc[i]["No_"] not in detected_log_ids])
    unknown_log_metadata = log_metadata[~log_metadata["No_"].isin(detected_log_ids)]

    # Anomaly detection using Isolation Forest
    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(unknown_log_embeddings)
    anomaly_scores = model.decision_function(unknown_log_embeddings)
    anomalies = model.predict(unknown_log_embeddings)

    anomaly_hypotheses = []
    for idx, (score, anomaly) in enumerate(zip(anomaly_scores, anomalies)):
        if anomaly == -1:  # Anomaly detected
            log_id = unknown_log_metadata.iloc[idx]["No_"]  # Use the "No_" column as the unique identifier
            timestamp = unknown_log_metadata.iloc[idx]["Time"]
            source_ip = unknown_log_metadata.iloc[idx]["Source"]
            destination_ip = unknown_log_metadata.iloc[idx]["Destination"]
            protocol = unknown_log_metadata.iloc[idx]["Protocol"]
            event_type = unknown_log_metadata.iloc[idx]["Info"]
            raw_log_message = unknown_log_metadata.iloc[idx]["Info"]

            hypothesis = {
                "log_id": log_id,
                "timestamp": timestamp,
                "source_ip": source_ip,
                "destination_ip": destination_ip,
                "protocol": protocol,
                "event_type": event_type,
                "anomaly_score": float(score),
                "risk_level": "High",  # Since anomalies are considered high risk
                "suggested_mitigation": "Investigate the unusual pattern of behavior and apply appropriate security measures."  # Example mitigation for unknown threats
            }

            anomaly_hypotheses.append(hypothesis)

    return anomaly_hypotheses

try:
    # Load log embeddings and metadata
    log_embeddings = np.load(log_embeddings_path, allow_pickle=True)
    log_metadata = pd.read_csv(log_metadata_path)

    # Ensure alignment with embeddings
    if len(log_embeddings) != len(log_metadata):
        # Align log_metadata with log_embeddings
        log_metadata = log_metadata.iloc[:len(log_embeddings)]
    
    # Sample 1000 logs randomly
    sample_indices = random.sample(range(len(log_embeddings)), sample_size)
    sampled_log_embeddings = log_embeddings[sample_indices]
    sampled_log_metadata = log_metadata.iloc[sample_indices].reset_index(drop=True)

    # Print column names for debugging
    print("✅ Available Columns in log_metadata:", sampled_log_metadata.columns)

    mitre_embeddings = np.load(mitre_embeddings_path, allow_pickle=True)

    # Load MITRE ATT&CK IDs and descriptions
    with open(mitre_ids_path, "r") as f:
        mitre_ids = json.load(f)
    with open(mitre_data_path, "r", encoding="utf-8") as f:
        mitre_data = json.load(f)

    # Ensure log_embeddings is a numpy array of floats
    if not isinstance(sampled_log_embeddings, np.ndarray):
        raise ValueError("log_embeddings should be a numpy array")
    if sampled_log_embeddings.dtype != np.float32 and sampled_log_embeddings.dtype != np.float64:
        sampled_log_embeddings = sampled_log_embeddings.astype(np.float32)

    # Run hypothesis generation
    hypotheses, detected_log_ids = hypothesis_generation(sampled_log_embeddings, sampled_log_metadata, mitre_embeddings, mitre_ids, mitre_data, confidence_threshold)

    # Run anomaly detection
    anomaly_hypotheses = anomaly_detection(sampled_log_embeddings, sampled_log_metadata, detected_log_ids)

    # Combine hypotheses and anomalies
    all_hypotheses = hypotheses + anomaly_hypotheses

    # Convert all_hypotheses to a serializable format
    all_hypotheses_serializable = json.loads(json.dumps(all_hypotheses, default=convert_to_serializable))

    # Save combined hypotheses to JSON
    with open("../data/hypotheses.json", "w") as f:
        json.dump(all_hypotheses_serializable, f, indent=4)

    print(f"✅ Generated and stored hypotheses in '../data/hypotheses.json'!")

except FileNotFoundError as e:
    print(f"❌ File not found: {e.filename}")
except Exception as e:
    print(f"❌ An error occurred: {str(e)}")