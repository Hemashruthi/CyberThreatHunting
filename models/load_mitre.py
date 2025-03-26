import json
import os
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Load SecBERT model & tokenizer
model_name = "jackaduma/SecBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Load MITRE ATT&CK dataset
with open("../data/mitre_attack.json", "r", encoding="utf-8") as f:
    mitre_data = json.load(f)

# Extract Techniques (TTP IDs & Descriptions)
mitre_attacks = {}
count = 0
for technique in mitre_data["objects"]:
    if technique["type"] == "attack-pattern":
        ttp_id = technique["external_references"][0]["external_id"]
        ttp_name = technique["name"]
        description = technique.get("description", "No description available")
        mitre_attacks[ttp_id] = f"{ttp_name} - {description}"
        
        count += 1
        if count >= 1000:  # Stop after 1000 techniques
            break

# Ensure data directory exists
os.makedirs("../data", exist_ok=True)

# Save MITRE ATT&CK dataset to JSON
with open("../data/mitre_dataset_1000.json", "w", encoding="utf-8") as f:
    json.dump(mitre_attacks, f, indent=4)

print(f"Stored {len(mitre_attacks)} MITRE ATT&CK techniques in '../data/mitre_dataset_1000.json'!")

# Load MITRE ATT&CK techniques from JSON
with open("../data/mitre_dataset_1000.json", "r", encoding="utf-8") as f:
    mitre_attacks = json.load(f)

# Verify data structure
print("Sample Data:")  # Print first 2 items for debugging
if isinstance(mitre_attacks, dict):  # Ensure it's a dictionary
    mitre_texts = [f"{ttp_id} - {description}" for ttp_id, description in mitre_attacks.items()]
else:
    raise TypeError("❌ mitre_attacks is not a dictionary! Check JSON format.")
mitre_ids = list(mitre_attacks.keys())

print("Sample MITRE texts:")

# Batch Processing for MITRE ATT&CK embeddings
batch_size = 50  # Adjust batch size based on memory
all_embeddings = []

for i in range(0, len(mitre_texts), batch_size):
    batch_texts = mitre_texts[i:i + batch_size]
    
    # Tokenizing batch
    inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Generate embeddings for batch
    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    # Ensure correct shape for single batch items
    if embeddings.ndim == 1:
        embeddings = embeddings[None, :]

    all_embeddings.append(embeddings)

    print(f"Processed batch {i // batch_size + 1}/{(len(mitre_texts) // batch_size) + 1}")

# Concatenate all embeddings into a single array
mitre_embeddings = np.concatenate(all_embeddings, axis=0)

# Save embeddings & IDs for later matching
np.save("../data/mitre_embeddings.npy", mitre_embeddings)
with open("../data/mitre_ids.json", "w") as f:
    json.dump(mitre_ids, f)

print(f"Stored {len(mitre_ids)} MITRE ATT&CK embeddings in '../data/mitre_embeddings.npy'!")