from transformers import AutoTokenizer, AutoModel
import torch
import json
import numpy as np
import os

# Load SecBERT model & tokenizer
model_name = "jackaduma/SecBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Load MITRE ATT&CK techniques from JSON
with open("../data/mitre_dataset_1000.json", "r", encoding="utf-8") as f:
    mitre_attacks = json.load(f)

# Verify data structure
print("Sample Data:")  # Print first 2 items for debugging

# Extract MITRE ATT&CK descriptions (Ensure it's a dictionary)
if isinstance(mitre_attacks, dict):  # Ensure it's a dictionary
    mitre_texts = [f"{ttp_id} - {description}" for ttp_id, description in mitre_attacks.items()]
else:
    raise TypeError("❌ mitre_attacks is not a dictionary! Check JSON format.")
mitre_ids = list(mitre_attacks.keys())

print("✅ Sample MITRE texts:")

## **Batch Processing for MITRE ATT&CK embeddings**
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

    print(f"✅ Processed batch {i // batch_size + 1}/{(len(mitre_texts) // batch_size) + 1}")

# Concatenate all embeddings into a single array
mitre_embeddings = np.concatenate(all_embeddings, axis=0)

# Save embeddings & IDs for later matching
os.makedirs("../data", exist_ok=True)
np.save("../data/mitre_embeddings.npy", mitre_embeddings)
with open("../data/mitre_ids.json", "w") as f:
    json.dump(mitre_ids, f)

print(f"✅ Stored {len(mitre_ids)} MITRE ATT&CK embeddings in '../data/mitre_embeddings.npy'!")