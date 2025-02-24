from transformers import AutoTokenizer, AutoModel
import torch
import sqlite3
import pandas as pd
import numpy as np  # Added for array processing

# Load SecBERT model & tokenizer
model_name = "jackaduma/SecBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Connect to SQLite & Load Preprocessed Logs (LIMIT to 1000 logs)
conn = sqlite3.connect("../data/logs.db")
df = pd.read_csv("../data/preprocessed_logs.csv").head(1000)
conn.close()

# Print column names for debugging
print("✅ Available Columns:", df.columns)

# Construct `llm_input` column
df["llm_input"] = (
    df["Time"].astype(str) + " | " +
    df["Source"].astype(str) + " → " +
    df["Destination"].astype(str) + " | " +
    df["Protocol"].astype(str) + " | " +
    df["Info"].astype(str)
)
print("✅ llm_input column created!")

# Batch processing logs into embeddings
texts = df["llm_input"].tolist()  # List of log texts
batch_size = 100  # Adjust batch size if needed
all_embeddings = []

for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i + batch_size]
    
    # Tokenizing batch
    inputs = tokenizer(
        batch_texts, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=512
    )

    # Ensure batch processing inside loop
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract embeddings
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    # Ensure correct shape for single batch items
    if embeddings.ndim == 1:
        embeddings = embeddings[None, :]

    all_embeddings.append(embeddings)

# Concatenate all batch embeddings into a single array
all_embeddings = np.concatenate(all_embeddings, axis=0)
df["log_embedding"] = list(all_embeddings)

# Save embeddings for later use
df.to_pickle("../data/log_embeddings.pkl")

print("✅ Successfully processed 1000 logs and saved embeddings!")
