from transformers import AutoTokenizer, AutoModel
import torch
import sqlite3
import pandas as pd
import numpy as np

# Load SecBERT model & tokenizer
model_name = "jackaduma/SecBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # Set model to evaluation mode

# Connect to SQLite & Load Preprocessed Logs (LIMIT to 50,000 logs)
conn = sqlite3.connect("../data/logs.db")
df = pd.read_csv("../data/preprocessed_logs.csv").head(50000)  # Load first 50,000 logs
conn.close()

# Print column names for debugging
print("Available Columns:", df.columns)

# Construct `llm_input` column
df["llm_input"] = (
    df["Time"].astype(str) + " | " +
    df["Source"].astype(str) + " → " +
    df["Destination"].astype(str) + " | " +
    df["Protocol"].astype(str) + " | " +
    df["Info"].astype(str)
)
print(f"llm_input column created for {len(df)} logs!")

# Batch processing logs into embeddings
texts = df["llm_input"].tolist()  # List of log texts
batch_size = 512  # Adjust batch size if needed
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
    ).to(device)

    # Ensure batch processing inside loop
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract embeddings (mean pooling over token embeddings)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    # Ensure correct shape for single batch items
    if embeddings.ndim == 1:
        embeddings = embeddings[None, :]

    all_embeddings.append(embeddings)

    print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} logs...")

# Concatenate all batch embeddings into a single array
all_embeddings = np.concatenate(all_embeddings, axis=0)

# Save embeddings for later use
np.save("../data/log_embeddings.npy", all_embeddings)

print(f"Successfully processed {len(all_embeddings)} logs and saved embeddings!")
