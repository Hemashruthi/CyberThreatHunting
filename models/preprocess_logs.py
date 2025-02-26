import sqlite3
import pandas as pd

# Connect to SQLite and retrieve logs
conn = sqlite3.connect("../data/logs.db")
df = pd.read_sql_query("SELECT * FROM network_logs;", conn)
conn.close()

# Print column names to debug issues
print("✅ Columns in dataset:", df.columns)

# Drop empty values
df = df.dropna()

# Normalize text data
def clean_text(text):
    return str(text).lower().strip().replace("\n", " ").replace("\r", "")

# Apply cleaning only to text columns
text_columns = df.select_dtypes(include=["object"]).columns
df[text_columns] = df[text_columns].applymap(clean_text)

# Dynamically find necessary columns
required_columns = ["timestamp", "source_ip", "event_type", "log_message"]
available_columns = [col for col in required_columns if col in df.columns]

# Ensure all required columns exist before processing
if len(available_columns) == len(required_columns):
    df["llm_input"] = df["timestamp"] + " | " + df["source_ip"] + " | " + df["event_type"] + " | " + df["log_message"]
else:
    print(f"⚠ Warning: Some required columns are missing! Found: {df.columns}")

# Save preprocessed data
df.to_csv("../data/preprocessed_logs.csv", index=False)

print("Logs preprocessed and saved for LLM input!")
