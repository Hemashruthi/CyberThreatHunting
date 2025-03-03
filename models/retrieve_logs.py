import sqlite3
import pandas as pd

# Connect to SQLite database
conn = sqlite3.connect("../data/logs.db")

# Retrieve logs
query = "SELECT * FROM network_logs LIMIT 10;"  # Fetch first 10 rows for preview
df = pd.read_sql_query(query, conn)

conn.close()

# Display retrieved logs
print("Retrieved Logs from SQLite:")
print(df.head())
