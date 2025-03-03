import sqlite3
# print(sqlite3.version)


import pandas as pd

# Load dataset (modify filename as needed)
file_path = "../data/Midterm_53_group.csv"  
df = pd.read_csv("D:\\SEM 10\\SETS\\projectV2\\venv\\data\\Midterm_53_group.csv")


# Clean column names: Remove special characters and replace spaces with underscores
df.columns = [col.strip().replace(" ", "_").replace(".", "_").replace("-", "_") for col in df.columns]


# Connect to SQLite database (creates if not exists)
conn = sqlite3.connect("../data/logs.db")
cursor = conn.cursor()

# Generate CREATE TABLE statement dynamically
columns = ", ".join([f'"{col}" TEXT' for col in df.columns])  # Use double quotes to handle reserved words
create_table_query = f'CREATE TABLE IF NOT EXISTS network_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, {columns})'
cursor.execute(create_table_query)


# Insert dataset into SQLite
df.to_sql("network_logs", conn, if_exists="replace", index=False)

conn.commit()
conn.close()

print("Logs stored successfully in SQLite!")
