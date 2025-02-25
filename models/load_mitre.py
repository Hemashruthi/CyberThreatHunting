import json
import os


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

# Print confirmation
print(f" Stored {len(mitre_attacks)} MITRE ATT&CK techniques in '../data/mitre_dataset_1000.json'!")

# Print sample
print("Loaded MITRE ATT&CK Techniques (Sample):")
for key, value in list(mitre_attacks.items())[:5]:  # Show first 5 techniques
    print(f"{key}: {value}")
