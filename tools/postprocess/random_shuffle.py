import json
import random

data_path = "/share/tengjianing/songyuhao/segmentation/datasets/0609/train_mapillary_18000.json"

with open (data_path, "r") as data_file:
    data = json.load(data_file)

# Randomly select a subset of dictionaries
subset = random.sample(data, 10000)

# Save to a new JSON file
output_path = "/share/tengjianing/songyuhao/segmentation/datasets/0702/train_mapillary_10000.json"
with open(output_path, 'w') as f:
    json.dump(subset, f, indent=4)

output_path
