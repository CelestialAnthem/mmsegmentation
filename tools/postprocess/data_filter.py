import os
import glob
import json
import matplotlib.pyplot as plt
import numpy as np

s = 0
t = 40000

def get_all_json_files(root_dir):
    """Recursively search for all JSON files in the given directory."""
    return glob.glob(os.path.join(root_dir, '**', '*.json'), recursive=True)

def merge_and_flatten_json_files(json_file_list):
    """Merge and flatten JSON files into a single list sorted by 'image_score'."""
    merged_data = []
    
    for json_file in json_file_list:
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for key, value in data.items():
                if isinstance(value, list):
                    merged_data.extend(value)
                else:
                    merged_data.append(value)
    
    # Ensure all image_scores are less than 1
    merged_data = [entry for entry in merged_data if entry.get('image_score', 0) < 1]

    # Sort by 'image_score' in descending order
    merged_data.sort(key=lambda x: x.get('image_score', 0), reverse=True)
    
    return merged_data

def load_jsonl(jsonl_file_path):
    """Load JSONL file into a list of JSON objects."""
    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

def match_top_data_with_jsonl(top_data, jsonl_data):
    """Match the top data entries with JSONL data based on 'img_path'."""
    jsonl_dict = {entry['img_path']: entry for entry in jsonl_data}
    return [jsonl_dict[entry['img_path']] for entry in top_data if entry['img_path'] in jsonl_dict]

# Specify root directory
root_directory = '/mnt/ve_share/songyuhao/seg_cleanlab/res'

# Get all JSON files' paths
all_json_files = get_all_json_files(root_directory)

# Filter JSON files containing "processed"
all_json_list = [json_file for json_file in all_json_files if "processed" in json_file]

# Merge, flatten, and sort the data
flattened_sorted_data = merge_and_flatten_json_files(all_json_list)

# Save flattened and sorted data
flattened_sorted_data_file = '/share/tengjianing/songyuhao/segmentation/datasets/0626/flattened_sorted_data.json'
with open(flattened_sorted_data_file, 'w', encoding='utf-8') as outfile:
    json.dump(flattened_sorted_data, outfile, ensure_ascii=False, indent=4)

# Get top 20000 data entries
top_20000_data = flattened_sorted_data[:t]

# Load JSONL file
jsonl_file_path = '/share/tengjianing/songyuhao/segmentation/datasets/0617/train_80000_p.jsonl'
jsonl_data = load_jsonl(jsonl_file_path)

# Match top data with JSONL data
matched_data = match_top_data_with_jsonl(top_20000_data, jsonl_data)

# Extract image_scores for histogram
image_scores = [entry.get('image_score', 0) for entry in flattened_sorted_data]

# Plot histogram of image_scores
plt.figure(figsize=(10, 6))
plt.hist(image_scores, bins=50, color='blue', edgecolor='black')
plt.title('Histogram of Image Scores')
plt.xlabel('Image Score')
plt.ylabel('Frequency')
output_image_path = '/root/image_score_histogram.png'
plt.savefig(output_image_path)

# Save matched JSON data
output_file = f'/share/tengjianing/songyuhao/segmentation/datasets/0702/top{s}_{t}.json'
with open(output_file, 'w', encoding='utf-8') as outfile:
    json.dump(matched_data, outfile, ensure_ascii=False, indent=4)

print(f"Flattened and sorted data saved to {flattened_sorted_data_file}")
print(f"Matched JSON data saved to {output_file}")
