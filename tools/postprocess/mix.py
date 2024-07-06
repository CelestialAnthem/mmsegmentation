import json
import jsonlines
import random

def load_jsonl(file_path):
    data = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.append(obj)
    return data

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def mix_data(data1, data2, ratio, total_length=20000):
    num_from_data2 = int(total_length * ratio)
    num_from_data1 = total_length - num_from_data2
    
    sampled_data1 = random.sample(data1, min(num_from_data1, len(data1)))
    sampled_data2 = random.sample(data2, min(num_from_data2, len(data2)))
    
    mixed_data = sampled_data1 + sampled_data2
    random.shuffle(mixed_data)
    return mixed_data


def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def main():
    # Load the JSONL files
    file1 = '/share/tengjianing/songyuhao/segmentation/datasets/0617/train_80000_p.json'
    file2 = '/share/tengjianing/songyuhao/segmentation/datasets/0609/train_mapillary_18000.json'
    
    data1 = load_jsonl(file1)
    data2 = load_json(file2)
    
    ratios = [i/10 for i in range(11)]  # Ratios from 0% to 100%

    for ratio in ratios:
        mixed_data = mix_data(data1, data2, ratio)
        save_json(mixed_data, f'/share/tengjianing/songyuhao/segmentation/datasets/0617/mixed_data_{int(ratio*100)}.json')

if __name__ == "__main__":
    main()
