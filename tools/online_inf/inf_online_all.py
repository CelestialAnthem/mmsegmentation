import argparse
import os
import glob
import logging
import torch
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from mmseg.apis import init_model, inference_model
import numpy as np
import cv2
import time
import json, uuid
from oss_utils import OssUpload
from concurrent.futures import ThreadPoolExecutor

classes = ('Bird', 'Ground Animal', 'Curb', 'Fence', 'Guard Rail', 'Barrier',
           'Wall', 'Bike Lane', 'Crosswalk - Plain', 'Curb Cut', 'Parking',
           'Pedestrian Area', 'Rail Track', 'Road', 'Service Lane',
           'Sidewalk', 'Bridge', 'Building', 'Tunnel', 'Person', 'Bicyclist',
           'Motorcyclist', 'Other Rider', 'Lane Marking - Crosswalk',
           'Lane Marking - General', 'Mountain', 'Sand', 'Sky', 'Snow',
           'Terrain', 'Vegetation', 'Water', 'Banner', 'Bench', 'Bike Rack',
           'Billboard', 'Catch Basin', 'CCTV Camera', 'Fire Hydrant',
           'Junction Box', 'Mailbox', 'Manhole', 'Phone Booth', 'Pothole',
           'Street Light', 'Pole', 'Traffic Sign Frame', 'Utility Pole',
           'Traffic Light', 'Traffic Sign (Back)', 'Traffic Sign (Front)',
           'Trash Can', 'Bicycle', 'Boat', 'Bus', 'Car', 'Caravan',
           'Motorcycle', 'On Rails', 'Other Vehicle', 'Trailer', 'Truck',
           'Wheeled Slow', 'Car Mount', 'Ego Vehicle')
palette = [[165, 42, 42], [0, 192, 0], [196, 196, 196], [190, 153, 153],
           [180, 165, 180], [90, 120, 150], [102, 102, 156], [128, 64, 255],
           [140, 140, 200], [170, 170, 170], [250, 170, 160], [96, 96, 96],
           [230, 150, 140], [128, 64, 128], [110, 110, 110], [244, 35, 232],
           [150, 100, 100], [70, 70, 70], [150, 120, 90], [220, 20, 60],
           [255, 0, 0], [255, 0, 100], [255, 0, 200], [200, 128, 128],
           [255, 255, 255], [64, 170, 64], [230, 160, 50], [70, 130, 180],
           [190, 255, 255], [152, 251, 152], [107, 142, 35], [0, 170, 30],
           [255, 255, 128], [250, 0, 30], [100, 140, 180], [220, 220, 220],
           [220, 128, 128], [222, 40, 40], [100, 170, 30], [40, 40, 40],
           [33, 33, 33], [100, 128, 160], [142, 0, 0], [70, 100, 150],
           [210, 170, 100], [153, 153, 153], [128, 128, 128], [0, 0, 80],
           [250, 170, 30], [192, 192, 192], [220, 220, 0], [140, 140, 20],
           [119, 11, 32], [150, 0, 255], [0, 60, 100], [0, 0, 142],
           [0, 0, 90], [0, 0, 230], [0, 80, 100], [128, 64, 64], [0, 0, 110],
           [0, 0, 70], [0, 0, 192], [32, 32, 32], [120, 10, 10]]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

config_path = '/root/mmsegmentation/configs/dinov2/dinov2_vitg_mask2former_240k_mapillary_v2-672x672_online.py'
checkpoint_path = '/root/models/seg_0626.pth'

img_dir = "/data_path/test.txt"
out_dir = "/cpfs/temp_img"
output_json_dir = '/cpfs/output/hds/semantic'
os.makedirs(out_dir, exist_ok=True)
os.makedirs(output_json_dir, exist_ok=True)

add_labels = False 
batch_size = 1

def list_image_files_in_directory(directory):
    jpg_files = glob.glob(os.path.join(directory, '**', '*.jpg'), recursive=True)
    png_files = glob.glob(os.path.join(directory, '**', '*.png'), recursive=True)
    jpeg_files = glob.glob(os.path.join(directory, '**', '*.jpeg'), recursive=True)
    return jpg_files + png_files + jpeg_files

def load_image_list(img_dir):
    with open(img_dir) as input_f:
        json_list = [_.strip() for _ in input_f.readlines()]
    img_list = []
    for j in json_list:
        j = j if j.startswith("/") else "/" + j
        with open(j.strip(), 'r') as f:
            json_info = json.load(f)
        img_info = json_info['camera']
        for view in img_info:
            path = view['oss_path']
            path = path if path.startswith("/") else "/" + path
            img_list.append((path, j.strip()))
    return img_list

img_list = load_image_list(img_dir)
print(len(img_list))

logger.info('Initializing model...')
model = init_model(config_path, checkpoint_path, device='cuda:0')
logger.info('Model initialized successfully.')

category_colors = {i: tuple(color) for i, color in enumerate(palette)}

def adjust_brightness(color, factor):
    return tuple(min(int(c * factor), 255) for c in color)

def is_position_valid(avg_x, avg_y, text_size, existing_labels, buffer=10):
    for label_pos in existing_labels:
        lx, ly, lwidth, lheight = label_pos
        if not (avg_x + text_size[0] // 2 + buffer < lx or avg_x - text_size[0] // 2 - buffer > lx + lwidth or avg_y + text_size[1] // 2 + buffer < ly or avg_y - text_size[1] // 2 + buffer > ly + lheight):
            return False
    return True

def tensor_to_image(tensor, color_dict, classes, original_img, alpha=0.8, add_labels=True):
    tensor = tensor.squeeze(0).to('cuda')
    h, w = tensor.shape

    scale_factor = h / 2160
    font_size = int(48 * scale_factor)
    distance_threshold = 1500 * scale_factor
    move_distance = int(100 * scale_factor)

    font = ImageFont.truetype("/root/mmsegmentation/data/Arial.ttf", font_size)

    tensor_flat = tensor.flatten().cpu().numpy()
    unique_categories = np.unique(tensor_flat)
    color_array = np.zeros((tensor_flat.size, 4), dtype=np.uint8)

    for category in unique_categories:
        color = color_dict.get(category, (0, 0, 0))
        indices = np.where(tensor_flat == category)
        color_array[indices] = [*color, int(255 * alpha)]

    color_array = color_array.reshape(h, w, 4)
    overlay = Image.fromarray(color_array, 'RGBA')
    draw = ImageDraw.Draw(overlay)
    
    label_positions = {}
    for category in unique_categories:
        positions = np.column_stack(np.where(tensor.cpu().numpy() == category))
        label_positions[category] = positions.tolist()

    existing_labels = []

    if add_labels:
        for category, positions in label_positions.items():
            if category < len(classes):
                mask = np.zeros((h, w), dtype=np.uint8)
                mask[positions[:, 0], positions[:, 1]] = 255
                num_labels, labels_im = cv2.connectedComponents(mask)

                avg_positions = []
                for label in range(1, num_labels):
                    component_mask = (labels_im == label)
                    component_positions = np.column_stack(np.where(component_mask))
                    avg_x = np.mean(component_positions[:, 1])
                    avg_y = np.mean(component_positions[:, 0])
                    avg_positions.append((int(avg_x), int(avg_y)))

                merged_positions = []
                while avg_positions:
                    pos = avg_positions.pop(0)
                    merged_cluster = [pos]
                    to_remove = []
                    for other_pos in avg_positions:
                        if np.linalg.norm(np.array(pos) - np.array(other_pos)) < distance_threshold:
                            merged_cluster.append(other_pos)
                            to_remove.append(other_pos)
                    for item in to_remove:
                        avg_positions.remove(item)
                    merged_avg_x = int(np.mean([p[0] for p in merged_cluster]))
                    merged_avg_y = int(np.mean([p[1] for p in merged_cluster]))
                    merged_positions.append((merged_avg_x, merged_avg_y))

                if len(merged_positions) > 3:
                    merged_positions = merged_positions[:2]

                for avg_x, avg_y in merged_positions:
                    text = classes[category]
                    text_size = draw.textsize(text, font=font)
                    if not is_position_valid(avg_x, avg_y, text_size, existing_labels):
                        for dx, dy in [(-move_distance, 0), (move_distance, 0), (0, -move_distance), (0, move_distance), (-move_distance, -move_distance), (move_distance, move_distance), (-move_distance, move_distance), (move_distance, -move_distance)]:
                            new_x, new_y = avg_x + dx, avg_y + dy
                            if is_position_valid(new_x, new_y, text_size, existing_labels):
                                avg_x, avg_y = new_x, new_y
                                break

                    background_pos = (avg_x - (text_size[0]) // 2, avg_y - (text_size[1]) // 2)
                    draw.rectangle([background_pos, (background_pos[0] + text_size[0], background_pos[1] + text_size[1])], fill=(0, 0, 0))
                    bright_text_color = adjust_brightness(category_colors[category], 1.5)
                    text_pos = (background_pos[0] + (text_size[0] - text_size[0]) // 2, background_pos[1] + (text_size[1] - text_size[1]) // 2)
                    draw.text(text_pos, text, fill=bright_text_color, font=font)
                    existing_labels.append((background_pos[0], background_pos[1], text_size[0], text_size[1]))

        combined_img = Image.alpha_composite(original_img.convert('RGBA'), overlay)
        final_image = combined_img.convert('RGB')
    else:
        final_image = overlay

    return final_image

def load_image(img_path):
    image = Image.open(img_path).convert('RGB')
    return img_path, image

def tensor_to_image_2(tensor, color_dict, classes, original_img, alpha=0.8, add_labels=True):
    tensor_img_cpu = tensor.cpu().numpy()
    rgb_image = np.zeros((tensor_img_cpu.shape[0], tensor_img_cpu.shape[1], 3), dtype=np.uint8)
    for category, color in color_dict.items():
        rgb_image[tensor_img_cpu == category] = color

    output_image = Image.fromarray(rgb_image, 'RGB')
    return output_image

def process_batch(batch):
    results = inference_model(model, [img[0] for img in batch])
    with ThreadPoolExecutor() as executor:
        images = list(executor.map(load_image, [img[0] for img in batch]))
    
    for (img_path, original_img), result, json_path in zip(images, results, [img[1] for img in batch]):
        out_path = os.path.join(out_dir, "_".join(img_path.split("/")[-2:]).replace(".jpg", ".png"))
        os.makedirs(out_dir, exist_ok=True)
        tensor_img = result.pred_sem_seg.data[0]
        # img_result = tensor_to_image(tensor_img, category_colors, classes, original_img, alpha=1.0, add_labels=False)
        img_result = tensor_to_image_2(tensor_img, category_colors, classes, original_img, alpha=1.0, add_labels=False)
        img_result.save(out_path)
        oss_path = upload_img(out_path)
        logger.info(f'Processed and saved result for image: {oss_path}')
        
        output_json_path = os.path.join(output_json_dir, os.path.basename(json_path))
        if os.path.exists(output_json_path):
            with open(output_json_path, 'r') as f:
                json_data = json.load(f)
        else:
            with open(json_path, 'r') as f:
                json_data = json.load(f)
        img_info = json_data['camera']
        for view in img_info:
            if img_path == os.path.join('/', view['oss_path']):
                view['semantic_oss_path'] = oss_path
        
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump(json_data, f, indent=4)

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def upload_img(input_path):
    img_oss_path = ""
    png_name = input_path.split("/")[-1]
    img_oss_path = gen_path(1, png_name)
    ins = OssUpload.relative_path(img_oss_path)
    is_ok = ins.put_object_from_file(input_path)
    return img_oss_path

def gen_path(clip_id, video_name):
    prefix = "tos://haomo-public/lucas-generation/online_services/segementation"
    guid = str(uuid.uuid4())
    return f"{prefix}/{clip_id}/{guid}/{video_name}"

if __name__ == '__main__':
    batches = list(chunk_list(img_list, batch_size))
    for batch in tqdm(batches, desc="Processing Batches"):
        process_batch(batch)
