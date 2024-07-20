import matplotlib.pyplot as plt
import os
import glob
import logging
import torch
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from mmseg.apis import init_model, inference_model
import numpy as np
import cv2
import json, uuid
from concurrent.futures import ThreadPoolExecutor
import time

def create_bounding_boxes_from_mask(mask):
    """
    Create bounding boxes from a segmentation mask.

    Args:
    mask (numpy array): Binary segmentation mask.

    Returns:
    list of tuples: Each tuple is (x, y, w, h) where (x, y) is the top-left corner of the bounding box and
                    w, h are the width and height of the bounding box.
    """
    # Ensure the mask is binary
    mask = mask.astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create bounding boxes for each contour
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    
    return bounding_boxes


# mask = np.zeros((200, 200), dtype=np.uint8)
# cv2.rectangle(mask, (50, 50), (150, 150), 255, -1)

# # Get the bounding box
# x, y, w, h = create_bounding_box_from_mask(mask)

# # Display the result
# print(f"Bounding Box: x={x}, y={y}, w={w}, h={h}")


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

category_colors = {i: tuple(color) for i, color in enumerate(palette)}

def list_image_files_in_directory(directory):
    jpg_files = glob.glob(os.path.join(directory, '**', '*.jpg'), recursive=True)
    png_files = glob.glob(os.path.join(directory, '**', '*.png'), recursive=True)
    jpeg_files = glob.glob(os.path.join(directory, '**', '*.jpeg'), recursive=True)
    return jpg_files + png_files + jpeg_files



def tensor_to_image_2(tensor, color_dict):
    tensor_img_cpu = tensor.cpu().numpy()
    rgb_image = np.zeros((tensor_img_cpu.shape[0], tensor_img_cpu.shape[1], 3), dtype=np.uint8)
    for category, color in color_dict.items():
        rgb_image[tensor_img_cpu == category] = color

    output_image = Image.fromarray(rgb_image, 'RGB')
    return output_image

def load_image(img_path):
    start_time = time.time()
    image = Image.open(img_path).convert('RGB')
    end_time = time.time()
    logger.info(f"Loaded image {img_path} in {end_time - start_time:.2f} seconds.")
    return img_path, image

def process_batch(batch):
    batch_start_time = time.time()
    
    # 记录模型推理时间
    inference_start_time = time.time()
    results = inference_model(model, batch)
    inference_end_time = time.time()
    logger.info(f"Model inference time: {inference_end_time - inference_start_time:.2f} seconds.")
    
    # 处理和保存结果
    process_results_start_time = time.time()
    for result, img_path in zip(results, batch):
        out_path = os.path.join(out_dir, "_".join(img_path.split("/")[-2:]).replace(".jpg", ".png"))
        os.makedirs(out_dir, exist_ok=True)
        tensor_img = result.pred_sem_seg.data[0]
        tensor_img_cpu = tensor_img.cpu().numpy()
        print(tensor_img_cpu)
        output = tensor_to_image_2(tensor_img, category_colors)
        output.save("haha.png")
        mask = (tensor_img_cpu == 48).astype(int)
        print(np.any(mask == 1))
        print(create_bounding_box_from_mask(mask))
        
    batch_end_time = time.time()
    logger.info(f"Processed batch in {batch_end_time - batch_start_time:.2f} seconds.")

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        

if __name__ == '__main__':
    total_start_time = time.time()
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    # img_dir = "/share2/zhulin/images/train"
    img_dir = "/root/mmsegmentation/data/test/trafficlight"
    img_list = list_image_files_in_directory(img_dir)[:10]

    config_path = '/root/mmsegmentation/configs/dinov2/dinov2_vitg_mask2former_240k_mapillary_v2-672x672_online.py'
    checkpoint_path = '/root/models/seg_0626.pth'

    out_dir = "/cpfs/temp_img"
    output_json_dir = '/cpfs/output/hds/semantic'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(output_json_dir, exist_ok=True)

    add_labels = False 
    batch_size = 1

    logger.info('Initializing model...')
    start_time = time.time()
    model = init_model(config_path, checkpoint_path, device='cuda:0')
    end_time = time.time()
    logger.info(f'Model initialized successfully in {end_time - start_time:.2f} seconds.')


    print(len(img_list), img_list)
    batches = list(chunk_list(img_list, batch_size))
    for batch in tqdm(batches, desc="Processing Batches"):
        process_batch(batch)
    total_end_time = time.time()
    logger.info(f"Total processing time: {total_end_time - total_start_time:.2f} seconds.")
