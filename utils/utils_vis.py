import matplotlib.pyplot as plt
import numpy as np
import torch


id2label = {
    0: "background",
    1: "aeroplane", 
    2: "bicycle", 
    3: "bird", 
    4: "boat", 
    5: "bottle", 
    6: "bus", 
    7: "car", 
    8: "cat", 
    9: "chair", 
    10: "cow", 
    11: "diningtable", 
    12: "dog", 
    13: "horse", 
    14: "motorbike", 
    15: "person", 
    16: "potted plant", 
    17: "sheep", 
    18: "sofa", 
    19: "train", 
    20: "tv/monitor"
}

id2color = {k: list(np.random.choice(range(256), size=3)) for k,v in id2label.items()}


def visualize_map(image, segmentation_map):
    color_seg = np.zeros(
        (segmentation_map.shape[0], segmentation_map.shape[1], segmentation_map.shape[2], 3), 
        dtype=np.uint8
    ) # batch, height, width, 3
    # seg map batch, h, w
    for label, color in id2color.items():
        color_seg[segmentation_map == label, :] = color

    # Show image + mask
    color_seg = color_seg.transpose(0, 3, 1, 2)
    img = np.array(image) * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)      # [batch, channels, h, w]

    return img


def generate_grid(source, vis_map):
    num_images = source.shape[0] # batch size
    width = 224; height = 224; rows = 2
        
    grid_image = np.zeros((3, num_images*height, rows*width))
    for i in range(num_images):
        s = i * height
        e = s + height
        grid_image[:, s:e, :width] = source[i, :, :, :]
        grid_image[:, s:e, width:] = vis_map[i, :, :, :]
    return grid_image