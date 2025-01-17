import matplotlib.pyplot as plt
import numpy as np
import torch


id2label = {
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

id2color = np.array([
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
])


def visualize_map(image, segmentation_map):
    color_seg = np.zeros(
        (segmentation_map.shape[0], segmentation_map.shape[1], segmentation_map.shape[2], 3), 
        dtype=np.uint8
    ) # batch, height, width, 3
    # seg map batch, h, w
    for label, color in enumerate(id2color):
        color_seg[segmentation_map == label, :] = color

    # Show image + mask
    color_seg = color_seg.transpose(0, 3, 1, 2)
    img = np.array(image) * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)      # [batch, channels, h, w]
    return img


def generate_grid(source, true_map, vis_map):
    num_images = source.shape[0] # batch size
    width = 224; height = 224; rows = 3
        
    grid_image = np.zeros((3, num_images*height, rows*width))
    for i in range(num_images):
        s = i * height
        e = s + height
        grid_image[:, s:e, :width] = source[i, :, :, :]
        grid_image[:, s:e, width:2*width] = true_map[i, :, :, :]
        grid_image[:, s:e, 2*width:] = vis_map[i, :, :, :]
    return grid_image