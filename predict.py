#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import PIL
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from networks.cnn import awesome_3D_CNN
from scipy import ndimage
from tensorflow.keras.optimizers import Adam
from skimage.transform import resize
from matplotlib import pyplot as plt
from detect import det


# Config
D = 64
W = 128
H = 128

SCAN_PATH = 'dataset/HIGH/1182214(O)_3'
WEIGHT_PATH = 'checkpoints/3dcnn_d64.h5'
CLASS_INDEX = 1
# layer name to visualize
LAYER_NAME = 'conv3d_2'


# load model
Model_3D = awesome_3D_CNN(D, W, H)
Model_3D.load_weights(WEIGHT_PATH)
print('Loading The Model...')
# Model_3D.summary()


# Create a graph that outputs target convolution and output
grad_model = tf.keras.models.Model([Model_3D.inputs], [Model_3D.get_layer(LAYER_NAME).output, Model_3D.output])
grad_model.summary()


def read_data_file(filepath):
    slices = []
    xy_set = []

    for scan in sorted(os.listdir(filepath)):
        img_path = os.path.join(filepath, scan)

        # Get xy of detection result
        xy = det(img_path, size=512)
        if(xy): xy_set.append(xy)

        slice = np.asarray(PIL.Image.open(img_path).convert('L'))
        slices.append(slice)

    # Cut all images with same bbox
    x1 = min([x[0] for x in xy_set])
    y1 = min([x[1] for x in xy_set])
    x2 = max([x[2] for x in xy_set])
    y2 = max([x[3] for x in xy_set]) 
    slices = [x[y1:y2, x1:x2] for x in slices]

    slices = np.array(slices)
    return slices

def normalize(volume):
    min = 0
    max = 255
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume


def sample_data(img):
    desired_depth = D
    desired_width = W
    desired_height = H
    current_depth = img.shape[0]
    current_width = img.shape[1]
    current_height = img.shape[2]
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    img = ndimage.zoom(img, (depth_factor, width_factor, height_factor), order=1)
    return img


def process_scan(path):
    volume = read_data_file(path)
    volume = sample_data(volume)
    # volume = np.moveaxis(volume, 0, -1) # change shape order => (width, height, depth)
    return volume


# load volume
original_volume = process_scan(SCAN_PATH)
normalized_volume = normalize(original_volume)

# compute gradient
with tf.GradientTape() as tape:
    input_io = np.expand_dims(normalized_volume, axis=0)
    conv_outputs, predictions = grad_model(input_io)

    # lb = tf.constant([[0., 1.]], dtype=tf.float32)
    # loss_fn = tf.losses.BinaryCrossentropy()
    # loss = loss_fn(lb, predictions)
    class_channel = predictions[:, CLASS_INDEX]

# Extract filters and gradients
output = conv_outputs[0]
grads = tape.gradient(class_channel, conv_outputs)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
# print(output)
# print(grads)
print(predictions)

# average gradients spatially
# weights = tf.reduce_mean(grads, axis=(0, 1, 2))

# build a ponderated map of filters according to gradients importance
# cam = np.zeros(output.shape[0:3], dtype=np.float32)

# for index, w in enumerate(weights):
#     cam += w * output[:, :, :, index]


# capi = sample_data(cam)
# capi = np.maximum(capi, 0)
# heatmap = (capi - capi.min()) / (capi.max() - capi.min())
# heatmap = heatmap * 255.

heatmap = output @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)
heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
heatmap = heatmap.numpy()
heatmap = sample_data(heatmap)
# print(heatmap.shape)

row = 16
col = 4

f, axarr = plt.subplots(row, col, figsize=(col * 4, row * 4));
f.suptitle('Grad-CAM')

for i in range(row):
    for j in range(col):  
        axial_ct_img = original_volume[i * col + j, :,:]
        axial_grad_cmap_img = heatmap[i * col + j, :,:] * 255.

        # img_plot = axarr[i, j].imshow(axial_ct_img, cmap='gray');
        # axarr[i, j].axis('off')
        # axarr[i, j].set_title('CT')
        # print('{}: {}'.format([i, j], i * col + j))

        # img_plot = axarr[i, j].imshow(axial_grad_cmap_img, cmap='jet');
        # axarr[i, j].axis('off')
        # axarr[i, j].set_title('Grad-CAM')

        axial_overlay = cv2.addWeighted(axial_ct_img, 0.2, axial_grad_cmap_img, 0.8, 0, dtype=cv2.CV_32F)

        img_plot = axarr[i, j].imshow(axial_overlay, cmap='jet');
        axarr[i, j].axis('off')
        # axarr[i, j].set_title('Overlay')

plt.savefig('heatmap.jpg')
plt.show()