import numpy as np
import os
import random
import PIL
import cv2
import torch
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy import ndimage
from tensorflow import keras
from tqdm import tqdm
from keras.utils import np_utils
from tensorflow.keras.applications import VGG19, ResNet50
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from networks.cnn import awesome_3D_CNN, awesome_3D_UNet
from keras_flops import get_flops


# Config
D = 64
W = 128
H = 128

# LR = 0.0001
# epochs = 40
# batch_size = 4

# # Locad custom YOLOv5 model
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='checkpoints/yolov5_512_500.pt')
# # Set IoU confidence
# model.iou = 0.1
# # Set confidence confidence
# model.conf = 0.1


# def det(img, size):
#     # Inference
#     results = model(img, size=size)
#     # result
#     crops = results.crop(save=False)
#     return [int(x.item()) for x in crops[0]['box']] if len(crops) > 0 else False


# def read_data_file(filepath):
#     slices = []
#     xy_set = []

#     for scan in sorted(os.listdir(filepath)):
#         img_path = os.path.join(filepath, scan)

#         # Get xy of detection result
#         xy = det(img_path, size=512)
#         if(xy): xy_set.append(xy)

#         slice = np.asarray(PIL.Image.open(img_path).convert('L'))
#         slices.append(slice)

#     # Cut all images with same bbox
#     x1 = min([x[0] for x in xy_set])
#     y1 = min([x[1] for x in xy_set])
#     x2 = max([x[2] for x in xy_set])
#     y2 = max([x[3] for x in xy_set]) 
#     slices = [x[y1:y2, x1:x2] for x in slices]

#     slices = np.array(slices)
#     return slices


# def normalize(volume):
#     min = 0
#     max = 255
#     volume[volume < min] = min
#     volume[volume > max] = max
#     volume = (volume - min) / (max - min)
#     volume = volume.astype("float32")
#     return volume


# def sample_data(img):
#     desired_depth = img.shape[0]
#     desired_width = W
#     desired_height = H
#     current_depth = img.shape[0]
#     current_width = img.shape[1]
#     current_height = img.shape[2]
#     depth = current_depth / desired_depth
#     width = current_width / desired_width
#     height = current_height / desired_height
#     depth_factor = 1 / depth
#     width_factor = 1 / width
#     height_factor = 1 / height
#     img = ndimage.zoom(img, (depth_factor, width_factor, height_factor), order=1)
#     return img


# def process_scan(path):
#     volume = read_data_file(path)
#     volume = normalize(volume)
#     volume = sample_data(volume)
#     return volume


# normal_scan_paths = [
#     os.path.join(os.getcwd(), "dataset/LOW", x)
#     for x in os.listdir("dataset/LOW")
# ]
# abnormal_scan_paths = [
#     os.path.join(os.getcwd(), "dataset/HIGH", x)
#     for x in os.listdir("dataset/HIGH")
# ]

# random.shuffle(normal_scan_paths)
# random.shuffle(abnormal_scan_paths)

# print("mri scans with normal heart: " + str(len(normal_scan_paths)))
# print("mri scans with abnormal heart: " + str(len(abnormal_scan_paths)))

# '''
# Build train and test datasets
# Downsample the scans to have
# shape of 128x128x128.
# split the dataset into train and test subsets.
# '''

# normal_scans = np.array([process_scan(path) for path in tqdm(normal_scan_paths)])
# abnormal_scans = np.array([process_scan(path) for path in tqdm(abnormal_scan_paths)])

# normal_slices = []
# abnormal_slices = []

# for scan in normal_scans:
#     for img in scan:
#         normal_slices.append(np.dstack((img, img, img)))
# for scan in abnormal_scans:
#     for img in scan:
#         abnormal_slices.append(np.dstack((img, img, img)))

# normal_slices = np.array(normal_slices)
# abnormal_slices = np.array(abnormal_slices)

# print(normal_slices.shape)




# # Assign 1 for stroke's, for the normal ones assign 0.
# normal_labels = np.array([0 for _ in range(len(normal_slices))])
# abnormal_labels = np.array([1 for _ in range(len(abnormal_slices))])

# # Split data in the ratio 70-30 for training and testing.
# a = int(len(abnormal_slices) * 0.65)
# b = int(len(normal_slices) * 0.65)
# x_train = np.concatenate((abnormal_slices[:a], normal_slices[:b]), axis=0)
# y_train = np.concatenate((abnormal_labels[:a], normal_labels[:b]), axis=0)
# x_test = np.concatenate((abnormal_slices[a:], normal_slices[b:]), axis=0)
# y_test = np.concatenate((abnormal_labels[a:], normal_labels[b:]), axis=0)

# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)

# print(
#     "Number of samples in train and test are %d and %d."
#     % (x_train.shape[0], x_test.shape[0])
# )


# def train_preprocessing(volume, label):
#     # volume = tf.expand_dims(volume, axis=-1)
#     # volume = tf.expand_dims(volume, axis=0)
#     return volume, label


# def test_preprocessing(volume, label):
#     # volume = tf.expand_dims(volume, axis=-1)
#     # volume = tf.expand_dims(volume, axis=0)
#     return volume, label

# # Define data loaders.
# train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# test_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))

# train_dataset = (
#     train_loader.shuffle(len(x_train))
#     .map(train_preprocessing)
#     .batch(batch_size)
#     .prefetch(2)
# )
# test_dataset = (
#     test_loader.shuffle(len(x_test))
#     .map(test_preprocessing)
#     .batch(batch_size)
#     .prefetch(2)
# )

# Build model.
vggModel = VGG19(weights="imagenet", include_top=False,
    input_tensor=Input(shape=(W, H, 3)))

outputs = vggModel.output
outputs = Flatten(name="flatten")(outputs)
outputs = Dropout(0.5)(outputs)
outputs = Dense(2, activation="softmax")(outputs)

model = Model(inputs=vggModel.input, outputs=outputs)

for layer in vggModel.layers:
    layer.trainable = False

model.summary()
flops = get_flops(vggModel, batch_size=1)
print(f"FLOPS: {flops / 10 ** 6:.03} M")

# '''
#  Train model
# '''
# # initial_learning_rate = LR
# # lr_schedule = keras.optimizers.schedules.ExponentialDecay(
# #     initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
# # )
# model.compile(
#     loss="binary_crossentropy",
#     optimizer='adam', 
#     metrics=["acc"],
# )

# # Define callbacks.
# checkpoint_cb = keras.callbacks.ModelCheckpoint(
#     "./checkpoints/vgg_40.h5", save_best_only=True
# )
# early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

# # Train the model, doing validation at the end of each epoch
# model.fit(
#     train_dataset,
#     validation_data=test_dataset,
#     epochs=epochs,
#     shuffle=True,
#     verbose=2,
#     callbacks=[checkpoint_cb],
# )

# a = model.history.history['acc']
# b = model.history.history['val_acc']

# '''
#  Visualizing model performance
# '''
# fig, ax = plt.subplots(1, 2, figsize=(20, 3))
# ax = ax.ravel()

# met = ['Accuracy', 'Loss']
# for i, metric in enumerate(["acc", "loss"]):
#     ax[i].plot(model.history.history[metric])
#     ax[i].plot(model.history.history["val_" + metric])
#     ax[i].set_title("Model {}".format(met[i]))
#     ax[i].set_xlabel("epochs")
#     ax[i].set_ylabel(metric)
#     ax[i].legend(["Train", "Val"])
# fig.savefig('vgg_result_40.jpg')

# # Inference on test set
# model.load_weights("./checkpoints/vgg_40.h5")
# y_pred = []

# for img in x_test:
#     img = np.expand_dims(img, axis=0)  # rank 4 tensor for prediction
#     y = model.predict(img)
#     y_pred.append(y[:][0])

# y_pred = np.array(y_pred)
# print(y_pred.shape)

# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import roc_curve, auc
# print('Area under ROC curve : ', roc_auc_score(y_test, y_pred) *100 )

# print(a)
# print(b)

# # Compute ROC curve and ROC area for each class
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(2):
#     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])


# # Compute micro-average ROC curve and ROC area
# cls = 1 # class name
# fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# #print(roc_auc)
# print("Area under the ROC curve for positive class:", roc_auc[1])

# from sklearn.metrics import classification_report
# for i in range(len(y_pred)):
#     max_value = max(y_pred[i])
#     for j in range(len(y_pred[i])):
#         if max_value == y_pred[i][j]:
#             y_pred[i][j] = 1
#         else:
#             y_pred[i][j] = 0
# print("Report:", classification_report(y_test, y_pred))

# plt.figure()
# lw = 2 # line width
# plt.plot(fpr[cls], tpr[cls], color='darkorange', lw=lw, label='ROC Curve (area = %0.2f)' % roc_auc[cls])
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.legend(loc="lower right")
# plt.savefig('{}_{}.png'.format('vgg', 'roc'), dpi=500)