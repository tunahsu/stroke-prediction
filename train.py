import numpy as np
import os
import random
import PIL
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy import ndimage
from tensorflow import keras
from tqdm import tqdm
from models.cnn import awesome_3D_network


# Config
D = 128
W = 256
H = 256
LR = 0.0001
epochs = 50
batch_size = 2


def read_data_file(filepath):
    slices = []
    for scan in os.listdir(filepath):
        slice = np.asarray(PIL.Image.open(os.path.join(filepath, scan)))
        slices.append(slice)
    slices = np.array(slices)
    return slices

def normalize(volume):
    min = -1000
    max = 700
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
    # img = ndimage.rotate(img, 90, reshape=False)
    img = ndimage.zoom(img, (depth_factor, width_factor, height_factor), order=0)
    return img


def process_scan(path):
    volume = read_data_file(path)
    volume = normalize(volume)
    volume = sample_data(volume)
    return volume


normal_scan_paths = [
    os.path.join(os.getcwd(), "dataset/LOW", x)
    for x in os.listdir("dataset/LOW")
]
abnormal_scan_paths = [
    os.path.join(os.getcwd(), "dataset/HIGH", x)
    for x in os.listdir("dataset/HIGH")
]

print("mri scans with normal heart: " + str(len(normal_scan_paths)))
print("mri scans with abnormal heart: " + str(len(abnormal_scan_paths)))


'''
Build train and validation datasets
Downsample the scans to have
shape of 128x256x256.
split the dataset into train and validation subsets.
'''

normal_scans = np.array([process_scan(path) for path in tqdm(normal_scan_paths)])
abnormal_scans = np.array([process_scan(path) for path in tqdm(abnormal_scan_paths)])

# assign 1 for stroke's, for the normal ones assign 0.
normal_labels = np.array([0 for _ in range(len(normal_scans))])
abnormal_labels = np.array([1 for _ in range(len(abnormal_scans))])

# Split data in the ratio 70-30 for training and validation.
x_train = np.concatenate((abnormal_scans[:7], normal_scans[:7]), axis=0)
y_train = np.concatenate((abnormal_labels[:7], normal_labels[:7]), axis=0)
x_val = np.concatenate((abnormal_scans[7:], normal_scans[7:]), axis=0)
y_val = np.concatenate((abnormal_labels[7:], normal_labels[7:]), axis=0)

print(
    "Number of samples in train and validation are %d and %d."
    % (x_train.shape[0], x_val.shape[0])
)

'''
 Data augmentation
'''

@tf.function
def rotate(volume):
    def scipy_rotate(volume):
        angles = [-20, -10, -5, 5, 10, 20]
        # pick angles at random
        angle = random.choice(angles)
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume


def train_preprocessing(volume, label):
    # volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def validation_preprocessing(volume, label):
    volume = tf.expand_dims(volume, axis=3)
    return volume, label

# Define data loaders.
train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

train_dataset = (
    train_loader.shuffle(len(x_train))
    .map(train_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)
validation_dataset = (
    validation_loader.shuffle(len(x_val))
    .map(validation_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)

# Build model.
model = awesome_3D_network(depth=D, width=W, height=H)
model.summary()

'''
 Train model
'''
initial_learning_rate = LR
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["acc"],
)

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "./checkpoints/3dcnn.h5", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

# Train the model, doing validation at the end of each epoch
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    shuffle=True,
    verbose=2,
    callbacks=[checkpoint_cb],
)

'''
 Visualizing model performance
'''
fig, ax = plt.subplots(1, 2, figsize=(20, 3))
ax = ax.ravel()

for i, metric in enumerate(["acc", "loss"]):
    ax[i].plot(model.history.history[metric])
    ax[i].plot(model.history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])

'''
 Make predictions on a single mri scan
'''
model.load_weights("./checkpoints/3dcnn.h5")
prediction = model.predict(np.expand_dims(normal_scans[0], axis=0))[0]

class_names = ["normal", "abnormal"]
if prediction[0] > 0.5:
    print("abnormal confidence = ", prediction[0]*100)
else:
    print("normal confidence = ", prediction[0]*100)