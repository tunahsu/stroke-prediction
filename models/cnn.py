from tensorflow import keras
from keras import layers

def awesome_3D_CNN(depth, width, height):
    '''
    Building a 3D convolutional neural network model.
    Using relu activation function for Convolution layer, pool size 2 for pooling layer and sigmoid for densing layer
    '''
    inputs = keras.Input((depth, width, height, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=2, activation="softmax")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


def awesome_3D_UNet(width, height, depth):
    inputs = keras.Input((width, height, depth, 1))

    x = layers.BatchNormalization()(inputs)

    x = layers.Conv3D(filters=8, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Conv3D(filters=8, kernel_size=3, padding='same', activation='linear')(x)
    x = layers.Activation('relu')(layers.BatchNormalization()(x))
    x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.Conv3D(16, kernel_size=3, padding='same', activation='linear')(x)
    x = layers.Activation('relu')(layers.BatchNormalization()(x))
    x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.Conv3D(32, kernel_size=3, padding='same', activation='linear')(x)
    x = layers.Activation('relu')(layers.BatchNormalization()(x))

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=2, activation="softmax")(x)
    
    # Define the model.
    model = keras.Model(inputs, outputs, name="3dunet")
    return model