#!/usr/bin/env python3
"""A CNN module to split channels up individually and convolve them with weight sharing across channels (with different sets of output channels)"""

from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Lambda, Input, Flatten, Dense, concatenate
from keras.models import Model, Sequential
from keras.utils import to_categorical
import numpy as np

def create_split_conv2d(x_size: int, y_size: int, channels_in: int, channels_out_per_channel_in: int, **other_conv2d_params) -> Model:
    # Take the image inputs as a normal CNN layer
    inputs = Input(shape=(x_size, y_size, channels_in))
    # Add the outputs to a list for concatenation later
    conv_outputs = []
    # Create a convolutional layer shared across the channels
    conv_layer = Conv2D(filters=channels_out_per_channel_in, **other_conv2d_params)
    # Iterate over the channels; slice out each channel (with an indexing hack to preserve the last dimension with a size of 1) and put them through the regular CNN layer
    for channel_index in range(channels_in):
        x = Lambda(lambda tensor: tensor[:, :, :, channel_index:channel_index + 1], lambda shape: (shape[0], shape[1], shape[2], 1))(inputs)
        x = conv_layer(x)
        conv_outputs.append(x)
    # Concatenate them back together at the end, with shape (examples, x_size, y_size, channels_in * channels_out_per_channel_in)
    outputs = concatenate(conv_outputs)
    # Return a model with these inputs and outputs
    return Model(inputs=inputs, outputs=outputs)

# Load CIFAR-10 for testing purposes
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
# One-hot encode the labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Create a network with these split convolutional layers
activation = 'relu'
model = Sequential([
    create_split_conv2d(x_size=32, y_size=32, channels_in=3, channels_out_per_channel_in=64, kernel_size=5, activation=activation),
    Conv2D(kernel_size=1, filters=96, activation=activation),
    MaxPooling2D(pool_size=3, strides=2),
    create_split_conv2d(x_size=13, y_size=13, channels_in=96, channels_out_per_channel_in=2, kernel_size=5, activation=activation),
    Conv2D(kernel_size=1, filters=192, activation=activation),
    MaxPooling2D(pool_size=3, strides=2),
    Conv2D(kernel_size=3, filters=192, activation=activation),
    Conv2D(kernel_size=1, filters=192, activation=activation),
    Conv2D(kernel_size=1, filters=10, activation=activation),
    GlobalAveragePooling2D()
])
print(model.summary())
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the network on the CIFAR-10 data
model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=100)
