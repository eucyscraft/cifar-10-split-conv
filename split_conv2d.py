#!/usr/bin/env python3
"""A CNN module to split channels up individually and convolve them with weight sharing across channels (with different sets of output channels)"""

from cifar10_web import cifar10
from keras.layers import Conv2D, Lambda, Input, Flatten, Dense, concatenate
from keras.models import Model, Sequential
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
    model = Model(inputs=inputs, outputs=outputs)
    print(model.summary())
    return model

# Load CIFAR-10 for testing purposes
train_images, train_labels, test_images, test_labels = cifar10()
# Reshape the image tensors into proper 4D arrays
train_images = np.reshape(train_images, (-1, 3, 32, 32))
train_images = np.transpose(train_images, (0, 2, 3, 1))
test_images = np.reshape(test_images, (-1, 3, 32, 32))
test_images = np.transpose(test_images, (0, 2, 3, 1))

# Create a network with these split convolutional layers
activation = 'relu'
model = Sequential([
    create_split_conv2d(x_size=32, y_size=32, channels_in=3, channels_out_per_channel_in=8, kernel_size=4, activation=activation),
    create_split_conv2d(x_size=29, y_size=29, channels_in=24, channels_out_per_channel_in=8, kernel_size=4, activation=activation),
    Conv2D(kernel_size=1, filters=16, activation=activation),
    create_split_conv2d(x_size=26, y_size=26, channels_in=16, channels_out_per_channel_in=4, kernel_size=4, activation=activation),
    create_split_conv2d(x_size=23, y_size=23, channels_in=64, channels_out_per_channel_in=4, kernel_size=4, activation=activation),
    Conv2D(kernel_size=1, filters=16, activation=activation),
    Flatten(),
    Dense(10, activation='sigmoid')
])
print(model.summary())
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the network on the CIFAR-10 data
model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=100)
