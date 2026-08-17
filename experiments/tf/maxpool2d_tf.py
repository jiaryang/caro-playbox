import tensorflow as tf
from tensorflow.keras import layers, models, mixed_precision
import numpy as np
import argparse

# Enable mixed precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
tf.keras.backend.set_image_data_format('channels_last')
tf.config.optimizer.set_jit(True)

def create_conv_bn_layers(input_shape, pool_size, strides):
    model = models.Sequential()
    
    # Input layer with configurable input shape
    model.add(layers.InputLayer(input_shape=input_shape, dtype=tf.float32))

    # Convolutional layers with batch normalization
    model.add(layers.BatchNormalization(axis=-1))  # Axis=-1 corresponds to channels_last
    padding = 'same'
    model.add(layers.MaxPooling2D(pool_size, strides=strides, padding=padding))
    model.add(layers.BatchNormalization(axis=-1))  # Axis=-1 corresponds to channels_last
    model.add(layers.Activation('relu'))
    
    # Flatten layer to transition from convolutional layers to fully connected layers
    model.add(layers.Flatten())
    
    return model

def readArgv():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hi', help='Height of input', type=int, required=True)
    parser.add_argument('--wi', help='Weight of input', type=int, required=True)
    parser.add_argument('--ci', help='Number of channels of input', type=int, required=True)
    parser.add_argument('-p', help='Pool size', type=int, required=True)
    parser.add_argument('-n', help='Batch size', type=int, required=True)
    parser.add_argument('--stride', help='stride steps of conv kenel', type=int, required=True)

    args = parser.parse_args()

    return args


args=readArgv()
# Example configuration
input_shape = (args.hi, args.wi, args.ci)  # Configurable input shape
pool_size = (args.p, args.p)        # Configurable kernel size for convolutional layers
strides = (args.stride, args.stride)          # Configurable pool size for max pooling layers

# Create model with configurable layers and batch normalization
model = create_conv_bn_layers(input_shape, pool_size, strides)

# Display the model summary
model.summary()

# Continue with the rest of your training code, using model.fit() with your data
# For example, you can generate synthetic data for testing
batch_size = args.n
num_samples = batch_size * 20
train_images = np.random.rand(num_samples, *input_shape).astype(np.float32)
train_labels = np.random.randint(0, 2, size=num_samples)  # Binary classification labels

# Compile the model with mixed precision
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 1

model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, validation_split=0.0)

