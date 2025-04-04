import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, callbacks

class MaxFeatureMap(layers.Layer):
    def __init__(self, **kwargs):
        super(MaxFeatureMap, self).__init__(**kwargs)

    def call(self, inputs):
        # Allow dynamic channel size determination
        input_shape = tf.shape(inputs)
        channels = input_shape[-1]

        # Assuming channels are even as per model design.
        split = tf.split(inputs, num_or_size_splits=2, axis=-1)
        return tf.maximum(split[0], split[1])

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        if shape[-1] is not None:
            shape[-1] = shape[-1] // 2
        else:
            shape[-1] = None
        return tuple(shape)

def res_block(input_tensor, filters, stride=1):
    """Residual block with MFM activation"""
    # --- Main Path ---
    # First Conv block changes dimensions if stride != 1
    x = layers.Conv2D(filters * 2, kernel_size=3, strides=stride, padding='same', use_bias=False)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = MaxFeatureMap()(x)  # Output channels = filters

    # Second Conv block always has stride 1
    x = layers.Conv2D(filters * 2, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x) # Shape just before potential Add: (batch, height, width, filters * 2)

    # --- Shortcut Path ---
    shortcut = input_tensor
    input_channels_static = input_tensor.shape[-1] # Get static channel dimension if available

    # Determine if projection is needed:
    # 1. If stride is not 1, dimensions change, so projection is needed.
    # 2. If stride is 1, but input channels don't match the channels of 'x' before Add (which is filters * 2), projection is needed.
    needs_projection = False
    if stride != 1:
        needs_projection = True
    # Check channel mismatch only if static shape is known
    if input_channels_static is not None and input_channels_static != (filters * 2):
        needs_projection = True

    if needs_projection:
        # Project shortcut to match the shape of 'x' before the Add layer
        shortcut = layers.Conv2D(filters * 2, kernel_size=1, strides=stride, padding='same', use_bias=False)(input_tensor)
        shortcut = layers.BatchNormalization()(shortcut)

    # --- Add ---
    # Now both x and shortcut should have shape (..., filters * 2)
    x = layers.Add()([x, shortcut])

    # --- Final Activation ---
    # Apply MFM after adding. Output channels = filters
    x = MaxFeatureMap()(x)
    return x

def build_resmax(input_shape, num_classes=2):
    """Build the ResMax model for audio spoofing detection"""
    inputs = layers.Input(shape=input_shape)
    
    # Initial Conv Layer (Output channels must be even for MFM; 64 is fine)
    x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = MaxFeatureMap()(x)  # Output channels = 32

    # Residual Blocks
    x = res_block(x, filters=32, stride=1)  # Input=32, Output=32
    x = res_block(x, filters=32, stride=1)  # Input=32, Output=32

    x = res_block(x, filters=64, stride=2)  # Downsample: Input=32, Output=64
    x = res_block(x, filters=64, stride=1)  # Input=64, Output=64

    x = res_block(x, filters=128, stride=2)  # Downsample: Input=64, Output=128
    x = res_block(x, filters=128, stride=1)  # Input=128, Output=128

    x = res_block(x, filters=256, stride=2)  # Downsample: Input=128, Output=256
    x = res_block(x, filters=256, stride=1)  # Input=256, Output=256

    # Final Layers
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes)(x)  # Output layer for 2 classes (bonafide/spoof)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model