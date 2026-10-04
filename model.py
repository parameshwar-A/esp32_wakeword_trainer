import tensorflow as tf
from tensorflow.keras import layers, models
import config

def build_ds_cnn(input_shape=(config.NUM_SLICES, config.NUM_MFCC_FEATURES, 1), num_classes=2):
    """
    Builds a lightweight Convolutional Neural Network optimized for ESP32.
    Uses Strided and Separable Convolutions with MaxPooling.
    Avoids BatchNorm to prevent inference-time collapse on microcontroller datasets.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same", activation="relu"),
        layers.SeparableConv2D(48, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.SeparableConv2D(64, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax")
    ], name="ESP32_Wakeword_CNN")
    
    return model

if __name__ == "__main__":
    model = build_ds_cnn()
    model.summary()
