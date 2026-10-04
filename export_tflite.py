import os
import tensorflow as tf
from dataset import get_dataset
import config

def representative_dataset_gen():
    """
    Generator function that provides representative dataset for INT8 quantization.
    It yields unbatched MFCC features.
    """
    # Load dataset without batching
    dataset = get_dataset().take(100) # Use 100 samples for calibration
    for feature, label in dataset:
        # TFLite Converter expects a list of inputs, with batch dimension
        yield [tf.expand_dims(feature, 0)]

def export_tflite():
    model_path = "saved_model/best_model.keras"
    if not os.path.exists(model_path):
        model_path = "saved_model/best_model.h5"
    if not os.path.exists(model_path):
        print("Error: Could not find saved_model/best_model.keras or best_model.h5. Train the model first.")
        return
        
    print(f"Loading Keras model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    print("Initializing TFLiteConverter...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Set optimization for INT8 Quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    
    # Ensure that all ops are quantized to INT8 (required by ESP32 TFLM)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    print("Converting model...")
    tflite_quant_model = converter.convert()
    
    # Save the TFLite model
    tflite_model_path = "saved_model/wakeword_quantized.tflite"
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_quant_model)
        
    print(f"Success! INT8 Quantized TFLite model saved to {tflite_model_path}")
    print("This file can be converted to a C array using 'xxd -i' for ESP32 deployment.")

if __name__ == "__main__":
    export_tflite()
