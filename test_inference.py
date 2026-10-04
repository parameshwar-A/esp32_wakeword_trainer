import tensorflow as tf
import numpy as np
from dataset import decode_audio, get_spectrogram_and_label_id
import config
import sys

def infer_tflite(tflite_model_path, wav_file_path):
    # Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Load and process audio
    audio_binary = tf.io.read_file(wav_file_path)
    waveform = decode_audio(audio_binary)
    mfcc, _ = get_spectrogram_and_label_id(waveform, 0)
    
    # Add batch dimension
    input_data = tf.expand_dims(mfcc, 0)
    
    # TFLite uses INT8 for inputs since we fully quantized it. We need to quantize the float32 MFCCs.
    input_scale, input_zero_point = input_details[0]["quantization"]
    if input_scale > 0:
        input_data = input_data / input_scale + input_zero_point
        input_data = tf.cast(tf.round(input_data), tf.int8)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Get output and dequantize
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_scale, output_zero_point = output_details[0]["quantization"]
    if output_scale > 0:
        output_data = (tf.cast(output_data, tf.float32) - output_zero_point) * output_scale
        
    print(f"File: {wav_file_path}")
    print(f"Probabilities (Negative, Wake Word): {output_data[0]}")
    if output_data[0][1] > 0.5:
        print("Prediction: WAKE WORD DETECTED")
    else:
        print("Prediction: NEGATIVE")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_inference.py <model.tflite> <test_audio.wav>")
    else:
        infer_tflite(sys.argv[1], sys.argv[2])
