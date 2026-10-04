import os

# Data Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
WAKE_WORD_DIR = os.path.join(DATA_DIR, 'wake_word')
NEGATIVE_DIR = os.path.join(DATA_DIR, 'negative')
BACKGROUND_NOISE_DIR = os.path.join(DATA_DIR, 'background_noise')

# Audio Processing Parameters
SAMPLE_RATE = 16000
CLIP_DURATION_MS = 1000
CLIP_SAMPLES = int(SAMPLE_RATE * (CLIP_DURATION_MS / 1000))

# MFCC Parameters (Must match TFLite Micro frontend for ESP32)
# ESP32 TFLite Micro speech examples use:
# window size = 30ms, stride = 20ms
# num slices = 49, num features = 40 (standard)
WINDOW_SIZE_MS = 30
WINDOW_STRIDE_MS = 20
WINDOW_SIZE_SAMPLES = int(SAMPLE_RATE * (WINDOW_SIZE_MS / 1000))
WINDOW_STRIDE_SAMPLES = int(SAMPLE_RATE * (WINDOW_STRIDE_MS / 1000))
NUM_MFCC_FEATURES = 40
NUM_MEL_BINS = 40

# The number of spectrogram slices for 1 second audio
# (1000ms - 30ms) / 20ms + 1 = 48.5 -> 49 slices
NUM_SLICES = int((CLIP_DURATION_MS - WINDOW_SIZE_MS) / WINDOW_STRIDE_MS) + 1

# Training Parameters
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.001

# Data Augmentation
TIME_SHIFT_MS = 100
NOISE_VOLUME_RANGE = (0.0, 0.1) # 0 to 10% volume for background noise
