# ESP32 Wake Word Model Trainer (`esp32_wakeword_trainer`)

A Python-based training, evaluation, and quantization pipeline for custom wake word detection models targeting the standard ESP32 (dual-core Xtensa LX6) using TensorFlow Lite Micro.

This pipeline matches the firmware DSP feature extractor (16 kHz sampling rate, 30 ms Hann window, 20 ms stride, 40-bin Mel filterbank, DCT-II orthogonal transform producing 49 frames × 40 MFCCs). It trains a Depthwise Separable CNN (DS-CNN) and exports an INT8 quantized `.tflite` model optimized for microcontroller memory limits.

---

## Architecture Overview

1. **Audio Feature Extraction (`audio_processing.py`)**:
   - Computes MFCC features that mathematically mirror the ESP-IDF `esp-dsp` C implementation (`mel_tables.h` and `audio_frontend.c`).
   - Frame length: 30 ms (480 samples at 16 kHz).
   - Frame stride: 20 ms (320 samples).
   - Spectrogram dimension: 49 time frames × 40 MFCC channels.

2. **Model Design (`model.py`)**:
   - Lightweight Depthwise Separable CNN (DS-CNN).
   - Input shape: `(49, 40, 1)`.
   - Output: 2 classes (`[negative, wake_word]`) with softmax activation.
   - Designed specifically without BatchNormalization layers to prevent inference-time statistics shift on small embedded datasets.

3. **Data Augmentation (`dataset.py`)**:
   - Random time shifting (±100 ms).
   - Dynamic background noise mixing (0% to 10% volume) from ambient recordings.

4. **Full INT8 Quantization (`export_tflite.py`)**:
   - Calibrated using a representative dataset of real audio samples.
   - Enforces full integer quantization for all operators (`TFLITE_BUILTINS_INT8`) with `int8` input and output tensors for compatibility with ESP32 TFLM.

---

## Project Structure

```text
esp32_wakeword_trainer/
├── config.py              # Central hyperparameters (sampling rate, frame dimensions, epochs)
├── audio_processing.py    # MFCC audio preprocessing (Hann, FFT, Mel filterbank, DCT-II)
├── dataset.py             # Data loader, waveform decoding, and augmentation pipeline
├── model.py               # Depthwise Separable CNN architecture definition
├── train.py               # Model training script with ModelCheckpoint and EarlyStopping
├── export_tflite.py       # INT8 TFLite post-training quantization and export
├── test_inference.py      # Offline inference validation tool for WAV files
├── requirements.txt       # Python package dependencies
├── .gitignore             # Git exclusions (datasets, model weights, venvs, caches)
├── README.md              # Project documentation
├── data/
│   ├── wake_word/         # Positive wake word samples (.wav files, excluded from git)
│   ├── negative/          # Negative speech / other words (.wav files, excluded from git)
│   └── background_noise/  # Ambient noise / room silence (.wav files, excluded from git)
└── saved_model/           # Trained models and checkpoints (.keras, .h5, .tflite, excluded from git)
```

> [!IMPORTANT]
> **Audio Datasets and Model Weights are Excluded from Git:**
> Audio recordings (`data/*/*.wav`) and trained weight checkpoints (`saved_model/*`) are excluded via `.gitignore` to keep repository size lean and prevent checking in proprietary audio data. Follow the setup instructions below to populate your training data.

---

## Setup and Installation

### Prerequisites

- Python 3.9 - 3.11
- pip / virtualenv

### 1. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

*(Core dependencies: `tensorflow>=2.10.0`, `numpy`, `librosa`, `pyserial`).*

---

## Step-by-Step Training Workflow

All commands below run on your **[HOST PC]** inside the activated virtual environment.

---

### Step 1: Organize Training Data

Place your 1-second, 16 kHz, 16-bit mono `.wav` audio files properly and appropriately into their designated subdirectories inside `data/`:

- `data/wake_word/`: Positive target keyword utterances (e.g., 100+ variations across different speakers, pitches, and acoustics).
- `data/negative/`: Negative speech samples (random conversational speech, phonetically similar words, and competing names).
- `data/background_noise/`: Ambient noise profiles (room silence, air conditioning, fan noise, typing, or street sounds).

Ensure each audio file is exactly 1.0 second (16,000 samples) at 16 kHz 16-bit signed PCM mono.

For collecting clean training samples directly on physical hardware, we recommend using the companion [audio_recorder_esp32](https://github.com/parameshwar-A/audio_recorder_esp32) project.

---

### Step 2: Train the Model

Launch model training:

```bash
python3 train.py
```

Training saves the checkpoint with the highest validation accuracy to:
- `saved_model/best_model.keras`
- `saved_model/best_model.h5`

---

### Step 3: Export and Quantize to INT8 TFLite

Convert the trained Keras model into an INT8 quantized FlatBuffer model suitable for ESP32:

```bash
python3 export_tflite.py
```

Output file:
- `saved_model/wakeword_quantized.tflite` (~80-90 KB)

---

### Step 4: Test Model Inference Offline

Evaluate model predictions on individual `.wav` clips before deploying to hardware:

```bash
python3 test_inference.py saved_model/wakeword_quantized.tflite data/wake_word/sample_0001.wav
python3 test_inference.py saved_model/wakeword_quantized.tflite data/negative/sample_0001.wav
```

Output displays raw probabilities and classification:
```text
File: data/wake_word/sample_0001.wav
Probabilities (Negative, Wake Word): [0.03125, 0.96875]
Prediction: WAKE WORD DETECTED
```

---

### Step 5: Deploy to ESP32 Firmware

Once satisfied with test accuracy, deploy the model to your firmware project:

```bash
# From the firmware project root:
python3 scripts/update_model.py ../esp32_wakeword_trainer/saved_model/wakeword_quantized.tflite
```

This extracts quantization parameters and generates `main/model_data.h` ready for ESP-IDF compilation.
