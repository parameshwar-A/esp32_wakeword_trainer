import os
import tensorflow as tf
import config
from audio_processing import process_audio

def decode_audio(audio_binary):
    """Decode WAV audio into a waveform."""
    audio, _ = tf.audio.decode_wav(contents=audio_binary)
    return tf.squeeze(audio, axis=-1)

def get_label(file_path):
    """Extract the label from the directory name."""
    parts = tf.strings.split(file_path, os.path.sep)
    # If it's in the wake_word folder, label is 1, else 0 (negative)
    is_wake_word = tf.math.equal(parts[-2], 'wake_word')
    label = tf.cast(is_wake_word, tf.int32)
    return label

def get_waveform_and_label(file_path):
    """Load audio and extract label."""
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label

def get_spectrogram_and_label_id(waveform, label):
    """Convert waveform to MFCC features and return with label."""
    mfcc = process_audio(waveform)
    return mfcc, label

def load_background_noise(noise_dir=config.BACKGROUND_NOISE_DIR):
    """Loads background noise files and splits them into 1-second chunks."""
    bg_noise = []
    if not os.path.exists(noise_dir):
        return bg_noise
        
    for filename in os.listdir(noise_dir):
        if filename.endswith('.wav'):
            filepath = os.path.join(noise_dir, filename)
            audio_binary = tf.io.read_file(filepath)
            waveform = decode_audio(audio_binary)
            
            # Split into chunks of exactly CLIP_SAMPLES
            num_samples = tf.shape(waveform)[0]
            num_chunks = num_samples // config.CLIP_SAMPLES
            if num_chunks > 0:
                waveform = waveform[:num_chunks * config.CLIP_SAMPLES]
                chunks = tf.reshape(waveform, [num_chunks, config.CLIP_SAMPLES])
                for chunk in chunks:
                    bg_noise.append(chunk)
    return bg_noise

def get_dataset(data_dir=config.DATA_DIR):
    """
    Creates a tf.data.Dataset for training/validation.
    Assumes directory structure:
    data/
      wake_word/
      negative/
    """
    # Find all wav files in wake_word and negative folders
    filenames = tf.io.gfile.glob(os.path.join(data_dir, '*', '*.wav'))
    # Filter out the background noise folder
    filenames = [f for f in filenames if 'background_noise' not in f]
    
    filenames = tf.random.shuffle(filenames)
    
    files_ds = tf.data.Dataset.from_tensor_slices(filenames)
    waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    
    # We could insert a data augmentation step here (e.g. adding background noise, time shifting)
    # using load_background_noise() and adding it to the waveform before processing.
    
    # Process audio into MFCC features
    mfcc_ds = waveform_ds.map(get_spectrogram_and_label_id, num_parallel_calls=tf.data.AUTOTUNE)
    
    return mfcc_ds

def prepare_dataset(dataset, batch_size=config.BATCH_SIZE, train=True):
    if train:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
