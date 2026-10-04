import tensorflow as tf
import config

def get_spectrogram(waveform):
    """
    Computes the spectrogram of an audio waveform.
    Matches TFLite Micro audio frontend parameters.
    """
    # Zero-padding for an audio waveform with less than expected samples.
    zero_padding = tf.zeros(
        [config.CLIP_SAMPLES] - tf.shape(waveform),
        dtype=tf.float32)
    
    # Cast the waveform tensors' dtype to float32.
    waveform = tf.cast(waveform, tf.float32)
    
    # Concatenate the waveform with `zero_padding`, which ensures all audio
    # clips are of the same length.
    equal_length = tf.concat([waveform, zero_padding], 0)
    
    # Convert waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
        equal_length, 
        frame_length=config.WINDOW_SIZE_SAMPLES, 
        frame_step=config.WINDOW_STRIDE_SAMPLES)
    
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    
    return spectrogram

def get_mfcc(spectrogram):
    """
    Computes the MFCCs from a spectrogram.
    """
    # Create mel weight matrix
    num_spectrogram_bins = tf.shape(spectrogram)[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=config.NUM_MEL_BINS,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=config.SAMPLE_RATE,
        lower_edge_hertz=20.0,
        upper_edge_hertz=4000.0)
    
    # Apply mel weights
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    
    # Compute log mel spectrogram
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    
    # Compute MFCCs
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :config.NUM_MFCC_FEATURES]
    
    # Add a channel dimension for the CNN (shape: [num_slices, num_features, 1])
    mfccs = tf.expand_dims(mfccs, -1)
    
    return mfccs

def process_audio(waveform):
    """
    End-to-end processing from waveform to MFCC feature tensor.
    """
    spectrogram = get_spectrogram(waveform)
    mfcc = get_mfcc(spectrogram)
    return mfcc
