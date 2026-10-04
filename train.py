import os
import tensorflow as tf
from dataset import get_dataset, prepare_dataset
from model import build_ds_cnn
import config

def train():
    # 1. Load the dataset
    print("Loading dataset...")
    raw_dataset = get_dataset()
    
    # 2. Split dataset into train and validation (80/20)
    # Note: In a real scenario, you should split based on speakers or files to prevent leakage
    dataset_size = raw_dataset.cardinality().numpy()
    if dataset_size == tf.data.experimental.UNKNOWN_CARDINALITY or dataset_size == 0:
        print("Warning: Dataset is empty or cardinality unknown. Did you add audio files to data/wake_word and data/negative?")
        # For demonstration purposes, we will proceed even if empty, but it will fail.
    
    train_size = int(0.8 * dataset_size)
    
    train_ds = raw_dataset.take(train_size)
    val_ds = raw_dataset.skip(train_size)
    
    train_ds = prepare_dataset(train_ds, train=True)
    val_ds = prepare_dataset(val_ds, train=False)
    
    # 3. Build and compile the model
    print("Building model...")
    model = build_ds_cnn()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    # 4. Callbacks for saving the best model and early stopping
    os.makedirs("saved_model", exist_ok=True)
    checkpoint_keras = tf.keras.callbacks.ModelCheckpoint(
        "saved_model/best_model.keras", 
        save_best_only=True, 
        monitor='val_accuracy'
    )
    checkpoint_h5 = tf.keras.callbacks.ModelCheckpoint(
        "saved_model/best_model.h5", 
        save_best_only=True, 
        monitor='val_accuracy'
    )
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        patience=20, 
        restore_best_weights=True,
        monitor='val_loss'
    )
    
    # 5. Train the model
    print("Starting training...")
    try:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.EPOCHS,
            callbacks=[checkpoint_keras, checkpoint_h5, early_stopping_cb]
        )
        print("Training complete!")
    except Exception as e:
        print(f"Training failed. Make sure you have enough data. Error: {e}")

if __name__ == "__main__":
    train()
