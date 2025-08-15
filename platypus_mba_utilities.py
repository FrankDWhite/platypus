import tensorflow as tf
import tensorflow_ranking as tfr
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import polars as pl
from pathlib import Path
import numpy as np

# --- Constants ---
# The MBA model will take the two outputs from the "nerd" model (profit/loss predictions)
# plus any additional features we decide to add.
MBA_INPUT_FEATURES = 2 # Starting with just the two nerd outputs

def build_mba_model() -> tf.keras.Model:
    """
    Builds the Platypus MBA (Multi-Layer Perceptron - MLP) model.

    This model is designed for a Learning to Rank (LTR) task. Its goal is not to predict a value,
    but to output a single numerical score for each options contract it sees. A higher score
    indicates a more promising opportunity. [cite: 31, 35]

    The architecture is a simple but powerful Feedforward Neural Network (also called an MLP),
    which is excellent for the kind of structured, tabular data we'll be feeding it. [cite: 32, 33]
    """
    # Define the input layer. It will receive a flat vector of features
    # (e.g., the profit/loss predictions from the nerd model).
    input_layer = Input(shape=(MBA_INPUT_FEATURES,), name='mba_input')

    # Define the hidden layers of the network. These layers find complex, non-linear
    # patterns in the input features.
    dense_1 = Dense(16, activation='relu', name='mba_dense_1')(input_layer)
    dense_2 = Dense(8, activation='relu', name='mba_dense_2')(dense_1)

    # --- The Output Layer ---
    # The final layer is a single neuron with no activation function (linear).
    # This neuron outputs the raw numerical score we'll use for ranking. [cite: 34, 35]
    output_score = Dense(1, name='relevance_score')(dense_2)

    # Build the Keras model
    model = Model(inputs=input_layer, outputs=output_score, name="Platypus_MBA")

    return model

def _train_mba_on_single_file(
    mba_model: tf.keras.Model, 
    nerd_model: tf.keras.Model, 
    file_path: Path, 
    epochs: int, 
    batch_size: int
):
    """
    A helper function that encapsulates the logic for training the MBA model on a single file.
    This ensures data is loaded and discarded from memory one file at a time.
    """
    # 1. Load data and generate features for this file only.
    df = pl.read_parquet(file_path)
    time_series_data = np.array(df['time_series_data'].to_list(), dtype=np.float32)
    ticker_data = np.array(df['ticker_embedding'].to_list(), dtype=np.int32).reshape(-1, 1)
    option_type_data = np.array(df['option_type_embedding'].to_list(), dtype=np.int32).reshape(-1, 1)
    nerd_inputs = {'time_series_input': time_series_data, 'ticker_input': ticker_data, 'option_type_input': option_type_data}
    predicted_profits, predicted_losses = nerd_model.predict(nerd_inputs)
    
    mba_features = np.concatenate([predicted_profits, predicted_losses], axis=1)
    true_relevance = np.array(df['predicted_profit_percentage'].to_list(), dtype=np.float32)

    # 2. Re-compile the model to reset the optimizer.
    ranking_loss = tfr.keras.losses.ApproxNDCGLoss()
    mba_model.compile(optimizer='adam', loss=ranking_loss)
    
    true_relevance_reshaped = tf.expand_dims(true_relevance, axis=1)
    
    # 3. Fit the model on this file's data.
    early_stopping = EarlyStopping(monitor='loss', patience=5, mode='min', restore_best_weights=True)
    mba_model.fit(
        mba_features,
        true_relevance_reshaped,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )
    # When this function returns, the large data arrays are cleared from memory.

def train_mba_model(
    data_directory: str, 
    nerd_model_path: str, 
    mba_model_save_path: str,
    epochs=30, 
    batch_size=32
):
    """
    Main training function for the MBA model, now training safely file-by-file.
    """
    print("---  MBA Model Training Workflow ---")
    
    model_path = Path(mba_model_save_path)
    if model_path.exists():
        print(f"Existing MBA model found at {mba_model_save_path}. Loading for fine-tuning.")
        mba_model = tf.keras.models.load_model(mba_model_save_path, compile=False)
    else:
        print("No existing MBA model found. Building a new model from scratch.")
        mba_model = build_mba_model()

    print("Loading 'nerd' model to generate features for MBA model...")
    nerd_model = tf.keras.models.load_model(nerd_model_path)

    path = Path(data_directory)
    parquet_files = sorted(list(path.glob('*_options_training_data.parquet')))
    
    if not parquet_files:
        raise FileNotFoundError(f"No training files found in {data_directory}")

    # --- *** THE FIX IS HERE: CLEANER, MEMORY-SAFE LOOP *** ---
    for i, file_path in enumerate(parquet_files):
        print(f"\n--- Training MBA on file {i+1}/{len(parquet_files)}: {file_path.name} ---")
        _train_mba_on_single_file(mba_model, nerd_model, file_path, epochs, batch_size)
    
    print(f"--- MBA Model training finished ---")
    return mba_model