import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Masking
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import polars as pl
from pathlib import Path
import numpy as np
from typing import List, Dict, Any

# --- Import project-specific utilities ---
# We need the vocabulary sizes to correctly define our Embedding layers.
from embedding_utils import get_vocabulary_size

# --- Constants ---
# These constants help us define the "shape" of our input data, making the code cleaner.
# The time series data for each example has 40 sequential data points (time steps).
TIME_STEPS = 40
# Each of those 40 data points has 11 normalized financial features.
NUM_FEATURES = 11

def build_lstm_model(embedding_dim_ticker=4, embedding_dim_option_type=2) -> tf.keras.Model:
    """
    Builds the Platypus Nerd's brain: a multi-input LSTM model architecture.

    ## Why a Multi-Input Model?
    Our model needs to understand three different kinds of information at once:
    1.  **Sequential Data**: The recent history of an option's financial metrics (like price, volatility, etc.). This is where the LSTM shines, as it's designed to find patterns in sequences.
    2.  **Categorical Data (Ticker)**: Which stock is this option for (e.g., NVDA vs. AAPL)? The behavior of an NVDA option is different from an AAPL option.
    3.  **Categorical Data (Option Type)**: Is this a CALL or a PUT? These are fundamentally different instruments.

    This function defines the "blueprint" of the neural network, connecting all the layers together.

    Args:
        embedding_dim_ticker (int): The size of the dense vector for representing tickers. A larger number allows the model to learn more complex relationships between tickers, but can increase complexity. 4 is a reasonable starting point.
        embedding_dim_option_type (int): The size of the vector for representing the option type. 2 is suitable since there are only two types (CALL/PUT).

    Returns:
        tf.keras.Model: The compiled, untrained Keras model, ready for training.
    """
    # =================================================================================
    # STEP 1: DEFINE THE MODEL'S "DOORS" (INPUT LAYERS)
    # =================================================================================
    # We need to tell the model what kind of data to expect. We define three separate inputs.

    # Input for the time series data. Its shape is (40 time steps, 11 features).
    # 'None' would typically go where 40 is to allow variable-length sequences, but since ours are fixed, we specify it.
    time_series_input = Input(shape=(TIME_STEPS, NUM_FEATURES), name='time_series_input')

    # Input for the stock ticker's integer ID (e.g., 1 for MSFT, 2 for NVDA, etc.). Shape is (1,) because it's a single number.
    ticker_input = Input(shape=(1,), name='ticker_input')

    # Input for the option type's integer ID (e.g., 1 for CALL, 2 for PUT).
    option_type_input = Input(shape=(1,), name='option_type_input')

    # =================================================================================
    # STEP 2: TEACH THE MODEL ABOUT CATEGORIES (EMBEDDING LAYERS)
    # =================================================================================
    # A neural network doesn't understand "NVDA" or "CALL". It only understands numbers.
    # An Embedding layer is a smart lookup table that turns a simple integer ID (like 2 for "NVDA")
    # into a dense vector of numbers (e.g., [0.65, -0.23, 0.89, -0.41]).
    # WHY THIS IS IMPORTANT: The model *learns* the best vector representation for each category
    # during training. This allows it to discover similarities. For example, it might learn
    # that the vectors for GOOGL and AMZN are more similar to each other than to TSLA.

    ticker_vocab_size = get_vocabulary_size("ticker")
    ticker_embedding_layer = Embedding(input_dim=ticker_vocab_size, output_dim=embedding_dim_ticker, name='ticker_embedding')
    ticker_embedding = ticker_embedding_layer(ticker_input)
    # The output of an embedding is a 3D tensor (batch, 1, dim). We flatten it to 2D to combine it with other layers.
    ticker_embedding_flat = tf.keras.layers.Flatten()(ticker_embedding)

    option_type_vocab_size = get_vocabulary_size("option_type")
    option_type_embedding_layer = Embedding(input_dim=option_type_vocab_size, output_dim=embedding_dim_option_type, name='option_type_embedding')
    option_type_embedding = option_type_embedding_layer(option_type_input)
    option_type_embedding_flat = tf.keras.layers.Flatten()(option_type_embedding)

    # =================================================================================
    # STEP 3: PROCESS THE TIME SERIES DATA (LSTM LAYERS)
    # =================================================================================
    # This is the core of the model. An LSTM (Long Short-Term Memory) layer is a type of
    # Recurrent Neural Network (RNN) that is excellent at finding patterns in sequential data.

    # WHY MASKING IS IMPORTANT: Our data has gaps (pre-market, after-hours). The `masking_array`
    # tells us which time steps are just padding. The `Masking` layer ensures that the LSTM
    # completely ignores these time steps, preventing the model from learning from "fake" zeroed-out data.
    # We will use this masking layer later when we prepare the data. For now, we build the model assuming masking will be handled.
    
    # The LSTM processes the 40 time steps one by one, updating its internal "memory" (hidden state) at each step.
    # `return_sequences=True` makes the first LSTM layer output its hidden state at *every* time step.
    # This is useful for stacking LSTM layers, as the next layer needs a full sequence as input.
    lstm_out_1 = LSTM(64, return_sequences=True, name='lstm_layer_1')(time_series_input)

    # The second LSTM layer takes the sequence from the first one. By default (`return_sequences=False`),
    # it only outputs its final hidden state. This single vector represents the model's "summary"
    # of the entire 3.25-hour time series.
    lstm_out_2 = LSTM(32, name='lstm_layer_2')(lstm_out_1)

    # =================================================================================
    # STEP 4: COMBINE ALL KNOWLEDGE (CONCATENATE LAYER)
    # =================================================================================
    # Now we take the three processed pieces of information and merge them into a single, flat vector.
    # 1. The summary of the time series from the LSTM.
    # 2. The learned representation of the ticker from the Embedding layer.
    # 3. The learned representation of the option type from the Embedding layer.
    concatenated_features = Concatenate(name='concatenate_features')([
        lstm_out_2,
        ticker_embedding_flat,
        option_type_embedding_flat
    ])

    # =================================================================================
    # STEP 5: MAKE THE FINAL PREDICTION (DENSE LAYERS)
    # =================================================================================
    # A Dense layer is a standard, fully-connected neural network layer. It looks for patterns
    # in the combined feature vector.
    dense_layer = Dense(32, activation='relu', name='dense_layer_1')(concatenated_features)

    # We need two separate outputs, one for each target we want to predict.
    # Each output is a single neuron because we are predicting a single continuous value.
    profit_output = Dense(1, name='profit_output')(dense_layer)
    loss_output = Dense(1, name='loss_output')(dense_layer)

    # =================================================================================
    # STEP 6: BUILD AND COMPILE THE MODEL
    # =================================================================================
    # We define the final model by specifying its inputs and outputs.
    model = Model(
        inputs=[time_series_input, ticker_input, option_type_input],
        outputs=[profit_output, loss_output],
        name="Platypus_Nerd"
    )

    # Compilation configures the model for training.
    # - `optimizer='adam'`: Adam is a robust, commonly used optimization algorithm that adapts the learning rate during training.
    # - `loss`: This tells the model how to measure its error. 'mean_squared_error' is standard for regression tasks (predicting a continuous value). We define a separate loss for each output.
    # - `metrics`: These are additional metrics to monitor during training. RMSE (Root Mean Squared Error) is useful because it's in the same units as our target (percentage points).
    model.compile(
        optimizer='adam',
        loss={
            'profit_output': 'mean_squared_error',
            'loss_output': 'mean_squared_error'
        },
        metrics={
            'profit_output': tf.keras.metrics.RootMeanSquaredError(),
            'loss_output': tf.keras.metrics.RootMeanSquaredError()
        }
    )
    
    print("--- Model Architecture ---")
    model.summary()
    print("--------------------------")
    return model

def load_and_preprocess_data(directory_path: str) -> tuple:
    """
    Loads all Parquet training data, prepares it for the model, and splits it.
    """
    path = Path(directory_path)
    parquet_files = list(path.glob('*_options_training_data.parquet'))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in directory: {directory_path}")

    print(f"Found {len(parquet_files)} Parquet files. Loading...")
    df = pl.read_parquet(parquet_files)
    
    # --- Convert data to NumPy arrays ---
    # TensorFlow works with NumPy arrays, so we convert the columns from the DataFrame.
    # We also apply the masking here. When a mask value is True, we set the corresponding
    # time step's features to 0.0. The Masking layer in the model will then ignore it.
    time_series_list = df['time_series_data'].to_list()
    mask_list = df['masking_array'].to_list()
    
    # Apply the mask: set feature vectors to zero where the mask is True
    for i, mask_sequence in enumerate(mask_list):
        for j, is_masked in enumerate(mask_sequence):
            if is_masked:
                time_series_list[i][j] = [0.0] * NUM_FEATURES
                
    time_series_data = np.array(time_series_list, dtype=np.float32)
    ticker_data = np.array(df['ticker_embedding'].to_list(), dtype=np.int32).reshape(-1, 1)
    option_type_data = np.array(df['option_type_embedding'].to_list(), dtype=np.int32).reshape(-1, 1)
    
    profit_labels = np.array(df['predicted_profit_percentage'].to_list(), dtype=np.float32)
    loss_labels = np.array(df['predicted_loss_percentage'].to_list(), dtype=np.float32)
    
    # Package data into dictionaries matching the model's input/output names
    x_data = {
        'time_series_input': time_series_data,
        'ticker_input': ticker_data,
        'option_type_input': option_type_data
    }
    y_data = {
        'profit_output': profit_labels,
        'loss_output': loss_labels
    }
    
    # Split data into training (80%) and validation (20%) sets
    dataset_size = len(df)
    val_size = int(0.2 * dataset_size)
    indices = np.random.permutation(dataset_size)
    train_indices, val_indices = indices[val_size:], indices[:val_size]
    
    x_train = {key: value[train_indices] for key, value in x_data.items()}
    y_train = {key: value[train_indices] for key, value in y_data.items()}
    x_val = {key: value[val_indices] for key, value in x_data.items()}
    y_val = {key: value[val_indices] for key, value in y_data.items()}
    
    print(f"Data loaded and split into {len(train_indices)} training samples and {len(val_indices)} validation samples.")
    return x_train, y_train, x_val, y_val

def train_model(data_directory: str, model_save_path: str, epochs=50, batch_size=64):
    """
    Main training function to orchestrate the entire training process.
    """
    print("--- 🚀 Starting Platypus Nerd train_model Workflow ---")

    x_train, y_train, x_val, y_val = load_and_preprocess_data(data_directory)
    model = build_lstm_model()
    
    # --- Callbacks ---
    # Callbacks are utilities that can be applied at different stages of the training process.
    
    # `ModelCheckpoint`: Saves the model after every epoch where the validation loss improves.
    # This ensures that even if the training is interrupted, you will have the best version of the model saved.
    checkpoint = ModelCheckpoint(
        filepath=model_save_path,
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )
    # `EarlyStopping`: Monitors the validation loss and stops the training if it hasn't improved
    # for a set number of epochs (`patience`). This prevents overfitting and saves time.
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        verbose=1,
        restore_best_weights=True
    )
    
    print("\n--- Starting Model Training ---")
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )
    print("--- Model Training Finished ---")
    return history

def load_trained_model(model_path: str) -> tf.keras.Model:
    """
    Loads a previously saved Keras model from disk.
    """
    print(f"Loading trained model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
    return model

def perform_batch_inference(
    model: tf.keras.Model, 
    time_series_batch: List[List[List[float]]], 
    ticker_id_batch: List[int], 
    option_type_id_batch: List[int]
) -> List[tuple[float, float]]:
    """
    Performs inference on a BATCH of inputs at once for high efficiency.

    ## Why Batch Inference is Important
    Running `model.predict()` has some computational overhead. If you have 100 contracts to predict,
    calling the function 100 times in a loop is very slow. It's much faster to feed all 100 inputs
    to the model in a single batch and get all 100 predictions back at once. This is the standard
    way to perform high-throughput inference.

    Args:
        model (tf.keras.Model): The loaded, trained model.
        time_series_batch (List[List[List[float]]]): A list of time series inputs. 
                                                     Each element is a (40, 11) list/array.
        ticker_id_batch (List[int]): A list of ticker integer IDs.
        option_type_id_batch (List[int]): A list of option type integer IDs.

    Returns:
        A list of tuples, where each tuple contains the (predicted_profit, predicted_loss)
        for the corresponding input contract.
    """
    # Convert the lists of inputs into NumPy arrays with the correct shape.
    # The model expects a "batch" dimension, which is what this format provides.
    time_series_input = np.array(time_series_batch, dtype=np.float32)
    ticker_input = np.array(ticker_id_batch, dtype=np.int32).reshape(-1, 1)
    option_type_input = np.array(option_type_id_batch, dtype=np.int32).reshape(-1, 1)

    # Package the batch of inputs into a dictionary, just like in training.
    inference_input = {
        'time_series_input': time_series_input,
        'ticker_input': ticker_input,
        'option_type_input': option_type_input
    }
    
    # Perform the prediction on the entire batch in one go.
    predicted_profit_batch, predicted_loss_batch = model.predict(inference_input)
    
    # The output is two NumPy arrays of shape (batch_size, 1).
    # We need to zip them together to create a list of (profit, loss) tuples.
    results = []
    for profit, loss in zip(predicted_profit_batch, predicted_loss_batch):
        results.append((profit[0], loss[0]))
        
    return results