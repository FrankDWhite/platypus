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

def train_mba_model(
    data_directory: str, 
    nerd_model_path: str, 
    mba_model_save_path: str,
    epochs=30, 
    batch_size=32
):
    """
    Main training function for the Platypus MBA ranking model.

    This function simulates the full data pipeline for training:
    1. Loads the labeled data from the Parquet files.
    2. Uses the "nerd" model to generate predictions for that data.
    3. Uses the "nerd" predictions as input features for the MBA model.
    4. Defines the "true" relevance score for each option (e.g., the actual profit).
    5. Trains the MBA model using a specialized ranking loss function.
    """
    print("---  MBA Model Training Workflow ---")
    
    # --- Step 1: Load Labeled Data ---
    # This is the same data used to train the nerd model.
    path = Path(data_directory)
    parquet_files = list(path.glob('*_options_training_data.parquet'))
    if not parquet_files:
        raise FileNotFoundError(f"No training files found in {data_directory}")
    df = pl.read_parquet(parquet_files)

    # --- Step 2: Generate Features with the Nerd Model ---
    print("Loading 'nerd' model to generate features for MBA model...")
    nerd_model = tf.keras.models.load_model(nerd_model_path)
    
    # Prepare inputs for the nerd model
    time_series_data = np.array(df['time_series_data'].to_list(), dtype=np.float32)
    ticker_data = np.array(df['ticker_embedding'].to_list(), dtype=np.int32).reshape(-1, 1)
    option_type_data = np.array(df['option_type_embedding'].to_list(), dtype=np.int32).reshape(-1, 1)

    nerd_inputs = {
        'time_series_input': time_series_data,
        'ticker_input': ticker_data,
        'option_type_input': option_type_data
    }

    # Use the nerd model to predict profit and loss. These predictions will be the
    # input features for the MBA model. [cite: 16]
    predicted_profits, predicted_losses = nerd_model.predict(nerd_inputs)
    
    # This is our feature set (X) for the MBA model.
    mba_features = np.concatenate([predicted_profits, predicted_losses], axis=1)

    # --- Step 3: Define True Relevance Labels ---
    # For a ranking task, the label is the "true" score we want the model to learn to predict.
    # In our case, the actual profit percentage is a perfect relevance score.
    true_relevance = np.array(df['predicted_profit_percentage'].to_list(), dtype=np.float32)

    # --- Step 4: Build and Train the MBA model ---
    mba_model = build_mba_model()
    
    # --- CRITICAL: COMPILE WITH RANKING LOSS ---
    # This is the key to making this a ranking model. [cite: 42]
    # We use ApproxNDCGLoss, which directly optimizes the model to push items with the
    # highest true relevance scores to the top of the list. This perfectly matches our goal. [cite: 44, 45, 46]
    ranking_loss = tfr.keras.losses.ApproxNDCGLoss()
    
    # We need to reshape our data for the ranking loss function. It expects
    # a "list" of items, but since we are evaluating each item individually,
    # we can treat each one as a list of size 1.
    mba_features = tf.expand_dims(mba_features, axis=1)
    true_relevance = tf.expand_dims(true_relevance, axis=1)

    mba_model.compile(optimizer='adam', loss=ranking_loss)
    
    print("Starting MBA model training...")
    
    # Set up callbacks
    checkpoint = ModelCheckpoint(filepath=mba_model_save_path, save_best_only=True, monitor='loss', mode='min')
    early_stopping = EarlyStopping(monitor='loss', patience=5, mode='min', restore_best_weights=True)

    mba_model.fit(
        mba_features,
        true_relevance,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stopping]
    )
    
    print(f"--- MBA Model training finished. Best model saved to {mba_model_save_path} ---")