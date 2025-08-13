#!/usr/bin/python3.10

import numpy as np

# --- Import project utilities ---
# Import the functions we created for building, training, and running the model.
from platypus_nerd_utilities import (
    train_model, 
    load_trained_model, 
    perform_batch_inference,
    TIME_STEPS,       # We need these constants to create dummy data
    NUM_FEATURES
)
# Import the encoding functions to turn strings like "NVDA" into integer IDs for the model.
from embedding_utils import get_ticker_encoding, get_option_type_encoding

# --- Configuration ---
# This is the directory where you saved your single, high-quality training example.
# The training function will look for any .parquet files inside this directory.
TRAINING_DATA_DIRECTORY = "./training_data/training_data_single_example/"

# This is the file path where the script will save the best version of the trained model.
# The .keras format is the modern, recommended way to save TensorFlow models.
MODEL_SAVE_PATH = "./models/platypus_nerd_v1.keras"


def main():
    """
    This main function orchestrates the entire process:
    1. Trains the model on the data found in the training directory.
    2. Loads the best saved model back from the disk.
    3. Runs a sample batch prediction (inference) to demonstrate how to use the model.
    """
    print("--- 🚀 Starting Platypus Nerd Workflow ---")

    # --- 1. Train the Model ---
    # We call the main training function from our utility file.
    # This will handle loading the data, building the model, and running the training loop.
    # It will automatically save the best-performing model to `MODEL_SAVE_PATH`.
    trained_nerd_model = train_model(
        data_directory=TRAINING_DATA_DIRECTORY,
        model_save_path=MODEL_SAVE_PATH,
        epochs=5,  # For a single example, we only need a few epochs to see it run.
        batch_size=1
    )

        # --- Authoritative Save ---
    # We now explicitly save the final model object after training is complete.
    # This guarantees that the version with the restored best weights is saved.
    print(f"Saving final, fine-tuned model to {MODEL_SAVE_PATH}...")
    trained_nerd_model.save(MODEL_SAVE_PATH)
    print("Model saved successfully.")


    # --- 2. Load the Trained Model ---
    # In a real application, you would train the model once and then load the saved file
    # every time you need to make predictions. This simulates that process.
    try:
        platypus_nerd_model = load_trained_model(MODEL_SAVE_PATH)
    except Exception as e:
        print(f"Error: Could not load the model from {MODEL_SAVE_PATH}.")
        print(f"Please ensure the training step completed successfully. Details: {e}")
        return

    # --- 3. Prepare a Sample Batch for Inference ---
    # Here, we create a "dummy" batch of 3 option contracts to test the batch prediction function.
    # In a real scenario, this data would come from your live data retrieval script.
    print("\n--- Preparing Sample Data for Batch Inference ---")
    
    # Create 3 dummy time series inputs. Each one has the shape (40, 11).
    # We use random data here just for demonstration.
    sample_time_series_1 = np.random.rand(TIME_STEPS, NUM_FEATURES)
    sample_time_series_2 = np.random.rand(TIME_STEPS, NUM_FEATURES)
    sample_time_series_3 = np.random.rand(TIME_STEPS, NUM_FEATURES)

    # Package all the inputs into lists.
    batch_to_predict = {
        "time_series": [sample_time_series_1, sample_time_series_2, sample_time_series_3],
        "tickers": ["NVDA", "TSLA", "AAPL"],
        "option_types": ["CALL", "PUT", "CALL"]
    }

    # Use our utility functions to convert the string tickers and option types into the integer IDs the model expects.
    ticker_ids = [get_ticker_encoding(t) for t in batch_to_predict["tickers"]]
    option_type_ids = [get_option_type_encoding(ot) for ot in batch_to_predict["option_types"]]

    # --- 4. Perform Batch Inference ---
    # Call our optimized batch inference function to get predictions for all 3 contracts in one go.
    print("\n--- Performing Batch Inference ---")
    predictions = perform_batch_inference(
        model=platypus_nerd_model,
        time_series_batch=batch_to_predict["time_series"],
        ticker_id_batch=ticker_ids,
        option_type_id_batch=option_type_ids
    )

    # --- 5. Display the Results ---
    print("\n--- ✅ Inference Results ---")
    for i, (profit, loss) in enumerate(predictions):
        ticker = batch_to_predict["tickers"][i]
        opt_type = batch_to_predict["option_types"][i]
        print(f"Prediction for {ticker} {opt_type}:")
        print(f"  - Predicted Profit %: {profit:.2f}")
        print(f"  - Predicted Loss %:   {loss:.2f}\n")

    print("--- Workflow Complete ---")


if __name__ == "__main__":
    print("--- 🚀 Starting Platypus Nerd Workflow ---")

    main()