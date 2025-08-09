#!/usr/bin/python3.10

import numpy as np
import tensorflow as tf

# --- Import project utilities ---
from platypus_mba_utilities import train_mba_model

# --- Configuration ---
# Directory with the single training example to run the test.
TRAINING_DATA_DIRECTORY = "./training_data/training_data_single_example/"

# The path to the already-trained "nerd" model, which is required to generate features for the MBA model.
NERD_MODEL_PATH = "./models/platypus_nerd_v1.keras"

# The path where the script will save the newly trained MBA model.
MBA_MODEL_SAVE_PATH = "./models/platypus_mba_v1.keras"

def main():
    """
    This main function orchestrates the entire MBA model workflow:
    1. Trains the MBA model using the specified data and a pre-trained nerd model.
    2. Loads the best saved MBA model back from disk.
    3. Runs a sample batch prediction (inference) to demonstrate its use.
    """
    print("--- 🚀 Starting Platypus MBA Workflow ---")

    # --- 1. Train the MBA Model ---
    # This function handles the entire MBA training pipeline, including using the
    # nerd model to generate the features it needs.
    train_mba_model(
        data_directory=TRAINING_DATA_DIRECTORY,
        nerd_model_path=NERD_MODEL_PATH,
        mba_model_save_path=MBA_MODEL_SAVE_PATH,
        epochs=5, # A few epochs are enough for a test run
        batch_size=1
    )

    # --- 2. Load the Trained MBA Model ---
    try:
        # We can use TensorFlow's generic model loader for our MBA model.
        platypus_mba_model = tf.keras.models.load_model(MBA_MODEL_SAVE_PATH)
        print(f"Successfully loaded trained MBA model from: {MBA_MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"Error: Could not load the MBA model from {MBA_MODEL_SAVE_PATH}. Details: {e}")
        return

    # --- 3. Prepare a Sample Batch for Inference ---
    # To test the MBA model, we need to simulate the output of the "nerd" model.
    # We'll create a dummy batch of 5 "nerd" predictions (profit, loss).
    print("\n--- Preparing Sample Data for MBA Batch Inference ---")
    
    # Each row is a [predicted_profit, predicted_loss] pair.
    sample_nerd_outputs = np.array([
        [15.5, -5.2],   # A very promising prediction
        [-2.1, -10.8],  # A poor prediction
        [5.3, -4.5],    # A moderately good prediction
        [25.1, -15.0],  # An excellent but risky prediction
        [0.5, -1.2]     # A neutral prediction
    ], dtype=np.float32)

    # --- 4. Perform Batch Inference ---
    print("\n--- Performing MBA Batch Inference ---")
    # The MBA model takes the (n, 2) NumPy array directly as input.
    predicted_scores = platypus_mba_model.predict(sample_nerd_outputs)

    # --- 5. Display and Rank the Results ---
    print("\n--- ✅ MBA Inference Results ---")
    # Combine the inputs with their predicted scores
    results = []
    for i, score in enumerate(predicted_scores):
        results.append({
            "input_nerd_prediction": sample_nerd_outputs[i],
            "mba_ranking_score": float(score[0])
        })
    
    # Sort the results by the MBA score to see the final ranking
    ranked_results = sorted(results, key=lambda x: x['mba_ranking_score'], reverse=True)
    
    print("Ranked Opportunities (Highest Score is Best):")
    for res in ranked_results:
        print(f"  - Score: {res['mba_ranking_score']:.4f} (from Nerd Profit: {res['input_nerd_prediction'][0]:.2f} / Loss: {res['input_nerd_prediction'][1]:.2f})")

    print("\n--- MBA Workflow Complete ---")


if __name__ == "__main__":
    main()