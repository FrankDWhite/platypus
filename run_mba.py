#!/usr/bin/python3.10

import numpy as np
import tensorflow as tf
import os

from platypus_mba_utilities import train_mba_model

TRAINING_DATA_DIRECTORY = "./training_data/training_data_single_example/"
NERD_MODEL_PATH = "./models/platypus_nerd_v1.keras"
MBA_MODEL_SAVE_PATH = "./models/platypus_mba_v1.keras"
os.makedirs("./models", exist_ok=True)

def main():
    print("--- 🚀 Starting Platypus MBA Workflow ---")

    # The function now returns the final, best model object.
    trained_mba_model = train_mba_model(
        data_directory=TRAINING_DATA_DIRECTORY,
        nerd_model_path=NERD_MODEL_PATH,
        mba_model_save_path=MBA_MODEL_SAVE_PATH,
        epochs=5,
        batch_size=1
    )

    # Perform the authoritative save.
    print(f"Saving final, fine-tuned MBA model to {MBA_MODEL_SAVE_PATH}...")
    trained_mba_model.save(MBA_MODEL_SAVE_PATH)
    print("MBA model saved successfully.")

    try:
        platypus_mba_model = tf.keras.models.load_model(MBA_MODEL_SAVE_PATH)
        print(f"Successfully loaded trained MBA model from: {MBA_MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"Error: Could not load the MBA model from {MBA_MODEL_SAVE_PATH}. Details: {e}")
        return

    print("\n--- Preparing Sample Data for MBA Batch Inference ---")
    sample_nerd_outputs = np.array([
        [15.5, -5.2], [-2.1, -10.8], [5.3, -4.5], [25.1, -15.0], [0.5, -1.2]
    ], dtype=np.float32)

    print("\n--- Performing MBA Batch Inference ---")
    predicted_scores = platypus_mba_model.predict(sample_nerd_outputs)

    print("\n--- ✅ MBA Inference Results ---")
    results = []
    for i, score in enumerate(predicted_scores):
        results.append({
            "input_nerd_prediction": sample_nerd_outputs[i],
            "mba_ranking_score": float(score[0])
        })
    
    ranked_results = sorted(results, key=lambda x: x['mba_ranking_score'], reverse=True)
    
    print("Ranked Opportunities (Highest Score is Best):")
    for res in ranked_results:
        print(f"  - Score: {res['mba_ranking_score']:.4f} (from Nerd Profit: {res['input_nerd_prediction'][0]:.2f} / Loss: {res['input_nerd_prediction'][1]:.2f})")

    print("\n--- MBA Workflow Complete ---")

if __name__ == "__main__":
    main()