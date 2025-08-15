import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
import tensorflow as tf
import threading

# --- Import Project Utilities ---
from platypus_nerd_utilities import train_model as train_nerd_model, load_trained_model
from platypus_mba_utilities import train_mba_model
from embedding_utils import get_ticker_encoding, get_option_type_encoding

# --- Server & Model Configuration ---
NERD_MODEL_PATH = "./models/platypus_nerd_v1.keras"
MBA_MODEL_PATH = "./models/platypus_mba_v1.keras"
NERD_MODEL_SAVE_PATH = "./models/via_mcp/platypus_nerd_v1.keras"
MBA_MODEL_SAVE_PATH = "./models/via_mcp/platypus_mba_v1.keras"
TRAINING_DATA_DIR = "./training_data/training_data_single_example/"

# --- Global Model Storage ---
# We will store the loaded models in global variables.
# The server loads the models into memory only once on startup. [cite: 23]
platypus_nerd_model: tf.keras.Model = None
platypus_mba_model: tf.keras.Model = None

# --- FastAPI Application ---
app = FastAPI(
    title="Platypus MCP (Master Control Program) Server",
    description="Unified inference server for the Platypus Nerd (LSTM) and MBA (Ranking) models."
)

# --- Pydantic Models for API Data Validation ---
# This class defines the structure of a single options contract for inference.
class OptionVector(BaseModel):
    time_series_data: List[List[float]] # The (40, 11) normalized data vector
    ticker: str
    option_type: str
    # This dictionary holds the extra, non-normalized data for post-inference filtering. [cite: 13]
    raw_features: Dict[str, Any] 

# This class defines the structure of the main inference request payload.
class InferenceRequest(BaseModel):
    # The request will contain a batch of ~120 option vectors. [cite: 11]
    vectors: List[OptionVector]

# --- Server Startup Event ---
@app.on_event("startup")
def load_models():
    """
    This function runs automatically when the FastAPI server starts.
    It loads both the nerd and MBA models into the global variables.
    """
    global platypus_nerd_model, platypus_mba_model
    print("--- 🧠 Loading models into memory... ---")
    try:
        platypus_nerd_model = load_trained_model(NERD_MODEL_PATH)
        platypus_mba_model = load_trained_model(MBA_MODEL_PATH)
        print("--- ✅ Models loaded successfully. Server is ready. ---")
    except Exception as e:
        print(f"🔥🔥🔥 FATAL ERROR: Could not load models on startup. {e} 🔥🔥🔥")
        # In a real production system, you might want the server to exit if models can't load.
        platypus_nerd_model = None
        platypus_mba_model = None

# --- API Endpoints ---

@app.post("/v1/infer")
def perform_inference(request: InferenceRequest):
    """
    The main inference endpoint. It processes a batch of option vectors through
    the full nerd -> MBA pipeline and returns a ranked list. [cite: 10, 17]
    """
    if not platypus_nerd_model or not platypus_mba_model:
        raise HTTPException(status_code=503, detail="Models are not loaded. Server is not ready.")

    # --- 1. Prepare data for the Nerd model ---
    nerd_time_series_batch = [v.time_series_data for v in request.vectors]
    nerd_ticker_batch = [get_ticker_encoding(v.ticker) for v in request.vectors]
    nerd_option_type_batch = [get_option_type_encoding(v.option_type) for v in request.vectors]

    nerd_inputs = {
        'time_series_input': np.array(nerd_time_series_batch),
        'ticker_input': np.array(nerd_ticker_batch).reshape(-1, 1),
        'option_type_input': np.array(nerd_option_type_batch).reshape(-1, 1)
    }

    # --- 2. Run Nerd Inference ---
    predicted_profits, predicted_losses = platypus_nerd_model.predict(nerd_inputs)

    # --- 3. Prepare data for the MBA model ---
    # The input for the MBA model is the output from the nerd model. [cite: 16]
    mba_features = np.concatenate([predicted_profits, predicted_losses], axis=1)

    # --- 4. Run MBA Inference ---
    mba_scores = platypus_mba_model.predict(mba_features)

    # --- 5. Combine, Rank, and Filter ---
    results = []
    for i, vector in enumerate(request.vectors):
        results.append({
            "options_contract": vector.raw_features, # Include the original human-readable data
            "nerd_predicted_profit": float(predicted_profits[i][0]),
            "nerd_predicted_loss": float(predicted_losses[i][0]),
            "mba_ranking_score": float(mba_scores[i][0])
        })
    
    # Sort the results in descending order based on the MBA model's score. [cite: 37]
    ranked_results = sorted(results, key=lambda x: x['mba_ranking_score'], reverse=True)
    
    return {"ranked_recommendations": ranked_results}

def trigger_training(model_type: str):
    """
    Helper function to run training in a background thread to avoid blocking the server.
    """
    if model_type == "nerd":
        print("--- Triggering 'nerd' model training in the background. ---")
        trained_nerd = train_nerd_model(data_directory=TRAINING_DATA_DIR, model_save_path=NERD_MODEL_PATH)
        trained_nerd.save(NERD_MODEL_SAVE_PATH)
        print("--- 'Nerd' model training finished. Overwriting complete. Please restart the server to load the new model. ---")
    elif model_type == "mba":
        print("--- Triggering 'mba' model training in the background. ---")
        trained_mba = train_mba_model(data_directory=TRAINING_DATA_DIR, nerd_model_path=NERD_MODEL_PATH, mba_model_save_path=MBA_MODEL_PATH)
        trained_mba.save(MBA_MODEL_SAVE_PATH)
        print("--- 'MBA' model training finished. Overwriting complete. Please restart the server to load the new model. ---")


@app.post("/v1/train-nerd")
def endpoint_train_nerd():
    """
    API endpoint to trigger a fine-tuning job for the nerd model.
    The training runs in a background thread.
    """
    # Using a thread to avoid blocking the API while training runs.
    training_thread = threading.Thread(target=trigger_training, args=("nerd",))
    training_thread.start()
    return {"message": "Nerd model training has been initiated. The new model will be available after the server is restarted."} [cite: 25, 26]

@app.post("/v1/train-mba")
def endpoint_train_mba():
    """
    API endpoint to trigger a fine-tuning job for the MBA model.
    """
    training_thread = threading.Thread(target=trigger_training, args=("mba",))
    training_thread.start()
    return {"message": "MBA model training has been initiated. The new model will be available after the server is restarted."} [cite: 25, 26]


# --- To run this server: ---
# 1. Save the file as mcp_server.py
# 2. Open a terminal and run: uvicorn mcp_server:app --reload
if __name__ == "__main__":
    # This allows you to run the server directly for testing.
    # The --reload flag makes the server automatically restart when you save changes to the file.
    uvicorn.run("mcp_server:app", host="127.0.0.1", port=8000, reload=True)