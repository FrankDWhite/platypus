#!/usr/bin/python3.10

from datetime import timedelta, timezone, datetime
import pickle
from pymongo import DESCENDING, ASCENDING
from pathlib import Path
import os

# --- Import Project Utilities ---
from mongodb_utilities import OptionsRawDataStore
from mcp_server import OptionVector
from platypus_nerd_utilities import TIME_STEPS, NUM_FEATURES

# --- Configuration ---
# The number of high-quality vectors we want to find and save.
NUM_VECTORS_TO_CREATE = 15
# The file path for the output batch file.
OUTPUT_VECTOR_FILE = Path("./models/inference_test_batch.pkl")

def create_and_save_test_vector_batch():
    """
    Finds a batch of high-quality data points, constructs their full feature windows,
    and saves them as a reusable test batch file.
    """
    print(f"--- 🔬 Creating a Reusable Inference Test Batch of {NUM_VECTORS_TO_CREATE} Vectors ---")
    
    db = OptionsRawDataStore()
    if not db.client:
        print("FATAL: Could not connect to MongoDB.")
        return

    # --- Step 1: Find a list of suitable candidate data points ---
    print("Searching for high-quality, unmasked data points in the database...")
    search_start_time = datetime.now(timezone.utc) - timedelta(hours=24)
    query = {
        "intervalTimestamp": {"$gte": search_start_time},
        "maskWhenTraining": False,
        "hasBeenLabeled": {"$ne": True}
    }
    
    # Fetch a larger number of candidates to increase our chances of building enough valid windows.
    candidate_docs = list(db.collection.find(query, sort=[("intervalTimestamp", DESCENDING)], limit=50))

    if not candidate_docs:
        print("Error: No suitable unmasked data points found from the last 24 hours.")
        db.close_connection()
        return
        
    print(f"Found {len(candidate_docs)} candidates. Attempting to build {NUM_VECTORS_TO_CREATE} valid feature windows...")

    test_vectors = []
    # --- Loop through candidates until we have enough vectors ---
    for i, current_doc in enumerate(candidate_docs):
        # Stop once we've collected the desired number of vectors.
        if len(test_vectors) >= NUM_VECTORS_TO_CREATE:
            break
        
        feature_window_start = current_doc['intervalTimestamp'] - timedelta(hours=4)
        feature_window = db.find_option_contract_time_series(
            symbol=current_doc["underlyingSymbol"],
            expiration_date=current_doc["expirationDate"],
            strike_price=current_doc["strikePrice"],
            option_type=current_doc["optionType"],
            start_timestamp=feature_window_start,
            end_timestamp=current_doc['intervalTimestamp'],
            sort_order=ASCENDING
        )
        
        if len(feature_window) >= TIME_STEPS:
            feature_window = feature_window[-TIME_STEPS:]
        else:
            continue

        time_series_for_inference = [d.get("normalizedData", []) for d in feature_window]
        
        if not all(isinstance(d, list) and len(d) == NUM_FEATURES for d in time_series_for_inference):
             continue

        raw_features = {
            "description": current_doc.get("overallDescription"),
            "delta": current_doc.get("delta"),
            "strikePrice": current_doc.get("strikePrice"),
            "hoursToExpiration": current_doc.get("hoursToExpiration")
        }

        vector = OptionVector(
            time_series_data=time_series_for_inference,
            ticker=current_doc["underlyingSymbol"],
            option_type=current_doc["optionType"],
            raw_features=raw_features
        )
        test_vectors.append(vector)
        print(f"  -> Successfully created vector {len(test_vectors)}/{NUM_VECTORS_TO_CREATE} from {current_doc['overallDescription']}")

    db.close_connection()

    # --- Step 4: Save the final list of objects to a file ---
    if len(test_vectors) >= NUM_VECTORS_TO_CREATE:
        os.makedirs(OUTPUT_VECTOR_FILE.parent, exist_ok=True)
        with open(OUTPUT_VECTOR_FILE, 'wb') as f:
            pickle.dump(test_vectors, f)
        print(f"\n--- ✅ Test vector batch saved successfully to: {OUTPUT_VECTOR_FILE} ---")
    else:
        print(f"\n--- ❌ Error: Failed to create enough test vectors. Only found {len(test_vectors)}/{NUM_VECTORS_TO_CREATE}. ---")

if __name__ == "__main__":
    create_and_save_test_vector_batch()