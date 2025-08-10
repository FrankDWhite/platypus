#!/usr/bin/python3.10

from datetime import timedelta
import requests
import pprint
from typing import List, Dict, Any
from pymongo import ASCENDING

# --- Import Project Utilities ---
from mongodb_utilities import OptionsRawDataStore
from mcp_server import OptionVector, InferenceRequest
# We need the feature window size for our query
from platypus_nerd_utilities import TIME_STEPS, NUM_FEATURES

# --- Configuration ---
MCP_SERVER_URL = "http://127.0.0.1:8000/v1/infer"

def prepare_and_run_inference(inserted_document_ids: List[str]):
    """
    Takes a list of new document IDs, assembles the full historical feature windows
    for each, and sends them to the MCP server for inference.
    """
    print("\n--- 🤖 Starting Platypus Trading Machine 🤖 ---")
    
    if not inserted_document_ids:
        print("No new documents to process for inference. Exiting.")
        return

    db = OptionsRawDataStore()
    if not db.client:
        print("FATAL: Could not connect to MongoDB. Aborting inference.")
        return
        
    current_documents = db.find_by_ids(inserted_document_ids)
    
    if not current_documents:
        print("Could not retrieve the specified documents for inference. Exiting.")
        db.close_connection()
        return

    inference_vectors = []
    print(f"Assembling historical feature windows for {len(current_documents)} contract(s)...")

    for doc in current_documents:
        
        # --- *** CORRECTED AND SIMPLIFIED LOGIC *** ---
        # As you correctly pointed out, we can get the data in the correct order directly.
        
        # 1. Define the start time for the historical window. A 4-hour buffer is safe.
        feature_window_start = doc['intervalTimestamp'] - timedelta(hours=4)

        # 2. Query the database for the last 40 data points in ASCENDING order.
        feature_window = db.find_option_contract_time_series(
            symbol=doc["underlyingSymbol"],
            expiration_date=doc["expirationDate"],
            strike_price=doc["strikePrice"],
            option_type=doc["optionType"],
            start_timestamp=feature_window_start, # Correctly use start_timestamp
            end_timestamp=doc['intervalTimestamp'],
            sort_order=ASCENDING # Ask for the data in the correct chronological order
        )
        
        # We only want the last 40 points from that query.
        if len(feature_window) >= TIME_STEPS:
            feature_window = feature_window[-TIME_STEPS:]
        else:
            print(f"  -> Skipping a window for {doc['overallDescription']} - not enough historical data.")
            continue
        # --- *** END OF CORRECTION *** ---

        time_series_for_inference = [d.get("normalizedData", []) for d in feature_window]
        
        if not all(isinstance(d, list) and len(d) == NUM_FEATURES for d in time_series_for_inference):
             print(f"  -> Skipping a window for {doc['overallDescription']} - corrupted normalizedData.")
             continue

        raw_features = {
            "description": doc.get("overallDescription"),
            "delta": doc.get("delta"),
            "strikePrice": doc.get("strikePrice"),
            "hoursToExpiration": doc.get("hoursToExpiration")
        }

        vector = OptionVector(
            time_series_data=time_series_for_inference,
            ticker=doc["underlyingSymbol"],
            option_type=doc["optionType"],
            raw_features=raw_features
        )
        inference_vectors.append(vector)

    db.close_connection()

    if not inference_vectors:
        print("No valid feature windows could be constructed. Exiting.")
        return

    inference_payload = InferenceRequest(vectors=inference_vectors)
    request_data = inference_payload.dict()
    
    print(f"\nSending {len(inference_vectors)} vectors to the MCP server for inference...")

    try:
        response = requests.post(MCP_SERVER_URL, json=request_data)
        response.raise_for_status()
        ranked_results = response.json()
        
        print("\n--- ✅ MCP Server Response Received ---")
        pprint.pprint(ranked_results)
        print("------------------------------------")

    except requests.exceptions.RequestException as e:
        print(f"\n--- ❌ ERROR: Could not connect to the MCP server. ---")
        print(f"Please ensure the mcp_server.py is running. Details: {e}")
    
    print("--- Trading Machine Finished ---")