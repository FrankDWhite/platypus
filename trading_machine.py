#!/usr/bin/python3.10

from datetime import datetime, timedelta, timezone
import time
import requests
import pprint
from typing import List, Dict, Any
from pymongo import ASCENDING, DESCENDING

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
            "hoursToExpiration": doc.get("hoursToExpiration"),
            "current_price": doc.get("optionPrice"),
            "underlyingSymbol": doc.get("underlyingSymbol"),
            "expirationDate": str(doc.get("expirationDate")),
            "optionType": doc.get("optionType")
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
        if ranked_results and ranked_results.get("ranked_recommendations"):
            top_trade = ranked_results["ranked_recommendations"][0]
            
            # Prepare the document to be saved in our new collection.
            trade_to_simulate = {
                "recommendation_time": datetime.now(timezone.utc),
                "entry_price": top_trade["options_contract"]["current_price"],
                # We need these fields to uniquely identify the contract later.
                "underlyingSymbol": top_trade["options_contract"]["underlyingSymbol"],
                "expirationDate": datetime.strptime(top_trade["options_contract"]["expirationDate"], '%Y-%m-%d').replace(tzinfo=timezone.utc),
                "strikePrice": top_trade["options_contract"]["strikePrice"],
                "optionType": top_trade["options_contract"]["optionType"],
                "result": None # This will be updated by the new function.
            }
            
            # Connect to the new collection and insert the trade.
            sim_db = OptionsRawDataStore(collection_name="simulated_open_trades")
            sim_db.collection.insert_one(trade_to_simulate)
            sim_db.close_connection()
            print(f"\n--- 📈 Saved top recommendation to 'simulated_open_trades' ---")

    except requests.exceptions.RequestException as e:
        print(f"\n--- ❌ ERROR: Could not connect to the MCP server. ---")
        print(f"Please ensure the mcp_server.py is running. Details: {e}")
    
    print("--- Trading Machine Finished ---")

def update_profit_loss_on_open_trades():
    """
    Checks all simulated trades opened in the last 4 hours and updates their
    current profit or loss based on the latest market price.
    """
    print("\n--- ⏱️ Updating P&L on Simulated Open Trades ---")
    sim_db = OptionsRawDataStore(collection_name="simulated_open_trades")
    if not sim_db.client:
        print("FATAL: Could not connect to simulated_open_trades collection.")
        return

    # Find all trades from the last 4 hours that haven't been closed out.
    four_hours_ago = datetime.now(timezone.utc) - timedelta(hours=4)
    open_trades = list(sim_db.collection.find({"recommendation_time": {"$gte": four_hours_ago}}))

    if not open_trades:
        print("No open trades to update.")
        sim_db.close_connection()
        return
        
    print(f"Found {len(open_trades)} open trades to update.")
    
    # We need to connect to the main data collection to get the latest prices.
    data_db = OptionsRawDataStore()
    if not data_db.client:
        print("FATAL: Could not connect to option_intervals collection.")
        sim_db.close_connection()
        return

    for trade in open_trades:
        # For each open trade, find the most recent data point in our main collection.
        latest_data_point = data_db.collection.find_one(
            {
                "underlyingSymbol": trade["underlyingSymbol"],
                "expirationDate": trade["expirationDate"],
                "strikePrice": trade["strikePrice"],
                "optionType": trade["optionType"],
            },
            sort=[("intervalTimestamp", DESCENDING)]
        )
        
        if latest_data_point and "optionPrice" in latest_data_point:
            current_price = latest_data_point["optionPrice"]
            entry_price = trade["entry_price"]

            if entry_price > 0:
                # Calculate the percentage change since the recommendation was made.
                percent_change = ((current_price - entry_price) / entry_price) * 100
                
                # Update the "result" field for this trade in the database.
                sim_db.collection.update_one(
                    {"_id": trade["_id"]},
                    {"$set": {"result": percent_change}}
                )
                print(f"  -> Updated {trade['description']}: Current P&L is {percent_change:.2f}%")

    sim_db.close_connection()
    data_db.close_connection()
    print("--- P&L Update Complete ---")

def trigger_daily_retrain(current_day_data_path: str):
    """
    Orchestrates the end-to-end daily training and server restart workflow
    by making a single, non-blocking API call.
    """
    print("\n--- 🚀 Starting Automated MCP Training Workflow ---")
    
    try:
        # --- Step 1: Trigger the entire workflow with one "fire and forget" call ---
        print("\n1. Triggering full Platypus fine-tuning and restart workflow...")
        response = requests.post(
            f"http://127.0.0.1:8000/v1/trigger-full-retraining-workflow",
            json={"data_directory": current_day_data_path}
            # No timeout is needed as the server responds instantly.
        )
        response.raise_for_status()
        print(f"  -> Server response: {response.json()['message']}")
        print("\n--- ✅ MCP Workflow Triggered Successfully ---")
        print("The server will now train both models and restart in the background.")

    except requests.exceptions.RequestException as e:
        print(f"\n--- ❌ ERROR: Could not connect to the MCP server to trigger workflow. ---")
        print(f"Please ensure the mcp_server.py is running. Details: {e}")
