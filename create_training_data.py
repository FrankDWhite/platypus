#!/usr/bin/python3.10
import pandas as pd
import pymongo
from pymongo import MongoClient
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple

# --- Import the new embedding utility ---
from embedding_utils import get_ticker_encoding, get_option_type_encoding

# --- Configuration ---
# We need to look back further to build our feature + label windows
HOURS_TO_PROCESS = 48

# Window for input features (3.25 hours of past data + current point)
FEATURE_WINDOW_POINTS = 40

# Window for calculating the label (3.25 hours of future data)
LABELING_WINDOW_POINTS = 39

# Total contiguous points needed to create one training example
TOTAL_WINDOW_SIZE = FEATURE_WINDOW_POINTS + LABELING_WINDOW_POINTS


def get_unique_contracts(db_collection, start_time: datetime) -> List[Dict]:
    """
    Finds all unique option contracts that have data points since the start_time.
    """
    pipeline = [
        {"$match": {"intervalTimestamp": {"$gte": start_time}}},
        {
            "$group": {
                "_id": {
                    "underlyingSymbol": "$underlyingSymbol",
                    "expirationDate": "$expirationDate",
                    "strikePrice": "$strikePrice",
                    "optionType": "$optionType",
                }
            }
        },
        {"$replaceRoot": {"newRoot": "$_id"}}
    ]
    unique_contracts = list(db_collection.aggregate(pipeline))
    print(f"Found {len(unique_contracts)} unique contracts in the last {HOURS_TO_PROCESS} hours.")
    return unique_contracts

def fetch_contract_time_series(db_collection, contract: Dict, start_time: datetime, end_time: datetime) -> List[Dict]:
    """
    Fetches the time series for a single contract within a given time window, sorted chronologically.
    """
    query = {
        "underlyingSymbol": contract["underlyingSymbol"],
        "expirationDate": contract["expirationDate"],
        "strikePrice": contract["strikePrice"],
        "optionType": contract["optionType"],
        "intervalTimestamp": {"$gte": start_time, "$lte": end_time}
    }
    cursor = db_collection.find(query).sort("intervalTimestamp", pymongo.ASCENDING)
    return list(cursor)

def label_data_point(current_price: float, future_data_points: List[Dict]) -> Tuple[float, float]:
    """
    Calculates the max profit and max loss percentage over the future data points.
    """
    if not future_data_points:
        return 0.0, 0.0

    max_price = current_price
    min_price = current_price

    for point in future_data_points:
        price = point.get("optionPrice", current_price)
        if price > max_price:
            max_price = price
        if price < min_price:
            min_price = price
            
    # Avoid division by zero
    if current_price == 0:
        return 0.0, 0.0

    predicted_profit_percentage = ((max_price - current_price) / current_price) * 100.0
    predicted_loss_percentage = ((min_price - current_price) / current_price) * 100.0

    return predicted_profit_percentage, predicted_loss_percentage


def process_daily_data():
    """
    Main function to fetch, process, and save data to a parquet file.
    """
    client = MongoClient("mongodb://localhost:27017/")
    db = client["options_raw_data"]
    collection = db["option_intervals"]
    print("Successfully connected to MongoDB.")

    now_utc = datetime.now(timezone.utc)
    start_time = now_utc - timedelta(hours=HOURS_TO_PROCESS)

    unique_contracts = get_unique_contracts(collection, start_time)
    
    if not unique_contracts:
        print("No contracts found to process. Exiting.")
        return

    processed_data_rows = []

    for i, contract in enumerate(unique_contracts):
        print(f"\nProcessing contract {i+1}/{len(unique_contracts)}: {contract['underlyingSymbol']} {contract['strikePrice']} {contract['optionType']}")
        
        ticker_encoding = get_ticker_encoding(contract["underlyingSymbol"])
        if ticker_encoding == 0:
            print(f"  -> Skipping, ticker {contract['underlyingSymbol']} not in vocabulary.")
            continue

        # Fetch all available recent data for the contract
        time_series = fetch_contract_time_series(collection, contract, start_time, now_utc)

        # Check if we have enough data to form even one complete feature + label window
        if len(time_series) < TOTAL_WINDOW_SIZE:
            print(f"  -> Skipping, not enough data points ({len(time_series)}) to form a full {TOTAL_WINDOW_SIZE}-point window.")
            continue
            
        # Iterate through each possible "present" moment in the time series
        # The range ensures there's enough data for a full look-back and look-forward
        for i in range(FEATURE_WINDOW_POINTS - 1, len(time_series) - LABELING_WINDOW_POINTS):
            
            # --- Correctly Slice the Data ---
            
            # The INPUT features are the PAST data leading up to and including the present
            feature_window = time_series[i - (FEATURE_WINDOW_POINTS - 1) : i + 1]
            
            # The LABEL is determined by what happens in the FUTURE, after the present
            labeling_window = time_series[i + 1 : i + 1 + LABELING_WINDOW_POINTS]

            # The "current" data point is the last one in our feature window
            current_data_point = feature_window[-1]

            # --- Assemble the Training Example ---

            # Input features for the model
            time_series_data = [dp.get("normalizedData", []) for dp in feature_window]
            masking_array = [dp.get("maskWhenTraining", True) for dp in feature_window]
            
            # If absolutely none of the datapoints for this time series occurred during market hours, don't bother training with it 
            if all(masking_array):
                continue
            
            if not all(isinstance(d, list) and len(d) == 11 for d in time_series_data):
                print(f"  -> Skipping a window due to missing or invalid normalizedData.")
                continue

            current_price = current_data_point.get("optionPrice")
            if current_price is None:
                print("  -> Skipping a window due to missing current price.")
                continue

            # The label_data_point function is now correctly given only future data
            profit_perc, loss_perc = label_data_point(current_price, labeling_window)

            option_type_encoding = get_option_type_encoding(contract["optionType"])

            processed_data_rows.append({
                "ticker_embedding": ticker_encoding,
                "option_type_embedding": option_type_encoding,
                "time_series_data": time_series_data,
                "masking_array": masking_array,
                "predicted_profit_percentage": profit_perc,
                "predicted_loss_percentage": loss_perc
            })

    if not processed_data_rows:
        print("\nNo processable training data was generated.")
        client.close()
        return

    df = pd.DataFrame(processed_data_rows)
    
    file_date = now_utc.strftime('%Y-%m-%d')
    file_name = f"{file_date}_options_training_data.parquet"
    
    df.to_parquet(file_name, index=False)
    
    print(f"\nSuccessfully created {len(df)} training examples.")
    print(f"Data saved to {file_name}")

    client.close()


if __name__ == "__main__":
    process_daily_data()