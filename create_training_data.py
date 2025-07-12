import pandas as pd
import pymongo
from pymongo import MongoClient
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple

# --- Import the new embedding utility ---
from embedding_utils import get_ticker_encoding, get_option_type_encoding

# --- Configuration ---
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "options_raw_data"
COLLECTION_NAME = "option_intervals"
HOURS_TO_PROCESS = 24
LABELING_WINDOW_HOURS = 3.25
LABELING_WINDOW_POINTS = 39 # 3.25 hours * (60 minutes / 5 minutes)
TIME_SERIES_LENGTH = 40 # 39 future points for labeling + 1 current point

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
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    print("Successfully connected to MongoDB.")

    now_utc = datetime.now(timezone.utc)
    start_time = now_utc - timedelta(hours=HOURS_TO_PROCESS)
    processing_cutoff_time = now_utc - timedelta(hours=LABELING_WINDOW_HOURS)

    unique_contracts = get_unique_contracts(collection, start_time)
    
    if not unique_contracts:
        print("No contracts found to process. Exiting.")
        return

    processed_data_rows = []

    for i, contract in enumerate(unique_contracts):
        print(f"\nProcessing contract {i+1}/{len(unique_contracts)}: {contract['underlyingSymbol']} {contract['strikePrice']} {contract['optionType']}")
        
        # --- Use the new utility to check if the ticker is in our list ---
        ticker_encoding = get_ticker_encoding(contract["underlyingSymbol"])
        if ticker_encoding == 0: # Skip if the ticker is not in our vocabulary
            print(f"  -> Skipping, ticker {contract['underlyingSymbol']} not in vocabulary.")
            continue

        time_series = fetch_contract_time_series(collection, contract, start_time, processing_cutoff_time)

        if len(time_series) < TIME_SERIES_LENGTH:
            print(f"  -> Skipping, not enough data points ({len(time_series)})")
            continue
            
        for i in range(len(time_series) - TIME_SERIES_LENGTH + 1):
            
            training_window = time_series[i : i + TIME_SERIES_LENGTH]
            current_data_point = training_window[0]
            future_points_for_labeling = training_window[1:]

            time_series_data = [dp.get("normalizedData", []) for dp in training_window]
            masking_array = [dp.get("maskWhenTraining", True) for dp in training_window]

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

            profit_perc, loss_perc = label_data_point(current_price, future_points_for_labeling)

            # --- Use the utility to get integer encodings ---
            option_type_encoding = get_option_type_encoding(contract["optionType"])

            processed_data_rows.append({
                "ticker_embedding": ticker_encoding, # Now an integer
                "option_type_embedding": option_type_encoding, # Now an integer
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