#!/usr/bin/python3.10
import pandas as pd
import pymongo
from pymongo import MongoClient
from datetime import datetime, time, timedelta, timezone
from typing import List, Dict, Tuple
import pytz

# --- Import the new embedding utility ---
from embedding_utils import get_ticker_encoding, get_option_type_encoding
from mongodb_utilities import OptionsRawDataStore
from pathlib import Path

# --- Configuration ---
# We need to look back further to build our feature + label windows.
# Increased to 96 hours (4 days) to ensure we can wrap labeling over a weekend.
HOURS_TO_PROCESS = 96

# Window for input features (3.25 hours of past data + current point)
FEATURE_WINDOW_POINTS = 40

# Window for calculating the label (3.25 hours of future data)
LABELING_WINDOW_POINTS = 39

SAVE_DIRECTORY = Path("/home/bcm/training_data")


def is_within_market_hours(dt_object: datetime) -> bool:
    """
    Checks if a datetime object is within regular US stock market hours.
    (Monday-Friday, 8:30 AM to 3:00 PM Central Time)
    This function now correctly handles naive datetimes from MongoDB by assuming they are in UTC.
    Args:
        dt_object (datetime): A datetime object, which can be naive (assumed UTC) or timezone-aware.
    Returns:
        bool: True if within market hours, False otherwise.
    """
    if not dt_object:
        return False

    # If the datetime from MongoDB is naive (tzinfo is None),
    # we explicitly tell pytz that it represents UTC time.
    if dt_object.tzinfo is None:
        dt_object = pytz.utc.localize(dt_object)

    central_tz = pytz.timezone('America/Chicago')
    dt_central = dt_object.astimezone(central_tz)

    if dt_central.weekday() > 4:  # Monday is 0, Sunday is 6
        return False  # It's a weekend

    market_open_time = time(8, 30)
    market_close_time = time(15, 0)

    return market_open_time <= dt_central.time() < market_close_time


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
            
    if current_price == 0:
        return 0.0, 0.0

    predicted_profit_percentage = ((max_price - current_price) / current_price) * 100.0
    predicted_loss_percentage = ((min_price - current_price) / current_price) * 100.0

    return predicted_profit_percentage, predicted_loss_percentage


def process_daily_data():
    """
    Main function to fetch, process, and save data to a parquet file.
    """
    options_db = OptionsRawDataStore()
    if not options_db.client:
        print("Exiting due to MongoDB connection failure.")
        return

    collection = options_db.collection
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

        time_series = fetch_contract_time_series(collection, contract, start_time, now_utc)

        if len(time_series) < FEATURE_WINDOW_POINTS:
            print(f"  -> Skipping, not enough data points ({len(time_series)}) to form a feature window.")
            continue
        
        raw_options_data_entries_labeled = []
            
        for i in range(FEATURE_WINDOW_POINTS - 1, len(time_series)):
            
            feature_window = time_series[i - (FEATURE_WINDOW_POINTS - 1) : i + 1]
            current_data_point = feature_window[-1]

            # --- Ensure the point we are labeling is itself from within market hours ---
            current_timestamp = current_data_point.get("intervalTimestamp")
            description = current_data_point.get("overallDescription")
            if not current_timestamp or not is_within_market_hours(current_timestamp):
                continue

            # 1. Get all potential future points
            potential_future_points = time_series[i + 1:]
            
            # 2. Filter them to include only points that occurred during market hours
            valid_future_market_points = [
                p for p in potential_future_points 
                if p.get("intervalTimestamp") and is_within_market_hours(p["intervalTimestamp"])
            ]
            
            # 3. Check if we have enough valid points to form a label
            if len(valid_future_market_points) >= LABELING_WINDOW_POINTS:
                # 4. If so, the labeling window is the next 39 valid points
                labeling_window = valid_future_market_points[:LABELING_WINDOW_POINTS]
            else:
                # Not enough future market data to create a label.
                continue

            # --- Assemble the Training Example ---
            time_series_data = [dp.get("normalizedData", []) for dp in feature_window]
            masking_array = [dp.get("maskWhenTraining", True) for dp in feature_window]
            
            # If absolutely none of the datapoints for this time series occurred during market hours, don't bother training with it 
            if all(masking_array):
                continue

            # If we have already trained the model using this datapoint, don't bother training with it again
            if current_data_point.get("hasBeenLabeled"):
                print(f"  -> Skipping a data point that has already been previously labeled.")
                continue
            
            if not all(isinstance(d, list) and len(d) == 11 for d in time_series_data):
                print(f"  -> Skipping a window due to missing or invalid normalizedData.")
                continue

            current_price = current_data_point.get("optionPrice")
            if current_price is None:
                print("  -> Skipping a window due to missing current price.")
                continue

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
            raw_options_data_entries_labeled.append(current_data_point['_id'])

        if raw_options_data_entries_labeled:
            options_db.flag_document_as_labeled(raw_options_data_entries_labeled)
    if not processed_data_rows:
        print("\nNo processable training data was generated.")
        options_db.close_connection()
        return

    df = pd.DataFrame(processed_data_rows)
    SAVE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    file_date = now_utc.strftime('%Y-%m-%d')
    file_name = f"{file_date}_options_training_data.parquet"
    full_file_path = SAVE_DIRECTORY / file_name
    df.to_parquet(full_file_path, index=False)

    print(f"\nSuccessfully created {len(df)} training examples.")
    print(f"Data saved to {full_file_path}")

    options_db.close_connection()


if __name__ == "__main__":
    process_daily_data()