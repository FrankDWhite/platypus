from typing import Dict, List
import pymongo
from pymongo import MongoClient, ASCENDING, DESCENDING
from datetime import datetime, timedelta, timezone
import random
import uuid

import pytz
import numpy as np

from models import OptionChainResponse, OptionData # For generating unique _id if needed, though ObjectId is default

class OptionsRawDataStore:
    """
    A Python utility for interacting with a local MongoDB instance
    to manage raw options data for an AI project.

    This utility assumes a running MongoDB instance at mongodb://localhost:27017/.
    For sharding, it assumes connection to a mongos router.
    """

    def __init__(self, uri="mongodb://localhost:27017/", db_name="options_raw_data", collection_name="option_intervals"):
        """
        Initializes the MongoDB client and sets up database and collection.

        Args:
            uri (str): MongoDB connection URI.
            db_name (str): The name of the database to use.
            collection_name (str): The name of the collection to store data in.
        """
        self.uri = uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = None
        self.db = None
        self.collection = None
        self._connect()

    def _connect(self):
        """Establishes connection to MongoDB."""
        try:
            self.client = MongoClient(self.uri)
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            print(f"Successfully connected to MongoDB: {self.uri}")
            print(f"Using database: {self.db_name}, collection: {self.collection_name}")
        except pymongo.errors.ConnectionFailure as e:
            print(f"Could not connect to MongoDB at {self.uri}: {e}")
            self.client = None # Ensure client is None on failure

    def close_connection(self):
        """Closes the MongoDB connection."""
        if self.client:
            self.client.close()
            print("MongoDB connection closed.")

    def create_compound_index(self):
        """
        Creates the primary compound index for efficient querying and sharding readiness.
        Index: { underlyingSymbol: 1, expirationDate: 1, strikePrice: 1, optionType: 1, intervalTimestamp: 1 }
        """
        if self.client is None or self.collection is None:
            print("Collection not initialized. Cannot create index.")
            return

        index_name = "options_contract_timeseries_index"
        existing_indexes = self.collection.index_information()

        if index_name in existing_indexes:
            print(f"Index '{index_name}' already exists.")
            return

        try:
            self.collection.create_index(
                [
                    ("underlyingSymbol", ASCENDING),
                    ("expirationDate", ASCENDING),
                    ("strikePrice", ASCENDING),
                    ("optionType", ASCENDING),
                    ("intervalTimestamp", ASCENDING)
                ],
                name=index_name,
                background=True # Create index in background to not block operations
            )
            print(f"Compound index '{index_name}' created successfully.")
        except pymongo.errors.PyMongoError as e:
            print(f"Error creating index: {e}")

    # --- Write Operations ---

    def insert_many_option_data(self, data_list: list[dict]):
        """
        Inserts multiple documents into the collection.
        Args:
            data_list (list[dict]): A list of documents to insert.
        Returns:
            list[str]: A list of _ids of the inserted documents, or empty list on failure.
        """
        if self.client is None or self.collection is None:
            print("Collection not initialized. Cannot insert many.")
            return []

        try:
            result = self.collection.insert_many(data_list, ordered=False) # ordered=False for better performance
            print(f"Inserted {len(result.inserted_ids)} documents.")
            return [str(uid) for uid in result.inserted_ids]
        except pymongo.errors.BulkWriteError as e:
            print(f"Error inserting many documents (some might have failed): {e.details}")
            # You can inspect e.details['writeErrors'] for specific errors
            return [str(uid) for uid in e.details.get('insertedIds', [])]
        except pymongo.errors.PyMongoError as e:
            print(f"General error inserting many documents: {e}")
            return []

    def update_normalized_data(self, doc_id: str, normalized_array: np.ndarray):
        """
        Updates a specific option contract with normalized data.

        Args:
            doc_id (str): The _id of the document to update.
            normalized_array (np.ndarray): The NDArray containing the normalized data.
        """
        if self.collection is None:
            print("Collection not initialized. Cannot update document.")
            return

        try:
            from bson.objectid import ObjectId
            object_id = ObjectId(doc_id)
            normalized_list = normalized_array.tolist()
            result = self.collection.update_one(
                {"_id": object_id},
                {"$set": {"normalizedData": normalized_list}}
            )
            # if result.modified_count > 0:
            #     print(f"Successfully updated document with _id: {doc_id} with normalized data.")
            # else:
            #     print(f"Could not find document with _id: {doc_id} to update.")
        except pymongo.errors.PyMongoError as e:
            print(f"Error updating document with normalized data: {e}")
        except Exception as e:
            print(f"Invalid doc_id format: {e}")


    def find_by_ids(self, ids: list[str]) -> list[dict]:
        """
        Queries the collection for entries with the specified IDs.

        Args:
            ids (list[str]): A list of string representations of the _ids to query.

        Returns:
            list[dict]: A list of the matching documents.
        """
        if not ids:
            return []

        try:
            from bson.objectid import ObjectId
            object_ids = [ObjectId(id_str) for id_str in ids if ObjectId.is_valid(id_str)]
            if not object_ids:
                return []

            query = {"_id": {"$in": object_ids}}
            results = list(self.collection.find(query))
            print(f"Found {len(results)} documents with the specified IDs.")
            return results
        except pymongo.errors.PyMongoError as e:
            print(f"Error querying documents by IDs: {e}")
            return []

    # --- Read Operations ---
    def find_option_contract_time_series(
        self,
        symbol: str,
        expiration_date: datetime,
        strike_price: float,
        option_type: str,
        start_timestamp: datetime = None,
        end_timestamp: datetime = None,
        limit: int = 0,
        sort_order: int = ASCENDING # 1 for ascending, -1 for descending
    ) -> list[dict]:
        """
        Retrieves the full historical time series for a specific option contract.
        Uses the composite 'partition key' and 'sort key' for efficient querying.

        Args:
            symbol (str): Underlying stock symbol (e.g., "AAPL").
            expiration_date (datetime): Expiration date of the option (UTC).
            strike_price (float): Strike price of the option.
            option_type (str): Type of option ("CALL" or "PUT").
            start_timestamp (datetime, optional): Start of the time range (inclusive).
            end_timestamp (datetime, optional): End of the time range (inclusive).
            limit (int, optional): Maximum number of documents to return. 0 means no limit.
            sort_order (int, optional): Sort order for intervalTimestamp (1 for ASC, -1 for DESC).

        Returns:
            list[dict]: A list of matching option data documents.
        """
        if self.client is None or self.collection is None:
            print("Collection not initialized. Cannot query.")
            return []

        query = {
            "underlyingSymbol": symbol,
            "expirationDate": expiration_date,
            "strikePrice": strike_price,
            "optionType": option_type,
        }

        # Add timestamp range filter if provided
        if start_timestamp or end_timestamp:
            query["intervalTimestamp"] = {}
            if start_timestamp:
                query["intervalTimestamp"]["$gte"] = start_timestamp
            if end_timestamp:
                query["intervalTimestamp"]["$lte"] = end_timestamp

        try:
            cursor = self.collection.find(query).sort("intervalTimestamp", sort_order)
            if limit > 0:
                cursor = cursor.limit(limit)

            results = list(cursor)
            print(f"Found {len(results)} documents for contract {symbol}/{strike_price}/{option_type}/{expiration_date}.")
            return results
        except pymongo.errors.PyMongoError as e:
            print(f"Error querying documents: {e}")
            return []
        


        # --- New Function: Ingest OptionChainResponse ---
    def insert_option_chain_response(self, option_chain_response: OptionChainResponse, market_open):
        """
        Parses an OptionChainResponse object, extracts relevant option contract data,
        and inserts it into the MongoDB collection.

        Args:
            option_chain_response (OptionChainResponse): The Pydantic object
                                                         containing options chain data.
        Returns:
            list[str]: A list of _ids of the inserted documents.
        """
        if self.collection is None:
            print("Collection not initialized. Cannot insert option chain response.")
            return []

        documents_to_insert = []
        underlying_symbol = option_chain_response.symbol
        current_underlying_price = option_chain_response.underlyingPrice
        current_underlying_volume = option_chain_response.underlying.totalVolume

        # Determine the current time
        now = datetime.now(timezone.utc)

        # Calculate the nearest 5-minute interval
        remainder = now.minute % 5
        if remainder >= 3:
            round_to = 5 - remainder
            interval_timestamp = now.replace(minute=(now.minute + round_to) % 60, second=0, microsecond=0)
            if (now.minute + round_to) >= 60:
                interval_timestamp += timedelta(hours=1)
        else:
            round_to = -remainder
            interval_timestamp = now.replace(minute=now.minute + round_to, second=0, microsecond=0)

        def calculate_hours_to_expiry(utc_date_str):
            """
            Calculates the number of hours from now until 3:00 PM Central Time on a specified UTC date.

            Args:
                utc_date_str (str): A UTC date string in 'YYYY-MM-DD' format (e.g., "2025-07-11").

            Returns:
                float: The number of hours until 3:00 PM CT on the specified date, or None if an error occurs.
            """
            try:
                # 1. Define the Central Time zone
                central_timezone = pytz.timezone('America/Chicago')

                # 2. Parse the input UTC date string
                # We parse it as a naive datetime first, then combine with time
                target_date_naive = datetime.strptime(utc_date_str, '%Y-%m-%d')

                # 3. Create a datetime object for 3:00 PM CT on the target date
                # We need to localize it to the Central Timezone
                target_time_ct = central_timezone.localize(
                    datetime(target_date_naive.year, target_date_naive.month, target_date_naive.day, 15, 0, 0)
                )

                # 4. Get the current time in UTC
                now_utc = datetime.now(pytz.utc)

                # 5. Calculate the difference
                time_difference = target_time_ct - now_utc

                # 6. Convert the difference to hours
                hours_difference = round(time_difference.total_seconds() / 3600)

                return hours_difference

            except ValueError as e:
                print(f"Error parsing date or time: {e}")
                return None
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                return None

        def process_option_map(option_map: Dict[str, Dict[str, List[OptionData]]]):
            for exp_date_str, strike_map in option_map.items():
                # The expirationDate string in Schwab API is like "YYYY-MM-DD:days"
                # We need to parse the YYYY-MM-DD part for our datetime object
                try:
                    # Split 'YYYY-MM-DD:days' to get 'YYYY-MM-DD'
                    exp_date_only_str = exp_date_str.split(':')[0]
                    # Convert to datetime object (ensure UTC for consistency)
                    expiration_date_dt = datetime.strptime(exp_date_only_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    hours_to_expiration = calculate_hours_to_expiry(exp_date_only_str)
                except ValueError:
                    print(f"Warning: Could not parse expirationDate string '{exp_date_str}'. Skipping.")
                    continue

                for strike_price_str, option_data_list in strike_map.items():
                    for option_data in option_data_list:
                        # Extract only the attributes we defined in our MongoDB schema
                        # and common AI features.


                        document = {
                            "overallDescription": option_data.description,
                            "underlyingSymbol": underlying_symbol,
                            "strikePrice": option_data.strikePrice,
                            "optionType": option_data.putCall, # 'CALL' or 'PUT' from the model
                            "expirationDate": expiration_date_dt,
                            "optionPrice": option_data.mark * 100,
                            "volume": option_data.totalVolume,
                            "gamma": option_data.gamma,
                            "theta": option_data.theta,
                            "vega": option_data.vega,
                            "hoursToExpiration": hours_to_expiration,
                            "impliedVolatility": option_data.volatility, # Assuming 'volatility' field is implied volatility
                            "intrinsicValue": option_data.intrinsicValue,
                            "extrinsicValue": option_data.extrinsicValue,
                            "inTheMoney": option_data.inTheMoney,

                            "underlyingPrice": current_underlying_price, # Add underlying price
                            "underlyingVolume": current_underlying_volume,

                            "maskWhenTraining": not market_open,
                            "normalizedData" : { },
                            "intervalTimestamp": interval_timestamp, # Our generated timestamp
                        }
                        documents_to_insert.append(document)

        # Process Call Options
        process_option_map(option_chain_response.callExpDateMap)
        # Process Put Options
        process_option_map(option_chain_response.putExpDateMap)

        if not documents_to_insert:
            print(f"No documents extracted from OptionChainResponse for {underlying_symbol}.")
            return []

        # Use insert_many to efficiently write all collected documents
        print(f"Attempting to insert {len(documents_to_insert)} documents for {underlying_symbol} at {interval_timestamp.isoformat()}.")
        inserted_ids = self.insert_many_option_data(documents_to_insert)
        return inserted_ids
    
        # --- New Function: Scan All Documents ---
    def scan_all_documents(self, limit: int = 0, sort_key: str = None, sort_order: int = ASCENDING) -> list[dict]:
        """
        Scans all documents in the collection.
        WARNING: This can be very inefficient on large collections and should primarily
        be used for testing or small datasets.

        Args:
            limit (int, optional): Maximum number of documents to return. 0 means no limit.
            sort_key (str, optional): Field name to sort by. If None, no explicit sort.
            sort_order (int, optional): Sort order for the sort_key (1 for ASC, -1 for DESC).

        Returns:
            list[dict]: A list of all matching documents.
        """
        if self.client is None or self.collection is None:
            print("Collection not initialized. Cannot scan documents.")
            return []

        try:
            cursor = self.collection.find({})
            if sort_key:
                cursor = cursor.sort(sort_key, sort_order)
            if limit > 0:
                cursor = cursor.limit(limit)

            results = list(cursor)
            print(f"Scanned {len(results)} documents.")
            return results
        except pymongo.errors.PyMongoError as e:
            print(f"Error scanning documents: {e}")
            return []
        
        # --- New Function: Delete Old Documents ---
    def delete_old_documents(self, cutoff_timestamp: datetime) -> int:
        """
        Deletes all documents from the collection that have an 'intervalTimestamp'
        older than the specified cutoff_timestamp.

        Args:
            cutoff_timestamp (datetime): Documents with intervalTimestamp
                                         LESS THAN this datetime will be deleted.
                                         Ensure this is a timezone-aware UTC datetime.

        Returns:
            int: The number of documents deleted.
        """
        if self.collection is None:
            print("Collection not initialized. Cannot delete documents.")
            return 0

        # Ensure the cutoff_timestamp is timezone-aware and in UTC
        if cutoff_timestamp.tzinfo is None:
            print("Warning: cutoff_timestamp is naive. Assuming UTC.")
            cutoff_timestamp = cutoff_timestamp.replace(tzinfo=timezone.utc)
        elif cutoff_timestamp.tzinfo != timezone.utc:
            print(f"Warning: cutoff_timestamp is not UTC ({cutoff_timestamp.tzinfo}). Converting to UTC.")
            cutoff_timestamp = cutoff_timestamp.astimezone(timezone.utc)


        query = {
            "intervalTimestamp": {"$lt": cutoff_timestamp}
        }

        try:
            print(f"Attempting to delete documents older than: {cutoff_timestamp.isoformat()}")
            result = self.collection.delete_many(query)
            print(f"Deleted {result.deleted_count} documents.")
            return result.deleted_count
        except pymongo.errors.PyMongoError as e:
            print(f"Error deleting documents: {e}")
            return 0
