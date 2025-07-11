from pymongo import MongoClient, UpdateOne
from datetime import datetime
import math

import pymongo

class RunningStatsDB:
    def __init__(self, uri="mongodb://localhost:27017/", db_name="normalization_stats", collection_name="normalization_data"):
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

    def update_running_stats(self, data_items):
        """
        Updates the running mean and standard deviation for specified features using an incremental algorithm.

        Args:
            data_items (list[dict]): A list of raw option data items.
        """
        updates = []
        features_to_track = [
            "optionPriceRatioToUnderlying",
            "optionPriceRatioToIntrinsicValue",
            "optionPriceRatioToExtrinsicValue",
            "strikePriceRatioToUnderlying",
            "strikePriceRatioToOptionPrice",
            "volatility",
            "theta",
            "vega",
            "gamma",
            "log10Volume"
        ]

        for item in data_items:
            symbol = item.get("underlyingSymbol")
            option_type = item.get("optionType")

            if not symbol or not option_type:
                continue

            underlying_price = item.get("underlyingPrice", 1e-6)
            strike_price = item.get("strikePrice", 1e-6)
            option_price = item.get("optionPrice", 1e-6)
            intrinsic_value = item.get("intrinsicValue", 1e-6)
            extrinsic_value = item.get("extrinsicValue", 1e-6)
            volume = item.get("volume", 0)

            # Calculate ratios
            option_price_ratio_to_underlying = option_price / underlying_price if underlying_price != 0 else 0
            option_price_ratio_to_intrinsic_value = option_price / (intrinsic_value * 100) if intrinsic_value != 0 else 0
            option_price_ratio_to_extrinsic_value = option_price / (extrinsic_value * 100) if extrinsic_value != 0 else 0
            strike_price_ratio_to_underlying = strike_price / underlying_price if underlying_price != 0 else 0
            strike_price_ratio_to_option_price = strike_price / option_price if option_price != 0 else 0

            # Calculate log10(volume)
            log10_volume = math.log10(volume + 1e-6) if volume >= 0 else 0

            # Prepare the data item with the features to track
            processed_item = {
                "underlyingSymbol": symbol,
                "optionType": option_type,
                "optionPriceRatioToUnderlying": option_price_ratio_to_underlying,
                "optionPriceRatioToIntrinsicValue": option_price_ratio_to_intrinsic_value,
                "optionPriceRatioToExtrinsicValue": option_price_ratio_to_extrinsic_value,
                "strikePriceRatioToUnderlying": strike_price_ratio_to_underlying,
                "strikePriceRatioToOptionPrice": strike_price_ratio_to_option_price,
                "volatility": item.get("impliedVolatility"),
                "theta": item.get("theta"),
                "vega": item.get("vega"),
                "gamma": item.get("gamma"),
                "log10Volume": log10_volume
            }

            for feature in features_to_track:
                value = processed_item.get(feature)
                if value is not None and isinstance(value, (int, float)):
                    filter_criteria = {
                        "underlyingSymbol": symbol,
                        "optionType": option_type,
                        "feature": feature
                    }

                    update_operation = {
                        "$inc": {
                            "count": 1,
                            "sum": value,
                            "sum_sq": value ** 2
                        }
                    }
                    updates.append(UpdateOne(filter_criteria, update_operation, upsert=True))

        if updates:
            print(f"Making {len(updates)} updates to the normalization db.")
            self.collection.bulk_write(updates)

    def get_running_stats(self, symbol: str, option_type: str) -> dict[str, dict]:
        """
        Retrieves the running mean and standard deviation for all tracked features
        for a specific stock ticker and type.

        Args:
            symbol (str): Underlying stock symbol.
            option_type (str): Type of option ("CALL" or "PUT").

        Returns:
            dict[str, dict]: A dictionary where keys are feature names and
                             values are dictionaries containing 'mean' and 'std_dev'.
        """
        results = self.collection.find(
            {"underlyingSymbol": symbol, "optionType": option_type}
        )
        stats_map = {}
        for result in results:
            feature = result.get("feature")
            count = result.get("count", 0)
            if feature and count > 0:
                mean = result["sum"] / count
                variance = (result["sum_sq"] / count) - (mean ** 2)
                std_dev = math.sqrt(variance) if variance > 0 else 0
                stats_map[feature] = {"mean": mean, "std_dev": std_dev}
        return stats_map

    def clear_running_stats(self):
        """
        Deletes all data from the running_stats collection.
        """
        if self.collection is not None:
            self.collection.delete_many({})
            print(f"All data deleted from the {self.collection.name} collection.")
        else:
            print("Collection not initialized. Cannot clear data.")