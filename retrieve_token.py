from pymongo import ASCENDING, DESCENDING, MongoClient
from datetime import date, datetime, timedelta, timezone
import requests
import pandas
import pprint

from models import OptionChainResponse
from mongodb_utilities import OptionsRawDataStore
from normalization_stats import RunningStatsDB
from normalization_utilities import normalize_option_data
import numpy as np

def read_access_token():
    # MongoDB connection details
    mongo_uri = "mongodb://192.168.1.231:27017/"
    database_name = "tokens_db" # Use the database name you chose
    collection_name = "tokens_collection" # Use the collection name you chose
    token_name = "schwab_api_token"

    try:
        # Create a MongoClient to the running mongod instance
        client = MongoClient(mongo_uri)

        # Try to connect to the server (optional, but good for error checking)
        client.admin.command('ping')
        print("Successfully connected to MongoDB server.")

        # Get a reference to the database
        db = client[database_name]

        # Get a reference to the collection
        tokens_collection = db[collection_name]

        # Query for the document with the Schwab API token
        token_document = tokens_collection.find_one({"name": token_name})

        if token_document:
            retrieved_token = token_document.get("access_token")
            print(f"Retrieved Schwab API token from remote DB: {retrieved_token}")
            return retrieved_token
        else:
            print(f"No document found with name: {token_name} in the remote DB.")
            return None

    except Exception as e:
        print(f"An error occurred while connecting or reading from the remote DB: {e}")
        return None

    finally:
        # Close the MongoDB client
        client.close()

class AccountsTrading:
    def __init__(self):
        # Initialize access token by reading from MongoDB
        self.access_token = read_access_token()
        self.account_hash_value = None
        self.base_url = "https://api.schwabapi.com/trader/v1"
        self.headers = {"Authorization": f"Bearer {self.access_token}"}
        if self.access_token:
            self.get_account_number_hash_value()
        else:
            print("Failed to retrieve access token during initialization.")

    # We no longer need this method if the token refresh is handled elsewhere
    # def refresh_access_token(self):
    #     pass

    def get_account_number_hash_value(self):
        if self.access_token:
            response = requests.get(
                self.base_url + f"/accounts/accountNumbers", headers=self.headers
            )
            response.raise_for_status()
            response_frame = pandas.json_normalize(response.json())
            self.account_hash_value = response_frame["hashValue"].iloc[0]
        else:
            print("Access token is not available.")

    def get_quote(self, symbol):
        if self.access_token:
            quote_url = f"https://api.schwabapi.com/marketdata/v1/{symbol}/quotes"
            try:
                response = requests.get(quote_url, headers=self.headers)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"Error fetching quote for {symbol}: {e}")
                if response is not None:
                    print(f"Status code: {response.status_code}, Response: {response.text}")
                return None
        else:
            print("Access token is not available.")
            return None
    def get_quotes(self, symbols):
        if self.access_token:
            params = {
                "symbols": symbols
            }
            quote_url = f"https://api.schwabapi.com/marketdata/v1/quotes"
            try:
                response = requests.get(quote_url, headers=self.headers, params=params)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"Error fetching quote for {symbols}: {e}")
                if response is not None:
                    print(f"Status code: {response.status_code}, Response: {response.text}")
                return None
        else:
            print("Access token is not available.")
            return None
    def get_options(self, symbol):
        if self.access_token:
            today = date.today()
            from_date = today.strftime("%Y-%m-%d")
            seven_days_from_now = today + timedelta(days=7)
            to_date = seven_days_from_now.strftime("%Y-%m-%d")
            params = {
                "symbol": symbol,
                "contractType": "ALL",
                "strikeCount": 10,
                "fromDate": from_date,
                "toDate": to_date,
                "includeUnderlyingQuote": "true"

            }
            quote_url = f"https://api.schwabapi.com/marketdata/v1/chains"
            try:
                response = requests.get(quote_url, headers=self.headers, params=params)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"Error fetching quote for {symbol}: {e}")
                if response is not None:
                    print(f"Status code: {response.status_code}, Response: {response.text}")
                return None
        else:
            print("Access token is not available.")
            return None

    def get_market_hours(self):
        if self.access_token:
            params = {
                "markets": ['option']

            }
            quote_url = f"https://api.schwabapi.com/marketdata/v1/markets"
            try:
                response = requests.get(quote_url, headers=self.headers, params=params)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"Error fetching market hours")
                if response is not None:
                    print(f"Status code: {response.status_code}, Response: {response.text}")
                return None
        else:
            print("Access token not available.")
            return None

    def is_market_open(self, api_response):
        """
        Determines if the market is open based on an API response.
        Prioritizes EQO market hours check. If EQO hours are not found or
        parseable, it falls back to the general 'isOpen' flag from the response.

        Args:
            api_response (dict): The dictionary returned by the API.

        Returns:
            bool: True if EQO exists and current time is between its timestamps,
                OR if EQO does not exist/cannot be parsed and the general
                'isOpen' flag is True. Returns False otherwise.
        """
        current_time_utc = datetime.now(timezone.utc)

        # 1. Try to parse EQO market hours first
        if 'option' in api_response:
            if 'EQO' in api_response['option'] and isinstance(api_response['option']['EQO'], dict):
                eqo_data = api_response['option']['EQO']

                if 'sessionHours' in eqo_data and \
                'regularMarket' in eqo_data['sessionHours'] and \
                isinstance(eqo_data['sessionHours']['regularMarket'], list) and \
                len(eqo_data['sessionHours']['regularMarket']) > 0:

                    market_session = eqo_data['sessionHours']['regularMarket'][0]

                    if 'start' in market_session and 'end' in market_session:
                        try:
                            start_time = datetime.fromisoformat(market_session['start'])
                            end_time = datetime.fromisoformat(market_session['end'])

                            # Convert to UTC for consistent comparison
                            start_time_utc = start_time.astimezone(timezone.utc)
                            end_time_utc = end_time.astimezone(timezone.utc)

                            # Check if current time is within EQO market hours
                            return start_time_utc <= current_time_utc < end_time_utc

                        except ValueError as e:
                            print(f"Warning: Error parsing EQO timestamps, returning false")
                            return False

        # 2. If EQO data wasn't found or parsing failed, fall back to the general 'isOpen' flag
        # This specifically targets weekend queries
        if 'option' in api_response and \
        'option' in api_response['option'] and \
        isinstance(api_response['option']['option'], dict) and \
        'isOpen' in api_response['option']['option']:
            return api_response['option']['option']['isOpen']

        # 3. If neither valid EQO hours nor the specific 'isOpen' flag is found, default to False
        return False

if __name__ == "__main__":
    trading_client = AccountsTrading()
    if trading_client.account_hash_value:
        print(f"Account Hash Value: {trading_client.account_hash_value}")
        # Now you can use other methods in the AccountsTrading class
        # to make further API calls using trading_client
        stock_symbol = "AMZN"  # Replace with the symbol you want
        quote_data = trading_client.get_options(stock_symbol)
        if quote_data:
            # print(f"Quote for {stock_symbol}:")
            # pprint.pprint(quote_data)
            option_chain = OptionChainResponse.model_validate(quote_data)
            print(f"Successfully parsed data for symbol: {option_chain.symbol}")
            print(f"Underlying asset description: {option_chain.underlying.description}")
            # print(f"Expiration Date: {option_chain.callExpDateMap['2025-07-03:0']['222.5'][0].expirationDate}")
                # 1. Initialize the data store utility
            options_db = OptionsRawDataStore()
            normalization_db = RunningStatsDB()

            # Check connection
            if not options_db.client:
                print("Exiting due to MongoDB connection failure.")
                exit()

            market_hours = trading_client.get_market_hours()
            if not market_hours:
                print("Exiting due to failure fetching market hours.")
                exit()
            print(f"market_hours {market_hours}")
            
            market_open = trading_client.is_market_open(market_hours)

            print(f"is market open? {market_open}")

            # 2. Create the compound index (idempotent operation)
            options_db.create_compound_index()
            inserted_ids_from_chain = options_db.insert_option_chain_response(option_chain, market_open)
            print(f"Successfully inserted {len(inserted_ids_from_chain)} documents from OptionChainResponse.")

            inserted_documents = options_db.find_by_ids(inserted_ids_from_chain)
            
            if market_open:
                normalization_db.update_running_stats(inserted_documents)

            for doc in inserted_documents:
                if (doc["optionType"] == "CALL"):
                    normalized_values = normalize_option_data(doc, normalization_db.get_running_stats(stock_symbol, "CALL"))
                else:
                    normalized_values = normalize_option_data(doc, normalization_db.get_running_stats(stock_symbol, "PUT"))
                options_db.update_normalized_data(doc["_id"], normalized_values)

            # all_documents_scanned = options_db.scan_all_documents(limit=20, sort_key="intervalTimestamp", sort_order=DESCENDING)
            # for i, doc in enumerate(all_documents_scanned):
            #     print(f"\n  --- Document {i+1} ---")
            #     for key, value in doc.items():
            #         # Special handling for datetime objects to make them readable
            #         if isinstance(value, datetime):
            #             print(f"    {key}: {value.isoformat()}")
            #         elif isinstance(value, dict):
            #             print(f"    {key}:")
            #             for sub_key, sub_value in value.items():
            #                 print(f"      {sub_key}: {sub_value}")
            #         else:
            #             print(f"    {key}: {value}")
                      
            # delete_older_than = datetime(2025, 7, 11, 0, 37, 10, tzinfo=timezone.utc)

            # deleted_count = options_db.delete_old_documents(delete_older_than)
            options_db.close_connection()

            # print(f"Function reported {deleted_count} documents deleted.")
            # normalization_db.clear_running_stats()
            normalization_db.close_connection()
            
        else:
            print(f"Failed to get quote for {stock_symbol}.")
    else:
        print("Failed to retrieve account hash value.")