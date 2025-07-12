#!/usr/bin/ python3.10

from datetime import datetime, timedelta, timezone
from mongodb_utilities import OptionsRawDataStore

# --- Configuration ---
# The number of days of data to keep. Everything older than this will be deleted.
DAYS_TO_KEEP = 7

def main():
    """
    Connects to the options database and deletes all documents older than
    the specified number of days.
    """
    print("--- Starting Old Data Cleanup ---")

    # 1. Initialize the data store utility
    options_db = OptionsRawDataStore()

    # Check for a successful connection
    if not options_db.client:
        print("Exiting due to MongoDB connection failure.")
        return

    # 2. Calculate the cutoff timestamp
    # We want to delete any document with an 'intervalTimestamp' older than this.
    cutoff_timestamp = datetime.now(timezone.utc) - timedelta(days=DAYS_TO_KEEP)
    
    print(f"Purging all documents from the options database older than: {cutoff_timestamp.isoformat()}")

    # 3. Call the existing delete function
    # This function is already part of your mongodb_utilities.py file
    # and does the actual work of deleting the documents.
    deleted_count = options_db.delete_old_documents(cutoff_timestamp)

    # 4. Report the result and close the connection
    print(f"Cleanup complete. Function reported {deleted_count} documents deleted.")
    options_db.close_connection()
    print("--- Cleanup Finished ---")


if __name__ == "__main__":
    main()