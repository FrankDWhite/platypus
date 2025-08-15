#!/usr/bin/python3.10

from pathlib import Path

# --- Import the main workflow function from our trading machine ---
from trading_machine import trigger_daily_retrain

# --- Configuration ---
# This path must match the CURRENT_DAY_SAVE_DIRECTORY in create_training_data.py
# It points to the directory containing only the most recent day's training data file.
CURRENT_DAY_DATA_DIRECTORY = Path("./training_data/current_day")

def main():
    """
    This script serves as the single entry point to kick off the entire automated
    daily fine-tuning and server restart workflow for the Platypus project.
    """
    print("---  initiating the full, automated MCP training workflow. ---")
    
    # Check if the directory with the new training data exists.
    if not CURRENT_DAY_DATA_DIRECTORY.exists():
        print(f"Error: Training data directory not found at '{CURRENT_DAY_DATA_DIRECTORY}'.")
        print("Please run create_training_data.py first to generate today's data.")
        return
        
    # Call the main orchestration function.
    # We convert the Path object to a string, as required by the function.
    trigger_daily_retrain(str(CURRENT_DAY_DATA_DIRECTORY))
    
    print("--- MCP workflow trigger has been sent. Check the MCP server logs for progress. ---")

if __name__ == "__main__":
    main()