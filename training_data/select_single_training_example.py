#!/usr/bin/python3.10

import polars as pl
from pathlib import Path

# --- Configuration ---
# Point this to one of your large, existing training data files.
INPUT_PARQUET_FILE = Path("./2025-08-01_options_training_data.parquet")

# Define where to save the new single-example file.
# It's a good practice to put it in a separate directory for testing.
OUTPUT_DIRECTORY = Path("./training_data_single_example/")
OUTPUT_PARQUET_FILE = OUTPUT_DIRECTORY / "single_high_quality_example.parquet"

def create_single_example_file():
    """
    Reads a large Parquet file, extracts the first "highest-quality" data row,
    and saves it to a new file for testing purposes.
    
    A "highest-quality" row is one where the `masking_array` contains no True values,
    meaning every data point in its feature window was recorded during market hours.
    """
    print(f"--- Creating Single Training Example ---")
    
    # 1. Check if the input file exists.
    if not INPUT_PARQUET_FILE.exists():
        print(f"Error: Input file not found at '{INPUT_PARQUET_FILE}'")
        print("Please update the 'INPUT_PARQUET_FILE' variable in this script.")
        return

    print(f"Reading data from: {INPUT_PARQUET_FILE}")
    
    # 2. Load the full dataset into a Polars DataFrame.
    full_df = pl.read_parquet(INPUT_PARQUET_FILE)
    
    # 3. Filter the DataFrame to find all "highest-quality" entries.
    # The .list.sum() operation on the boolean `masking_array` will be 0
    # only if all values in the list are False.
    highest_quality_df = full_df.filter(
        pl.col('masking_array').list.sum() == 0
    )
    
    # 4. Check if any such entries were found.
    if highest_quality_df.height == 0:
        print("Error: No 'highest-quality' (fully unmasked) entries were found in the input file.")
        return
        
    print(f"Found {highest_quality_df.height} highest-quality entries.")
    
    # 5. Select only the very first entry from the filtered DataFrame.
    single_example_df = highest_quality_df.head(1)
    
    # 6. Create the output directory if it doesn't already exist.
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    
    # 7. Write the single-row DataFrame to the new Parquet file.
    single_example_df.write_parquet(OUTPUT_PARQUET_FILE)
    
    print(f"\nSuccessfully created a single-example file.")
    print(f" -> Saved to: {OUTPUT_PARQUET_FILE}")
    print(f"You can now use this file to test your training pipeline.")
    print("--- Finished ---")

if __name__ == "__main__":
    create_single_example_file()