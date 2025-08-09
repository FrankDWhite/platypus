#!/usr/bin/python3.10

import polars as pl

file_path = './training_data/2025-07-24_options_training_data.parquet'
df = pl.read_parquet(file_path)


# Filter for rows where the masking_array sum is 0 (all False).
# Then, get the .height (number of rows) of the resulting DataFrame.
count = df.filter(
    pl.col('masking_array').list.sum() == 0
).height

print(f"Total number of 'highest-quality' entries: {count}")

print(f"The total number of examples in the file is: {df.height}")


# Get your sample of 5 entries
highest_quality_entries = df.filter(
    pl.col('masking_array').list.sum() == 0
).head(1)

# --- Loop through the results and print each one vertically ---
# .to_dicts() converts the DataFrame rows into a list of Python dictionaries
for i, row_dict in enumerate(highest_quality_entries.to_dicts()):
    print(f"--- Entry {i + 1} ---")
    for column_name, value in row_dict.items():
        print(f"'{column_name}': {value}")
    print("-" * 20 + "\n") # Add a separator for readability