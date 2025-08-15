#!/usr/bin/python3.10

import pickle
from pathlib import Path
import requests
import pprint

# --- Import Project Utilities ---
from mcp_server import InferenceRequest

# --- Configuration ---
# Updated to point to the new batch file.
TEST_VECTOR_FILE = Path("./models/inference_test_batch.pkl")
MCP_SERVER_URL = "http://127.0.0.1:8000/v1/infer"

def run_test():
    """
    Loads the saved batch of 15 test vectors, sends them to the live MCP server,
    and prints the ranked recommendations.
    """
    print("--- 🧪 Running Inference Test on Saved Vector Batch ---")

    # --- Step 1: Load the saved test vector batch ---
    if not TEST_VECTOR_FILE.exists():
        print(f"Error: Test vector file not found at '{TEST_VECTOR_FILE}'")
        print("Please run create_test_vector.py first to create it.")
        return
        
    print(f"Loading test vector batch from: {TEST_VECTOR_FILE}")
    with open(TEST_VECTOR_FILE, 'rb') as f:
        # The .pkl file now contains a list of OptionVector objects.
        test_vectors_batch = pickle.load(f)

    print(f"Successfully loaded a batch of {len(test_vectors_batch)} vectors.")

    # --- Step 2: Prepare the API Request ---
    # The list of vectors from the file is already in the correct format for the InferenceRequest.
    inference_payload = InferenceRequest(vectors=test_vectors_batch)
    
    request_data = inference_payload.dict()

    print(f"\n--- Sending batch of {len(test_vectors_batch)} vectors to MCP server ---")

    # --- Step 3: Call the Server and Print the Response ---
    try:
        response = requests.post(MCP_SERVER_URL, json=request_data)
        response.raise_for_status()

        ranked_results = response.json()
        
        print("\n--- ✅ MCP Server Response Received ---")
        pprint.pprint(ranked_results)
        print("------------------------------------")

    except requests.exceptions.RequestException as e:
        print(f"\n--- ❌ ERROR: Could not connect to the MCP server. ---")
        print(f"Please ensure the mcp_server.py is running. Details: {e}")

    print("\n--- ✅ Inference Test Complete ---")


if __name__ == "__main__":
    run_test()