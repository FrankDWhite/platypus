# embedding_utils.py

# --- Vocabularies ---
# The specific list of 7 stocks you want to track.
TICKER_VOCABULARY = [
    "MSFT", "NVDA", "AMZN", "META", "AAPL", "GOOGL", "TSLA"
]

# The two types of options.
OPTION_TYPE_VOCABULARY = [
    "CALL", "PUT"
]

# --- Mappings ---
# Create a mapping from the string token to a unique integer.
# The Keras Embedding layer will use these integers as input.
# We reserve 0 for "unknown" or "out-of-vocabulary" tokens.
TICKER_TO_INT = {ticker: i + 1 for i, ticker in enumerate(TICKER_VOCABULARY)}
OPTION_TYPE_TO_INT = {opt_type: i + 1 for i, opt_type in enumerate(OPTION_TYPE_VOCABULARY)}

def get_ticker_encoding(ticker: str) -> int:
    """
    Converts a ticker symbol string to its integer encoding.

    Args:
        ticker (str): The stock ticker symbol (e.g., "AMZN").

    Returns:
        int: The integer encoding for the ticker, or 0 if not in the vocabulary.
    """
    return TICKER_TO_INT.get(ticker, 0) # Returns 0 for unknown tickers

def get_option_type_encoding(option_type: str) -> int:
    """
    Converts an option type string to its integer encoding.

    Args:
        option_type (str): The option type ("CALL" or "PUT").

    Returns:
        int: The integer encoding for the option type, or 0 if not in the vocabulary.
    """
    return OPTION_TYPE_TO_INT.get(option_type, 0)

def get_vocabulary_size(embedding_type: str) -> int:
    """
    Returns the size of the vocabulary for a given embedding type.
    This is useful for defining the input_dim of a Keras Embedding layer.
    We add 1 to account for the "unknown" token (0).

    Args:
        embedding_type (str): "ticker" or "option_type".

    Returns:
        int: The total size of the vocabulary.
    """
    if embedding_type == "ticker":
        return len(TICKER_VOCABULARY) + 1
    elif embedding_type == "option_type":
        return len(OPTION_TYPE_VOCABULARY) + 1
    else:
        raise ValueError("Unknown embedding_type specified.")