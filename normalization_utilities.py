import numpy as np
import math

def normalize_option_data(document, stats_from_db: dict[str, dict]):
    """
    Normalizes option data from a MongoDB-like document into a numpy array using statistics from the database.

    Args:
        document (dict): A dictionary containing the option data.
        stats_from_db (dict[str, dict]): A dictionary where keys are feature names and
                                         values are dictionaries containing 'mean' and 'std_dev'.

    Returns:
        numpy.ndarray: A numpy array containing the normalized option metrics.
    """
    underlying_price = document.get("underlyingPrice", 0)
    strike_price = document.get("strikePrice", 0)
    option_price = document.get("optionPrice", 0)
    volume = document.get("volume", 0)
    gamma = document.get("gamma", 0)
    theta = document.get("theta", 0)
    vega = document.get("vega", 0)
    hours_to_expiration = document.get("hoursToExpiration", 0)
    volatility = document.get("impliedVolatility", 0) # Assuming this is a percentage value
    intrinsic_value_per_share = document.get("intrinsicValue", 0)
    extrinsic_value_per_share = document.get("extrinsicValue", 0)
    # description = document.get("overallDescription", "")


    # print(f"\n \n ----- DESCRIPTION = {description} -----")

    # print(f"\n \n ----- stats from db is = {stats_from_db} -----")


    normalized_features = []

    # 1. Option price ratio to underlying stock price
    option_price_ratio_to_underlying = option_price / (underlying_price + 1e-6)
    if "optionPriceRatioToUnderlying" in stats_from_db:
        mean = stats_from_db["optionPriceRatioToUnderlying"].get("mean", 0)
        std_dev = stats_from_db["optionPriceRatioToUnderlying"].get("std_dev", 1e-6)
        normalized_value = (option_price_ratio_to_underlying - mean) / std_dev
    else:
        print("stats not found for optionPriceRatioToUnderlying")
        normalized_value = option_price_ratio_to_underlying
    normalized_features.append(normalized_value)
    # print(f"Option price ratio to underlying stock price {normalized_value}")

    # 2. Option price ratio to intrinsic value
    intrinsic_value_total = max(0, intrinsic_value_per_share * 100)
    option_price_ratio_to_intrinsic_value = option_price / (intrinsic_value_total + 1e-6) if intrinsic_value_total > 0 else 0
    if "optionPriceRatioToIntrinsicValue" in stats_from_db:
        mean = stats_from_db["optionPriceRatioToIntrinsicValue"].get("mean", 0)
        std_dev = stats_from_db["optionPriceRatioToIntrinsicValue"].get("std_dev", 1e-6)
        normalized_value = (option_price_ratio_to_intrinsic_value - mean) / std_dev
    else:
        print("stats not found for optionPriceRatioToIntrinsicValue")
        normalized_value = option_price_ratio_to_intrinsic_value
    normalized_features.append(normalized_value)
    # print(f"Option price ratio to intrinsic value {normalized_value}")

    # 3. Option price ratio to extrinsic value
    extrinsic_value_total = max(0, extrinsic_value_per_share * 100)
    option_price_ratio_to_extrinsic_value = option_price / (extrinsic_value_total + 1e-6) if extrinsic_value_total > 0 else 0
    if "optionPriceRatioToExtrinsicValue" in stats_from_db:
        mean = stats_from_db["optionPriceRatioToExtrinsicValue"].get("mean", 0)
        std_dev = stats_from_db["optionPriceRatioToExtrinsicValue"].get("std_dev", 1e-6)
        normalized_value = (option_price_ratio_to_extrinsic_value - mean) / std_dev
    else:
        print("stats not found for optionPriceRatioToExtrinsicValue")
        normalized_value = option_price_ratio_to_extrinsic_value
    normalized_features.append(normalized_value)
    # print(f"Option price ratio to extrinsic value {normalized_value}")

    # 4. Strike price ratio to underlying stock price
    strike_price_ratio_to_underlying = strike_price / (underlying_price + 1e-6)
    if "strikePriceRatioToUnderlying" in stats_from_db:
        mean = stats_from_db["strikePriceRatioToUnderlying"].get("mean", 0)
        std_dev = stats_from_db["strikePriceRatioToUnderlying"].get("std_dev", 1e-6)
        normalized_value = (strike_price_ratio_to_underlying - mean) / std_dev
    else:
        print("stats not found for strikePriceRatioToUnderlying")
        normalized_value = strike_price_ratio_to_underlying
    normalized_features.append(normalized_value)
    # print(f"Strike price ratio to underlying stock price {normalized_value}")

    # 5. Strike price ratio to option price
    strike_price_ratio_to_option_price = strike_price / (option_price + 1e-6) if option_price > 0 else 0
    if "strikePriceRatioToOptionPrice" in stats_from_db:
        mean = stats_from_db["strikePriceRatioToOptionPrice"].get("mean", 0)
        std_dev = stats_from_db["strikePriceRatioToOptionPrice"].get("std_dev", 1e-6)
        normalized_value = (strike_price_ratio_to_option_price - mean) / std_dev
    else:
        print("stats not found for strikePriceRatioToOptionPrice")
        normalized_value = strike_price_ratio_to_option_price
    normalized_features.append(normalized_value)
    # print(f"Strike price ratio to option price {normalized_value}")

    # 6. Volatility
    if "volatility" in stats_from_db:
        mean = stats_from_db["volatility"].get("mean", 0)
        std_dev = stats_from_db["volatility"].get("std_dev", 1e-6)
        normalized_value = (volatility - mean) / std_dev
    else:
        print("stats not found for volatility")
        normalized_value = volatility
    normalized_features.append(normalized_value)
    # print(f"Volatility {normalized_value}")

    # 7. Theta
    if "theta" in stats_from_db:
        mean = stats_from_db["theta"].get("mean", 0)
        std_dev = stats_from_db["theta"].get("std_dev", 1e-6)
        normalized_value = (theta - mean) / std_dev
    else:
        print("stats not found for theta")
        normalized_value = theta
    normalized_features.append(normalized_value)
    # print(f"theta {normalized_value}")

    # 8. Vega
    if "vega" in stats_from_db:
        mean = stats_from_db["vega"].get("mean", 0)
        std_dev = stats_from_db["vega"].get("std_dev", 1e-6)
        normalized_value = (vega - mean) / std_dev
    else:
        print("stats not found for vega")
        normalized_value = vega
    normalized_features.append(normalized_value)
    # print(f"vega {normalized_value}")

    # 9. Gamma
    if "gamma" in stats_from_db:
        mean = stats_from_db["gamma"].get("mean", 0)
        std_dev = stats_from_db["gamma"].get("std_dev", 1e-6)
        normalized_value = (gamma - mean) / std_dev
    else:
        print("stats not found for gamma")
        normalized_value = gamma
    normalized_features.append(normalized_value)
    # print(f"gamma {normalized_value}")

    # 10. Volume (Logarithmic base 10)
    log10_volume = 0
    if volume > 0:
        log10_volume = math.log10(volume + 1e-6)
    if "log10Volume" in stats_from_db:
        mean = stats_from_db["log10Volume"].get("mean", 0)
        std_dev = stats_from_db["log10Volume"].get("std_dev", 1e-6)
        normalized_value = (log10_volume - mean) / std_dev
    else:
        print("stats not found for volume")
        normalized_value = log10_volume
    normalized_features.append(normalized_value)
    # print(f"volume {normalized_value}")

    # 11. Hours to expiration
    two_weeks_in_hours = 7 * 24 * 2.0
    hours_to_expiration_norm = hours_to_expiration / two_weeks_in_hours
    normalized_features.append(hours_to_expiration_norm)
    # print(f"Hours to expiration {hours_to_expiration_norm}")

    return np.array(normalized_features)