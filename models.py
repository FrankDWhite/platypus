# Import necessary types from Pydantic and the standard 'typing' module
from pydantic import BaseModel, Field
from typing import List, Dict

# Represents the 'optionDeliverablesList' object
class OptionDeliverable(BaseModel):
    assetType: str
    deliverableUnits: float
    symbol: str

# Represents the main option data object found in both call and put maps
class OptionData(BaseModel):
    ask: float
    askSize: int
    bid: float
    bidAskSize: str
    bidSize: int
    closePrice: float
    daysToExpiration: int
    deliverableNote: str
    delta: float
    description: str
    exchangeName: str
    exerciseType: str
    expirationDate: str
    expirationType: str
    extrinsicValue: float
    gamma: float
    high52Week: float
    highPrice: float
    inTheMoney: bool
    intrinsicValue: float
    last: float
    lastSize: int
    lastTradingDay: int
    low52Week: float
    lowPrice: float
    mark: float
    markChange: float
    markPercentChange: float
    mini: bool
    multiplier: float
    netChange: float
    nonStandard: bool
    openInterest: int
    openPrice: float
    optionDeliverablesList: List[OptionDeliverable]
    optionRoot: str
    pennyPilot: bool
    percentChange: float
    putCall: str
    quoteTimeInLong: int
    rho: float
    settlementType: str
    strikePrice: float
    symbol: str
    theoreticalOptionValue: float
    theoreticalVolatility: float
    theta: float
    timeValue: float
    totalVolume: int
    tradeTimeInLong: int
    vega: float
    volatility: float

# Represents the 'underlying' asset data
class UnderlyingData(BaseModel):
    ask: float
    askSize: int
    bid: float
    bidSize: int
    change: float
    close: float
    delayed: bool
    description: str
    exchangeName: str
    fiftyTwoWeekHigh: float = Field(alias='fiftyTwoWeekHigh')
    fiftyTwoWeekLow: float = Field(alias='fiftyTwoWeekLow')
    highPrice: float
    last: float
    lowPrice: float
    mark: float
    markChange: float
    markPercentChange: float
    openPrice: float
    percentChange: float
    quoteTime: int
    symbol: str
    totalVolume: int
    tradeTime: int

# Represents the top-level API response object
class OptionChainResponse(BaseModel):
    assetMainType: str
    assetSubType: str
    callExpDateMap: Dict[str, Dict[str, List[OptionData]]]
    daysToExpiration: float
    dividendYield: float
    interestRate: float
    interval: float
    isChainTruncated: bool
    isDelayed: bool
    isIndex: bool
    numberOfContracts: int
    putExpDateMap: Dict[str, Dict[str, List[OptionData]]]
    status: str
    strategy: str
    symbol: str
    underlying: UnderlyingData
    underlyingPrice: float
    volatility: float
#Notes

# Intrinsic Value (Call) = max(0, Underlying Price - Strike Price) * 100
# Intrinsic Value (Put) = max(0, Strike Price - Underlying Price) * 100