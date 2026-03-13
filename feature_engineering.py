import pandas as pd
import numpy as np



def load_data(path):

    df = pd.read_csv(path)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values(["Ticker","Date"])

    return df
def macd_features(df):

    g = df.groupby("Ticker")

    df["MACD_diff"] = df["MACD"] - df["MACD_Signal"]

    df["MACD_change"] = g["MACD"].diff()

    df["MACD_signal_change"] = g["MACD_Signal"].diff()

    df["MACD_momentum"] = g["MACD_diff"].diff()

    return df

def moving_average_features(df):

    g = df.groupby("Ticker")

    df["MA_ratio"] = df["MA5"] / df["MA20"]

    df["Price_MA5"] = df["Close"] / df["MA5"]

    df["Price_MA20"] = df["Close"] / df["MA20"]

    df["MA_diff"] = df["MA5"] - df["MA20"]

    df["MA_slope"] = g["MA5"].diff()

    df["MA_acceleration"] = g["MA_slope"].diff()

    return df
def bollinger_features(df):

    df["BB_width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Mid"]

    df["BB_position"] = (
        (df["Close"] - df["BB_Lower"])
        / (df["BB_Upper"] - df["BB_Lower"])
    )

    df["BB_upper_dist"] = (df["BB_Upper"] - df["Close"]) / df["Close"]

    df["BB_lower_dist"] = (df["Close"] - df["BB_Lower"]) / df["Close"]

    return df

def rsi_features(df):

    g = df.groupby("Ticker")

    df["RSI_change"] = g["RSI"].diff()

    df["RSI_ma5"] = g["RSI"].transform(lambda x: x.rolling(5).mean())

    df["RSI_volatility"] = g["RSI"].transform(lambda x: x.rolling(10).std())

    return df

def return_features(df):

    g = df.groupby("Ticker")

    df["Return_5d"] = g["Daily_Return"].transform(lambda x: x.rolling(5).sum())

    df["Return_10d"] = g["Daily_Return"].transform(lambda x: x.rolling(10).sum())

    df["Return_std_5d"] = g["Daily_Return"].transform(lambda x: x.rolling(5).std())

    df["Return_std_10d"] = g["Daily_Return"].transform(lambda x: x.rolling(10).std())

    df["Return_skew_10d"] = g["Daily_Return"].transform(lambda x: x.rolling(10).skew())

    df["Return_kurt_10d"] = g["Daily_Return"].transform(lambda x: x.rolling(10).kurt())

    return df
def volume_features(df):

    g = df.groupby("Ticker")

    df["Volume_MA5"] = g["Volume"].transform(lambda x: x.rolling(5).mean())

    df["Volume_MA20"] = g["Volume"].transform(lambda x: x.rolling(20).mean())

    df["Volume_ratio"] = df["Volume"] / df["Volume_MA5"]

    df["Volume_std_10"] = g["Volume"].transform(lambda x: x.rolling(10).std())

    df["Volume_spike"] = df["Volume"] / df["Volume_MA20"]

    return df

def volatility_features(df):

    g = df.groupby("Ticker")

    df["Volatility_5d"] = g["Daily_Return"].transform(lambda x: x.rolling(5).std())

    df["Volatility_10d"] = g["Daily_Return"].transform(lambda x: x.rolling(10).std())

    df["Volatility_ratio"] = df["Volatility_5d"] / df["Rolling_Volatility_14d"]

    df["Volatility_change"] = g["Rolling_Volatility_14d"].diff()

    return df

def market_features(df):

    df["Market_relative_return"] = df["Daily_Return"] - df["SPY_Return"]

    df["Return_SPY_ratio"] = df["Daily_Return"] / df["SPY_Return"]

    df["Momentum_market"] = df["Momentum"] * df["SPY_Return"]

    return df
def build_feature_dataset(input_path, output_path):

    df = load_data(input_path)

    df = macd_features(df)

    df = moving_average_features(df)

    df = bollinger_features(df)

    df = rsi_features(df)

    df = return_features(df)

    df = volume_features(df)

    df = volatility_features(df)

    df = market_features(df)

    df = df.replace([np.inf, -np.inf], np.nan)

    df = df.dropna()

    df.to_csv(output_path, index=False)

    print("Final dataset shape:", df.shape)

    return df

if __name__ == "__main__":

    INPUT_FILE = "sector_risk_data_spy_xle_xlk_xlf.csv"

    OUTPUT_FILE = "sector_ml_dataset.csv"

    dataset = build_feature_dataset(INPUT_FILE, OUTPUT_FILE)

    print(dataset.head())