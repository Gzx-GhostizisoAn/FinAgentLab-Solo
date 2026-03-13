import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from sklearn.preprocessing import MinMaxScaler
import time
warnings.filterwarnings('ignore')

TICKERS = ['SPY', 'XLE', 'XLK', 'XLF']
START_DATE = '2018-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')

def fetch_financial_data(tickers, start, end):
    ticker_data = {}
    for ticker in tickers:
        print(f"正在采集 {ticker} 数据...")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                data = yf.Ticker(ticker).history(start=start, end=end, interval='1d')
                data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
                data['Ticker'] = ticker
                data = data.reset_index()
                ticker_data[ticker] = data
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    print(f"API限流，等待{wait_time}秒后重试...")
                    time.sleep(wait_time)
                else:
                    raise e
        time.sleep(2)  
    combined_data = pd.concat(ticker_data.values(), ignore_index=True)
    return combined_data

def calculate_technical_indicators(df):
    for ticker in TICKERS:
        mask = df['Ticker'] == ticker
        close = df.loc[mask, 'Close']
        high = df.loc[mask, 'High']
        low = df.loc[mask, 'Low']
        volume = df.loc[mask, 'Volume']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df.loc[mask, 'RSI'] = 100 - (100 / (1 + rs))
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df.loc[mask, 'MACD'] = ema12 - ema26
        df.loc[mask, 'MACD_Signal'] = df.loc[mask, 'MACD'].ewm(span=9, adjust=False).mean()
        sma20 = close.rolling(window=20).mean()
        std20 = close.rolling(window=20).std()
        df.loc[mask, 'BB_Mid'] = sma20
        df.loc[mask, 'BB_Upper'] = sma20 + 2 * std20
        df.loc[mask, 'BB_Lower'] = sma20 - 2 * std20
        df.loc[mask, 'Momentum'] = close - close.shift(14)
        vol_ma20 = volume.rolling(window=20).mean()
        df.loc[mask, 'Volume_Pressure'] = (volume - vol_ma20) / vol_ma20
    return df

def preprocess_data(raw_data):
    df = raw_data.copy()
    df = df.dropna(subset=['Close'])
    df = df.ffill()  
    
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[col] = df[col].clip(lower, upper)
    
    df = df.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)
    for ticker in TICKERS:
        mask = df['Ticker'] == ticker
        df.loc[mask, 'Daily_Return'] = df.loc[mask, 'Close'].pct_change()
        df.loc[mask, 'MA5'] = df.loc[mask, 'Close'].rolling(window=5).mean()
        df.loc[mask, 'MA20'] = df.loc[mask, 'Close'].rolling(window=20).mean()
        df.loc[mask, 'Rolling_Volatility_14d'] = df.loc[mask, 'Daily_Return'].rolling(window=14).std()
        df.loc[mask, 'High_Low_Ratio'] = (df.loc[mask, 'High'] - df.loc[mask, 'Low']) / df.loc[mask, 'Close']
        df.loc[mask, 'Volume_Change'] = df.loc[mask, 'Volume'].pct_change()
    df = calculate_technical_indicators(df)
    spy_returns = df[df['Ticker'] == 'SPY'][['Date', 'Daily_Return']].rename(columns={'Daily_Return': 'SPY_Return'})
    df = df.merge(spy_returns, on='Date', how='left')
    df['Excess_Return'] = df['Daily_Return'] - df['SPY_Return']
    for ticker in TICKERS:
        mask = df['Ticker'] == ticker
        df.loc[mask, 'Risk_Threshold'] = -2 * df.loc[mask, 'Rolling_Volatility_14d']
        df.loc[mask, 'Risk_Label'] = (df.loc[mask, 'Daily_Return'] < df.loc[mask, 'Risk_Threshold']).astype(int)
    
    df = df.dropna().reset_index(drop=True)
    return df

def split_train_test(df, test_ratio=0.2):
    all_dates = sorted(df['Date'].unique())
    split_date = all_dates[int(len(all_dates) * (1 - test_ratio))]
    train_data = df[df['Date'] <= split_date]
    test_data = df[df['Date'] > split_date]
    
    print(f"拆分日期：{split_date}")
    print(f"训练集时间范围：{train_data['Date'].min()} ~ {train_data['Date'].max()}")
    print(f"测试集时间范围：{test_data['Date'].min()} ~ {test_data['Date'].max()}")
    print(f"训练集样本数：{len(train_data)}, 测试集样本数：{len(test_data)}")
    
    feature_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'Daily_Return', 'MA5', 'MA20', 'Rolling_Volatility_14d',
        'High_Low_Ratio', 'Volume_Change', 'RSI', 'MACD',
        'MACD_Signal', 'BB_Mid', 'BB_Upper', 'BB_Lower',
        'Momentum', 'Volume_Pressure', 'SPY_Return', 'Excess_Return'
    ]
    
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train_data[feature_cols])
    X_test = scaler.transform(test_data[feature_cols])
    
    y_train = train_data['Risk_Label'].values
    y_test = test_data['Risk_Label'].values
    
    return X_train, X_test, y_train, y_test, train_data, test_data, scaler, feature_cols

if __name__ == "__main__":
    raw_data = fetch_financial_data(TICKERS, START_DATE, END_DATE)
    print(f"原始数据形状：{raw_data.shape}")
    
    processed_data = preprocess_data(raw_data)
    print(f"预处理后数据形状：{processed_data.shape}")
    print("\n预处理后数据示例（行业超额收益+风险标签）：")
    print(processed_data[['Date', 'Ticker', 'Daily_Return', 'SPY_Return', 'Excess_Return', 'Risk_Label']].head(10))
    
    X_train, X_test, y_train, y_test, train_df, test_df, scaler, feature_cols = split_train_test(processed_data)
    
    processed_data.to_csv('sector_risk_data_spy_xle_xlk_xlf.csv', index=False)
    print(f"\n特征列表：{feature_cols}")
    print("\n数据已保存为 sector_risk_data_spy_xle_xlk_xlf.csv")