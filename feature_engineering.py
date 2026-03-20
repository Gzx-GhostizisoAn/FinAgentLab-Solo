import pandas as pd
import numpy as np
from typing import Dict, Any

class FeatureEngineer:

    def __init__(self):
        pass
    
   
    def process_features(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
    
        if not raw_data.get("success"):
            return {"success": False, "error": "原始数据拉取失败，无法进行特征工程"}
        
        entity_type = raw_data.get("entity_type")
        
        try:
            if entity_type == "listed_company":
                features = self._process_company_features(raw_data)
            elif entity_type == "sector":
                features = self._process_sector_features(raw_data)
            elif entity_type == "market":
                features = self._process_market_features(raw_data)
            else:
                return {"success": False, "error": "不支持的主体类型"}
            
            return {
                "success": True,
                "entity_type": entity_type,
                "entity_name": raw_data.get("entity_name"),
                "feature_count": len(features.columns) if isinstance(features, pd.DataFrame) else 0,
                "feature_preview": features.head(10).to_dict("records") if isinstance(features, pd.DataFrame) else [],
                "feature_names": features.columns.tolist() if isinstance(features, pd.DataFrame) else [],
                "full_features": features  
            }
        
        except Exception as e:
            return {"success": False, "error": f"特征工程失败：{str(e)}"}
    
   
    def _process_company_features(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
     
        price_df = pd.DataFrame(raw_data["price_data"])
        tech_df = pd.DataFrame(raw_data["tech_indicators"])
        financial_data = raw_data.get("financial_data", {})
        news_data = raw_data.get("news_data", {})
        
 
        price_df["Date"] = pd.to_datetime(price_df["Date"])
        tech_df["Date"] = pd.to_datetime(tech_df["Date"])
        df = pd.merge(price_df, tech_df, on="Date", how="left")
        df = df.sort_values("Date").reset_index(drop=True)
        
   
        df["daily_return"] = df["Close"].pct_change()
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
        df["return_lag1"] = df["daily_return"].shift(1)
        df["return_lag2"] = df["daily_return"].shift(2)
        df["return_lag5"] = df["daily_return"].shift(5)
        
    
        df["volatility_lag1"] = df["volatility_20d"].shift(1)
        df["volatility_change"] = df["volatility_20d"].diff()
        df["volatility_pct_change"] = df["volatility_20d"].pct_change()
        
    
        df["momentum_5d"] = df["Close"] / df["Close"].shift(5) - 1
        df["momentum_20d"] = df["Close"] / df["Close"].shift(20) - 1
        df["momentum_60d"] = df["Close"] / df["Close"].shift(60) - 1
        
    
        df["price_dev_ma5"] = (df["Close"] - df["MA5"]) / df["MA5"]
        df["price_dev_ma20"] = (df["Close"] - df["MA20"]) / df["MA20"]
        
       
        df["rsi_change"] = df["RSI_14"].diff()
        df["rsi_overbought"] = (df["RSI_14"] > 70).astype(int)
        df["rsi_oversold"] = (df["RSI_14"] < 30).astype(int)
        
       
        df["volume_change"] = df["Volume"].pct_change()
        df["volume_ma5"] = df["Volume"].rolling(5).mean()
        df["volume_ma20"] = df["Volume"].rolling(20).mean()
        df["volume_ratio"] = df["Volume"] / df["volume_ma20"]
        

        if financial_data and "error" not in financial_data:
           
            df["has_financial_data"] = 1
          
        else:
            df["has_financial_data"] = 0
        
    
        if news_data and "error" not in news_data:
            sentiment = news_data.get("sentiment_analysis", {})
            df["news_sentiment_mean"] = sentiment.get("sentiment_mean", 0)
            df["news_negative_ratio"] = sentiment.get("negative_ratio", 0)
            df["has_news_data"] = 1
        else:
            df["news_sentiment_mean"] = 0
            df["news_negative_ratio"] = 0
            df["has_news_data"] = 0
        
    
        df["day_of_week"] = df["Date"].dt.dayofweek
        df["month"] = df["Date"].dt.month
        df["quarter"] = df["Date"].dt.quarter
        df["is_month_end"] = df["Date"].dt.is_month_end.astype(int)
        df["is_quarter_end"] = df["Date"].dt.is_quarter_end.astype(int)
        
       
        return df.dropna().reset_index(drop=True)
  
    def _process_sector_features(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
      
        price_df = pd.DataFrame(raw_data["price_data"])
        tech_df = pd.DataFrame(raw_data["tech_indicators"])
        relative_df = pd.DataFrame(raw_data["relative_strength"])
        volume_df = pd.DataFrame(raw_data["volume_trend"])
        news_data = raw_data.get("news_data", {})
        
    
        price_df["Date"] = pd.to_datetime(price_df["Date"])
        tech_df["Date"] = pd.to_datetime(tech_df["Date"])
        relative_df["Date"] = pd.to_datetime(relative_df["Date"])
        volume_df["Date"] = pd.to_datetime(volume_df["Date"])
        
        df = pd.merge(price_df, tech_df, on="Date", how="left")
        df = pd.merge(df, relative_df, on="Date", how="left")
        df = pd.merge(df, volume_df, on="Date", how="left")
        df = df.sort_values("Date").reset_index(drop=True)
        
      
        df["relative_momentum_5d"] = df["cumulative_return_vs_spy"].diff(5)
        df["relative_momentum_20d"] = df["cumulative_return_vs_spy"].diff(20)
        
      
        df["relative_strength_change"] = df["return_vs_spy"].diff()
        
       
        df["volume_ratio_change"] = df["volume_vs_spy"].pct_change()
        
       
        df = self._add_generic_features(df)
        
       
        if news_data and "error" not in news_data:
            sentiment = news_data.get("sentiment_analysis", {})
            df["sector_news_sentiment"] = sentiment.get("sentiment_mean", 0)
            df["has_sector_news"] = 1
        else:
            df["sector_news_sentiment"] = 0
            df["has_sector_news"] = 0
        
        return df.dropna().reset_index(drop=True)
    
  
    def _process_market_features(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
       
        core_df = pd.DataFrame(raw_data["market_core_data"])
        indicator_df = pd.DataFrame(raw_data["market_indicators"])
        vix_df = pd.DataFrame(raw_data["vix_analysis"])
        macro_df = pd.DataFrame(raw_data["macro_indicators"])
        
   
        core_df["Date"] = pd.to_datetime(core_df["Date"])
        indicator_df["Date"] = pd.to_datetime(indicator_df["Date"])
        vix_df["Date"] = pd.to_datetime(vix_df["Date"])
        macro_df["Date"] = pd.to_datetime(macro_df["Date"])
        
        df = pd.merge(core_df, indicator_df, on="Date", how="left")
        df = pd.merge(df, vix_df, on="Date", how="left")
        df = pd.merge(df, macro_df, on="Date", how="left")
        df = df.sort_values("Date").reset_index(drop=True)
      
        df["qqq_vs_spy"] = df["QQQ_Close"] / df["Close"] - 1
        df["qqq_vs_spy_change"] = df["qqq_vs_spy"].diff()
        
       
        df["vix_extreme_high"] = (df["VIX_history_percentile"] > 90).astype(int)
        df["vix_extreme_low"] = (df["VIX_history_percentile"] < 10).astype(int)
        df["vix_change"] = df["VIX"].pct_change()
        
      
        df["dollar_change"] = df["Dollar_Index"].pct_change()
        df["treasury_yield_change"] = df["10Y_Treasury_Yield"].diff()
        
       
        df = self._add_generic_features(df)
        
        return df.dropna().reset_index(drop=True)
    
 
    def _add_generic_features(self, df: pd.DataFrame) -> pd.DataFrame:
       
        df["daily_return"] = df["Close"].pct_change()
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
        df["momentum_5d"] = df["Close"] / df["Close"].shift(5) - 1
        df["momentum_20d"] = df["Close"] / df["Close"].shift(20) - 1
        
        
        if "MA5" in df.columns:
            df["price_dev_ma5"] = (df["Close"] - df["MA5"]) / df["MA5"]
        if "MA20" in df.columns:
            df["price_dev_ma20"] = (df["Close"] - df["MA20"]) / df["MA20"]
        
       
        df["day_of_week"] = df["Date"].dt.dayofweek
        df["month"] = df["Date"].dt.month
        df["quarter"] = df["Date"].dt.quarter
        
        return df