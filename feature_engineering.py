import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """金融数据特征工程处理引擎"""

    def __init__(self):
        """初始化特征工程师"""
        pass

    def process_features(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理原始数据并生成特征
        
        Args:
            raw_data: 包含价格数据、技术指标等的原始数据字典
            
        Returns:
            包含特征数据和元数据的字典
        """
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
        """
        处理上市公司特征
        
        Args:
            raw_data: 包含价格、技术指标等数据的字典
            
        Returns:
            特征DataFrame
        """
        price_df = pd.DataFrame(raw_data.get("price_data", []))
        tech_df = pd.DataFrame(raw_data.get("tech_indicators", []))
        financial_data = raw_data.get("financial_data", {})
        news_data = raw_data.get("news_data", {})
        
        # 验证数据
        if price_df.empty or tech_df.empty:
            raise ValueError("价格数据或技术指标为空")
        
        # 日期转换
        price_df["Date"] = pd.to_datetime(price_df["Date"])
        tech_df["Date"] = pd.to_datetime(tech_df["Date"])
        df = pd.merge(price_df, tech_df, on="Date", how="left")
        df = df.sort_values("Date").reset_index(drop=True)
        
        # 返回率特征
        df["daily_return"] = df["Close"].pct_change()
        # 安全的log_return计算，避免负数或0
        df["log_return"] = np.where(
            df["Close"] > 0,
            np.log(df["Close"] / df["Close"].shift(1).replace(0, np.nan)),
            np.nan
        )
        df["return_lag1"] = df["daily_return"].shift(1)
        df["return_lag2"] = df["daily_return"].shift(2)
        df["return_lag5"] = df["daily_return"].shift(5)
        
        
        # 波动率特征
        if "volatility_20d" in df.columns:
            df["volatility_lag1"] = df["volatility_20d"].shift(1)
            df["volatility_change"] = df["volatility_20d"].diff()
            df["volatility_pct_change"] = df["volatility_20d"].pct_change()
        
        # 动量特征
        for lag in [5, 20, 60]:
            col_name = f"momentum_{lag}d"
            df[col_name] = np.where(
                df["Close"].shift(lag) > 0,
                df["Close"] / df["Close"].shift(lag) - 1,
                np.nan
            )
        
        # 价格偏差特征
        if "MA5" in df.columns:
            df["price_dev_ma5"] = np.where(
                df["MA5"] > 0,
                (df["Close"] - df["MA5"]) / df["MA5"],
                np.nan
            )
        if "MA20" in df.columns:
            df["price_dev_ma20"] = np.where(
                df["MA20"] > 0,
                (df["Close"] - df["MA20"]) / df["MA20"],
                np.nan
            )
        
        # RSI特征
        if "RSI_14" in df.columns:
            df["rsi_change"] = df["RSI_14"].diff()
            df["rsi_overbought"] = (df["RSI_14"] > 70).astype(int)
            df["rsi_oversold"] = (df["RSI_14"] < 30).astype(int)
        
        # 成交量特征
        if "Volume" in df.columns:
            df["volume_change"] = df["Volume"].pct_change()
            df["volume_ma5"] = df["Volume"].rolling(5).mean()
            df["volume_ma20"] = df["Volume"].rolling(20).mean()
            df["volume_ratio"] = np.where(
                df["volume_ma20"] > 0,
                df["Volume"] / df["volume_ma20"],
                np.nan
            )
        

# 财务数据标志
        df["has_financial_data"] = int(financial_data and "error" not in financial_data)
        
        # 新闻数据特征
        if news_data and "error" not in news_data:
            sentiment = news_data.get("sentiment_analysis", {})
            df["news_sentiment_mean"] = sentiment.get("sentiment_mean", 0)
            df["news_negative_ratio"] = sentiment.get("negative_ratio", 0)
            df["has_news_data"] = 1
        else:
            df["news_sentiment_mean"] = 0
            df["news_negative_ratio"] = 0
            df["has_news_data"] = 0
        
        # 时间特征
        df["day_of_week"] = df["Date"].dt.dayofweek
        df["month"] = df["Date"].dt.month
        df["quarter"] = df["Date"].dt.quarter
        df["is_month_end"] = df["Date"].dt.is_month_end.astype(int)
        df["is_quarter_end"] = df["Date"].dt.is_quarter_end.astype(int)
        
        return df.dropna().reset_index(drop=True)
    def _process_sector_features(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """
        处理行业特征
        
        Args:
            raw_data: 包含价格、技术指标、相对强度等数据的字典
            
        Returns:
            特征DataFrame
        """
        price_df = pd.DataFrame(raw_data.get("price_data", []))
        tech_df = pd.DataFrame(raw_data.get("tech_indicators", []))
        relative_df = pd.DataFrame(raw_data.get("relative_strength", []))
        volume_df = pd.DataFrame(raw_data.get("volume_trend", []))
        news_data = raw_data.get("news_data", {})
        
        # 验证数据
        if price_df.empty:
            raise ValueError("价格数据为空")
        
        # 日期转换
        price_df["Date"] = pd.to_datetime(price_df["Date"])
        if not tech_df.empty:
            tech_df["Date"] = pd.to_datetime(tech_df["Date"])
        if not relative_df.empty:
            relative_df["Date"] = pd.to_datetime(relative_df["Date"])
        if not volume_df.empty:
            volume_df["Date"] = pd.to_datetime(volume_df["Date"])
        
        # 合并数据
        df = price_df.copy()
        if not tech_df.empty:
            df = pd.merge(df, tech_df, on="Date", how="left")
        if not relative_df.empty:
            df = pd.merge(df, relative_df, on="Date", how="left")
        if not volume_df.empty:
            df = pd.merge(df, volume_df, on="Date", how="left")
        df = df.sort_values("Date").reset_index(drop=True)
        
        # 相对动量特征
        if "cumulative_return_vs_spy" in df.columns:
            df["relative_momentum_5d"] = df["cumulative_return_vs_spy"].diff(5)
            df["relative_momentum_20d"] = df["cumulative_return_vs_spy"].diff(20)
        
        # 相对强度变化
        if "return_vs_spy" in df.columns:
            df["relative_strength_change"] = df["return_vs_spy"].diff()
        
        # 成交量比例变化
        if "volume_vs_spy" in df.columns:
            df["volume_ratio_change"] = df["volume_vs_spy"].pct_change()
        
        
        # 添加通用特征
        df = self._add_generic_features(df)
        
        # 新闻特征
        if news_data and "error" not in news_data:
            sentiment = news_data.get("sentiment_analysis", {})
            df["sector_news_sentiment"] = sentiment.get("sentiment_mean", 0)
            df["has_sector_news"] = 1
        else:
            df["sector_news_sentiment"] = 0
            df["has_sector_news"] = 0
        
        return df.dropna().reset_index(drop=True)
    def _process_market_features(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """
        处理整体市场特征
        
        Args:
            raw_data: 包含市场核心数据、指标等的字典
            
        Returns:
            特征DataFrame
        """
        core_df = pd.DataFrame(raw_data.get("market_core_data", []))
        indicator_df = pd.DataFrame(raw_data.get("market_indicators", []))
        vix_df = pd.DataFrame(raw_data.get("vix_analysis", []))
        macro_df = pd.DataFrame(raw_data.get("macro_indicators", []))
        
        # 验证数据
        if core_df.empty:
            raise ValueError("市场核心数据为空")
        
        # 日期转换
        core_df["Date"] = pd.to_datetime(core_df["Date"])
        if not indicator_df.empty:
            indicator_df["Date"] = pd.to_datetime(indicator_df["Date"])
        if not vix_df.empty:
            vix_df["Date"] = pd.to_datetime(vix_df["Date"])
        if not macro_df.empty:
            macro_df["Date"] = pd.to_datetime(macro_df["Date"])
        
        # 合并数据
        df = core_df.copy()
        if not indicator_df.empty:
            df = pd.merge(df, indicator_df, on="Date", how="left")
        if not vix_df.empty:
            df = pd.merge(df, vix_df, on="Date", how="left")
        if not macro_df.empty:
            df = pd.merge(df, macro_df, on="Date", how="left")
        df = df.sort_values("Date").reset_index(drop=True)
        
        # 科技vs大盘特征
        if "QQQ_Close" in df.columns and "Close" in df.columns:
            df["qqq_vs_spy"] = np.where(
                df["Close"] > 0,
                df["QQQ_Close"] / df["Close"] - 1,
                np.nan
            )
            df["qqq_vs_spy_change"] = df["qqq_vs_spy"].diff()
        
        # VIX特征
        if "VIX_history_percentile" in df.columns:
            df["vix_extreme_high"] = (df["VIX_history_percentile"] > 90).astype(int)
            df["vix_extreme_low"] = (df["VIX_history_percentile"] < 10).astype(int)
        if "VIX" in df.columns:
            df["vix_change"] = df["VIX"].pct_change()
        
        # 宏观经济特征
        if "Dollar_Index" in df.columns:
            df["dollar_change"] = df["Dollar_Index"].pct_change()
        if "10Y_Treasury_Yield" in df.columns:
            df["treasury_yield_change"] = df["10Y_Treasury_Yield"].diff()
        
        
        # 添加通用特征
        df = self._add_generic_features(df)
        
        return df.dropna().reset_index(drop=True)
    
    def _add_generic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加通用技术特征
        
        Args:
            df: 原始DataFrame
            
        Returns:
            添加特征后的DataFrame
        """
        # 返回率特征
        if "Close" in df.columns:
            df["daily_return"] = df["Close"].pct_change()
            df["log_return"] = np.where(
                df["Close"] > 0,
                np.log(df["Close"] / df["Close"].shift(1).replace(0, np.nan)),
                np.nan
            )
            
            # 动量特征
            for lag in [5, 20]:
                col_name = f"momentum_{lag}d"
                df[col_name] = np.where(
                    df["Close"].shift(lag) > 0,
                    df["Close"] / df["Close"].shift(lag) - 1,
                    np.nan
                )
        
        # 移动平均偏差特征
        if "MA5" in df.columns and "Close" in df.columns:
            df["price_dev_ma5"] = np.where(
                df["MA5"] > 0,
                (df["Close"] - df["MA5"]) / df["MA5"],
                np.nan
            )
        if "MA20" in df.columns and "Close" in df.columns:
            df["price_dev_ma20"] = np.where(
                df["MA20"] > 0,
                (df["Close"] - df["MA20"]) / df["MA20"],
                np.nan
            )
        
        # 时间特征
        if "Date" in df.columns:
            df["day_of_week"] = df["Date"].dt.dayofweek
            df["month"] = df["Date"].dt.month
            df["quarter"] = df["Date"].dt.quarter
        
        return df