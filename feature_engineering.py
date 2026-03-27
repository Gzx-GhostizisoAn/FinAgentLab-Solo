import logging
from typing import Any, Dict

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    def process_features(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        if not raw_data.get("success"):
            return {"success": False, "error": "Raw data unavailable"}

        entity_type = raw_data.get("entity_type")
        try:
            if entity_type == "listed_company":
                features = self._process_company_features(raw_data)
            elif entity_type == "sector":
                features = self._process_sector_features(raw_data)
            elif entity_type == "market":
                features = self._process_market_features(raw_data)
            else:
                return {"success": False, "error": "Unsupported entity type"}

            if features.empty:
                return {"success": False, "error": "Feature result is empty"}

            preview = features.head(10).copy()
            if "Date" in preview.columns:
                preview["Date"] = pd.to_datetime(preview["Date"]).dt.strftime("%Y-%m-%d")

            return {
                "success": True,
                "entity_type": entity_type,
                "entity_name": raw_data.get("entity_name"),
                "feature_count": len(features.columns),
                "feature_preview": preview.to_dict("records"),
                "feature_names": features.columns.tolist(),
                "full_features": features,
            }
        except Exception as exc:
            return {"success": False, "error": f"Feature engineering failed: {exc}"}

    def _process_company_features(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        price_df = pd.DataFrame(raw_data.get("price_data", []))
        tech_df = pd.DataFrame(raw_data.get("tech_indicators", []))
        if price_df.empty:
            raise ValueError("Company price data is empty")

        df = self._merge_price_and_tech(price_df, tech_df)
        df["has_financial_data"] = 1 if raw_data.get("financial_data") and "error" not in raw_data.get("financial_data", {}) else 0
        df = self._attach_news_features(df, raw_data.get("news_data", {}), prefix="")
        return self._finalize(df)

    def _process_sector_features(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        price_df = pd.DataFrame(raw_data.get("price_data", []))
        tech_df = pd.DataFrame(raw_data.get("tech_indicators", []))
        relative_df = pd.DataFrame(raw_data.get("relative_strength", []))
        volume_df = pd.DataFrame(raw_data.get("volume_trend", []))
        if price_df.empty:
            raise ValueError("Sector price data is empty")

        df = self._merge_price_and_tech(price_df, tech_df)
        for extra_df in [relative_df, volume_df]:
            if not extra_df.empty and "Date" in extra_df.columns:
                extra_df["Date"] = pd.to_datetime(extra_df["Date"])
                df = df.merge(extra_df, on="Date", how="left")

        if "return_vs_spy" in df.columns:
            df["relative_strength_change"] = df["return_vs_spy"].diff()
        if "cumulative_return_vs_spy" in df.columns:
            df["relative_momentum_5d"] = df["cumulative_return_vs_spy"].diff(5)
            df["relative_momentum_20d"] = df["cumulative_return_vs_spy"].diff(20)
        if "volume_vs_spy" in df.columns:
            df["volume_ratio_change"] = df["volume_vs_spy"].pct_change()

        df = self._attach_news_features(df, raw_data.get("news_data", {}), prefix="sector_")
        return self._finalize(df)

    def _process_market_features(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        core_df = pd.DataFrame(raw_data.get("market_core_data", []))
        indicator_df = pd.DataFrame(raw_data.get("market_indicators", []))
        vix_df = pd.DataFrame(raw_data.get("vix_analysis", []))
        macro_df = pd.DataFrame(raw_data.get("macro_indicators", []))

        if core_df.empty:
            raise ValueError("Market core data is empty")

        core_df["Date"] = pd.to_datetime(core_df["Date"])
        df = core_df.copy()
        for extra_df in [indicator_df, vix_df, macro_df]:
            if not extra_df.empty and "Date" in extra_df.columns:
                extra_df["Date"] = pd.to_datetime(extra_df["Date"])
                df = df.merge(extra_df, on="Date", how="left")

        if "QQQ_Close" in df.columns and "Close" in df.columns:
            df["qqq_vs_spy"] = df["QQQ_Close"] / df["Close"].replace(0, np.nan) - 1
            df["qqq_vs_spy_change"] = df["qqq_vs_spy"].diff()
        if "VIX" in df.columns:
            df["vix_change"] = df["VIX"].pct_change()
        if "VIX_history_percentile" in df.columns:
            df["vix_extreme_high"] = (df["VIX_history_percentile"] > 90).astype(int)
            df["vix_extreme_low"] = (df["VIX_history_percentile"] < 10).astype(int)
        if "Dollar_Index" in df.columns:
            df["dollar_change"] = df["Dollar_Index"].pct_change()
        if "10Y_Treasury_Yield" in df.columns:
            df["treasury_yield_change"] = df["10Y_Treasury_Yield"].diff()

        df = self._add_generic_features(df)
        return self._finalize(df)

    def _merge_price_and_tech(self, price_df: pd.DataFrame, tech_df: pd.DataFrame) -> pd.DataFrame:
        price_df["Date"] = pd.to_datetime(price_df["Date"])
        if not tech_df.empty:
            tech_df["Date"] = pd.to_datetime(tech_df["Date"])
            df = price_df.merge(tech_df, on="Date", how="left")
        else:
            df = price_df.copy()
        return self._add_generic_features(df.sort_values("Date").reset_index(drop=True))

    def _add_generic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Close" not in df.columns:
            return df

        df["daily_return"] = df["Close"].pct_change()
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1).replace(0, np.nan))
        df["return_lag1"] = df["daily_return"].shift(1)
        df["return_lag5"] = df["daily_return"].shift(5)
        df["price_range"] = (df["High"] - df["Low"]) / df["Close"].replace(0, np.nan) if {"High", "Low"}.issubset(df.columns) else np.nan

        for lag in [5, 20, 60]:
            df[f"momentum_{lag}d"] = df["Close"] / df["Close"].shift(lag).replace(0, np.nan) - 1

        if "MA5" in df.columns:
            df["price_dev_ma5"] = (df["Close"] - df["MA5"]) / df["MA5"].replace(0, np.nan)
        if "MA20" in df.columns:
            df["price_dev_ma20"] = (df["Close"] - df["MA20"]) / df["MA20"].replace(0, np.nan)
        if "MA60" in df.columns:
            df["price_dev_ma60"] = (df["Close"] - df["MA60"]) / df["MA60"].replace(0, np.nan)
        if "RSI_14" in df.columns:
            df["rsi_change"] = df["RSI_14"].diff()
            df["rsi_overbought"] = (df["RSI_14"] > 70).astype(int)
            df["rsi_oversold"] = (df["RSI_14"] < 30).astype(int)
        if "Volume" in df.columns:
            df["volume_change"] = df["Volume"].pct_change()
            df["volume_ma5"] = df["Volume"].rolling(5).mean()
            df["volume_ma20"] = df["Volume"].rolling(20).mean()
            df["volume_ratio"] = df["Volume"] / df["volume_ma20"].replace(0, np.nan)

        df["day_of_week"] = df["Date"].dt.dayofweek
        df["month"] = df["Date"].dt.month
        df["quarter"] = df["Date"].dt.quarter
        df["is_month_end"] = df["Date"].dt.is_month_end.astype(int)
        return df

    def _attach_news_features(self, df: pd.DataFrame, news_data: Dict[str, Any], prefix: str) -> pd.DataFrame:
        traditional = news_data.get("traditional_analysis", {})
        sentiment = traditional.get("sentiment_analysis", {})
        llm = news_data.get("llm_analysis") or {}
        llm_sentiment = llm.get("sentiment_analysis", {})
        llm_risk = llm.get("risk_assessment", {})

        df[f"{prefix}news_sentiment_mean"] = float(sentiment.get("sentiment_mean", 0))
        df[f"{prefix}news_negative_ratio"] = float(sentiment.get("negative_ratio", 0))
        df[f"{prefix}llm_sentiment_score"] = float(llm_sentiment.get("sentiment_score", 0))
        df[f"{prefix}llm_risk_level"] = self._map_text_level(llm_risk.get("risk_level"))
        df[f"{prefix}llm_market_impact"] = self._map_text_level(llm_risk.get("market_impact"))
        df[f"{prefix}has_news_data"] = 1 if news_data and "error" not in news_data else 0
        return df

    def _map_text_level(self, value: Any) -> float:
        mapping = {"low": 0.2, "medium": 0.6, "high": 1.0}
        return float(mapping.get(value, 0))

    def _finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        df = df.dropna().reset_index(drop=True)
        return df
