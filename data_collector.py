import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

from nlp_processor import process_news_data


SECTOR_STANDARD = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLI": "Industrials",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLC": "Communication Services",
}

PERIOD_MAP = {"short": 252, "medium": 756, "long": 1260}

AVAILABLE_PROVIDERS = {
    "twelve_data": {
        "label": "Twelve Data",
        "key_env": "TWELVE_DATA_API_KEY",
        "description": "Good unified API for stocks, ETFs, FX and crypto.",
    },
    "eodhd": {
        "label": "EODHD",
        "key_env": "EODHD_API_KEY",
        "description": "Good all-in-one API for end-of-day data, fundamentals and news.",
    },
}

DATA_SOURCE_CONFIG: Dict[str, Optional[str]] = {
    "provider": "twelve_data" if os.getenv("TWELVE_DATA_API_KEY") else ("eodhd" if os.getenv("EODHD_API_KEY") else None),
    "api_key": os.getenv("TWELVE_DATA_API_KEY") or os.getenv("EODHD_API_KEY"),
}


def configure_data_source(provider: str, api_key: str) -> Dict[str, Any]:
    provider = (provider or "").strip()
    api_key = (api_key or "").strip()

    if provider not in AVAILABLE_PROVIDERS:
        return {"success": False, "error": "Unsupported provider"}
    if not api_key:
        return {"success": False, "error": "API key is required"}

    DATA_SOURCE_CONFIG["provider"] = provider
    DATA_SOURCE_CONFIG["api_key"] = api_key
    return {"success": True, "provider": provider, "provider_label": AVAILABLE_PROVIDERS[provider]["label"]}


def get_data_source_status() -> Dict[str, Any]:
    provider = DATA_SOURCE_CONFIG.get("provider")
    return {
        "configured": bool(provider and DATA_SOURCE_CONFIG.get("api_key")),
        "provider": provider,
        "provider_label": AVAILABLE_PROVIDERS.get(provider, {}).get("label"),
        "providers": AVAILABLE_PROVIDERS,
    }


def _serialize_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    output = df.copy()
    if "Date" in output.columns:
        output["Date"] = pd.to_datetime(output["Date"]).dt.strftime("%Y-%m-%d")
    return output.to_dict("records")


def _compute_technical_indicators(price_df: pd.DataFrame) -> pd.DataFrame:
    tech_df = price_df[["Date", "Close"]].copy()
    tech_df["MA5"] = tech_df["Close"].rolling(5).mean()
    tech_df["MA20"] = tech_df["Close"].rolling(20).mean()
    tech_df["MA60"] = tech_df["Close"].rolling(60).mean()
    tech_df["daily_return"] = tech_df["Close"].pct_change()
    tech_df["volatility_20d"] = tech_df["daily_return"].rolling(20).std()

    delta = tech_df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    tech_df["RSI_14"] = 100 - (100 / (1 + rs))
    return tech_df.dropna().reset_index(drop=True)


def _http_get_json(base_url: str, params: Dict[str, Any]) -> Any:
    url = f"{base_url}?{urlencode(params)}"
    with urlopen(url, timeout=20) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def _normalize_price_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    if any(col not in df.columns for col in required):
        return pd.DataFrame()

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    return df[required].dropna().sort_values("Date").reset_index(drop=True)


def _twelve_fetch_time_series(symbol: str, outputsize: int) -> pd.DataFrame:
    data = _http_get_json(
        "https://api.twelvedata.com/time_series",
        {
            "symbol": symbol,
            "interval": "1day",
            "outputsize": outputsize,
            "apikey": DATA_SOURCE_CONFIG["api_key"],
            "format": "JSON",
        },
    )
    values = data.get("values") or []
    if not values:
        return pd.DataFrame()
    df = pd.DataFrame(values).rename(
        columns={
            "datetime": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    return _normalize_price_df(df)


def _twelve_fetch_profile(symbol: str) -> Dict[str, Any]:
    data = _http_get_json(
        "https://api.twelvedata.com/profile",
        {"symbol": symbol, "apikey": DATA_SOURCE_CONFIG["api_key"]},
    )
    if not isinstance(data, dict):
        return {}
    return data


def _eod_symbol(symbol: str) -> str:
    if "." in symbol:
        return symbol
    return f"{symbol}.US"


def _eod_fetch_eod(symbol: str, days: int) -> pd.DataFrame:
    start_date = (datetime.utcnow() - timedelta(days=int(days * 1.7))).strftime("%Y-%m-%d")
    data = _http_get_json(
        f"https://eodhd.com/api/eod/{_eod_symbol(symbol)}",
        {
            "api_token": DATA_SOURCE_CONFIG["api_key"],
            "fmt": "json",
            "period": "d",
            "from": start_date,
        },
    )
    if not isinstance(data, list) or not data:
        return pd.DataFrame()
    df = pd.DataFrame(data).rename(
        columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    return _normalize_price_df(df)


def _eod_fetch_fundamentals(symbol: str) -> Dict[str, Any]:
    data = _http_get_json(
        f"https://eodhd.com/api/fundamentals/{_eod_symbol(symbol)}",
        {"api_token": DATA_SOURCE_CONFIG["api_key"], "fmt": "json"},
    )
    if not isinstance(data, dict):
        return {}
    return data


def _eod_fetch_news(symbol: str) -> List[Dict[str, Any]]:
    data = _http_get_json(
        "https://eodhd.com/api/news",
        {"api_token": DATA_SOURCE_CONFIG["api_key"], "s": _eod_symbol(symbol), "limit": 10, "fmt": "json"},
    )
    if not isinstance(data, list):
        return []
    return [
        {
            "title": item.get("title", ""),
            "content": item.get("content", "") or item.get("title", ""),
            "date": item.get("date", ""),
            "source": item.get("source", ""),
            "url": item.get("link", ""),
        }
        for item in data
    ]


def _fetch_price_history(symbol: str, period_days: int) -> pd.DataFrame:
    provider = DATA_SOURCE_CONFIG["provider"]
    if provider == "twelve_data":
        return _twelve_fetch_time_series(symbol, period_days)
    if provider == "eodhd":
        return _eod_fetch_eod(symbol, period_days)
    return pd.DataFrame()


def _fetch_profile(symbol: str) -> Dict[str, Any]:
    provider = DATA_SOURCE_CONFIG["provider"]
    if provider == "twelve_data":
        return _twelve_fetch_profile(symbol)
    if provider == "eodhd":
        return _eod_fetch_fundamentals(symbol)
    return {}


def _fetch_news(symbol: str) -> List[Dict[str, Any]]:
    provider = DATA_SOURCE_CONFIG["provider"]
    if provider == "eodhd":
        return _eod_fetch_news(symbol)
    return []


def _extract_company_metadata(symbol: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    provider = DATA_SOURCE_CONFIG["provider"]
    if provider == "twelve_data":
        return {
            "company_name": profile.get("name", symbol),
            "sector": profile.get("sector", "Unknown"),
        }
    general = profile.get("General", {}) if isinstance(profile, dict) else {}
    highlights = profile.get("Highlights", {}) if isinstance(profile, dict) else {}
    return {
        "company_name": general.get("Name", symbol),
        "sector": general.get("Sector", "Unknown"),
        "market_capitalization": highlights.get("MarketCapitalization"),
        "ebitda": highlights.get("EBITDA"),
        "pe_ratio": highlights.get("PERatio"),
    }


def _config_guard() -> Optional[Dict[str, Any]]:
    if not DATA_SOURCE_CONFIG.get("provider") or not DATA_SOURCE_CONFIG.get("api_key"):
        return {"success": False, "error": "Data source is not configured"}
    return None


def _build_company_payload(entity_code: str, period_days: int, pull_news: bool) -> Dict[str, Any]:
    code = entity_code.upper()
    price_df = _fetch_price_history(code, period_days)
    if price_df.empty:
        return {"success": False, "error": f"Unable to fetch live price data for {code} from {DATA_SOURCE_CONFIG['provider']}"}

    tech_df = _compute_technical_indicators(price_df)
    profile = _fetch_profile(code)
    financial_data = _extract_company_metadata(code, profile)

    payload = {
        "success": True,
        "entity_name": financial_data.get("company_name", code),
        "entity_code": code,
        "data_source": DATA_SOURCE_CONFIG["provider"],
        "price_data": _serialize_records(price_df),
        "tech_indicators": _serialize_records(tech_df),
        "financial_data": financial_data,
        "data_count": int(len(price_df)),
    }

    if pull_news:
        news_list = _fetch_news(code)
        payload["news_data"] = process_news_data(news_list) if news_list else {"error": "Live news is unavailable for this provider"}
    return payload


def _build_sector_payload(entity_code: str, period_days: int, pull_news: bool) -> Dict[str, Any]:
    if entity_code not in SECTOR_STANDARD:
        return {"success": False, "error": "Unsupported sector code"}

    sector_df = _fetch_price_history(entity_code, period_days)
    bench_df = _fetch_price_history("SPY", period_days)
    if sector_df.empty:
        return {"success": False, "error": f"Unable to fetch live sector data for {entity_code} from {DATA_SOURCE_CONFIG['provider']}"}
    if bench_df.empty:
        return {"success": False, "error": f"Unable to fetch SPY benchmark data from {DATA_SOURCE_CONFIG['provider']}"}

    bench_df = bench_df[["Date", "Close", "Volume"]].rename(columns={"Close": "SPY_Close", "Volume": "SPY_Volume"})
    merged_df = sector_df.merge(bench_df, on="Date", how="left")
    merged_df["sector_daily_return"] = merged_df["Close"].pct_change()
    merged_df["spy_daily_return"] = merged_df["SPY_Close"].pct_change()
    merged_df["return_vs_spy"] = merged_df["sector_daily_return"] - merged_df["spy_daily_return"]
    merged_df["cumulative_return_vs_spy"] = merged_df["return_vs_spy"].fillna(0).cumsum()
    merged_df["volume_ma20"] = merged_df["Volume"].rolling(20).mean()
    merged_df["volume_change_rate"] = merged_df["Volume"].pct_change()
    merged_df["volume_vs_spy"] = merged_df["Volume"] / merged_df["SPY_Volume"].replace(0, np.nan)

    payload = {
        "success": True,
        "entity_name": SECTOR_STANDARD[entity_code],
        "entity_code": entity_code,
        "data_source": DATA_SOURCE_CONFIG["provider"],
        "price_data": _serialize_records(merged_df),
        "tech_indicators": _serialize_records(_compute_technical_indicators(sector_df)),
        "relative_strength": _serialize_records(merged_df[["Date", "return_vs_spy", "cumulative_return_vs_spy"]].dropna()),
        "volume_trend": _serialize_records(
            merged_df[["Date", "Volume", "volume_ma20", "volume_change_rate", "volume_vs_spy"]].dropna()
        ),
        "data_count": int(len(merged_df)),
    }

    if pull_news:
        news_list = _fetch_news(entity_code)
        payload["news_data"] = process_news_data(news_list) if news_list else {"error": "Live news is unavailable for this provider"}
    return payload


def _build_market_payload(period_days: int, pull_news: bool) -> Dict[str, Any]:
    spy_df = _fetch_price_history("SPY", period_days)
    qqq_df = _fetch_price_history("QQQ", period_days)
    if spy_df.empty:
        return {"success": False, "error": f"Unable to fetch SPY market data from {DATA_SOURCE_CONFIG['provider']}"}

    merged_df = spy_df.copy()
    if not qqq_df.empty:
        qqq_close = qqq_df[["Date", "Close"]].rename(columns={"Close": "QQQ_Close"})
        merged_df = merged_df.merge(qqq_close, on="Date", how="left")

    indicators = _compute_technical_indicators(spy_df)
    market_indicators = indicators.copy()
    market_indicators["advance_decline_line"] = np.where(
        market_indicators["daily_return"] > 0,
        1,
        -1,
    ).cumsum()

    payload = {
        "success": True,
        "entity_name": "Broad Market",
        "entity_code": "SPY+QQQ",
        "data_source": DATA_SOURCE_CONFIG["provider"],
        "market_core_data": _serialize_records(merged_df),
        "market_indicators": _serialize_records(market_indicators),
        "vix_analysis": [],
        "macro_indicators": [],
        "data_count": int(len(merged_df)),
    }

    if pull_news:
        news_list = _fetch_news("SPY")
        payload["news_data"] = process_news_data(news_list) if news_list else {"error": "Live news is unavailable for this provider"}
    return payload


async def pull_standard_data(entity_type: str, entity_code: str, time_horizon: str, pull_news: bool = False) -> Dict[str, Any]:
    config_error = _config_guard()
    if config_error:
        return config_error

    period_days = PERIOD_MAP.get(time_horizon, 252)
    await asyncio.sleep(0)

    try:
        if entity_type == "listed_company":
            result = _build_company_payload(entity_code, period_days, pull_news)
        elif entity_type == "sector":
            result = _build_sector_payload(entity_code, period_days, pull_news)
        elif entity_type == "market":
            result = _build_market_payload(period_days, pull_news)
        else:
            return {"success": False, "error": "Unsupported entity type"}
    except Exception as exc:
        return {"success": False, "error": f"Provider request failed: {exc}"}

    result["entity_type"] = entity_type
    result["time_horizon"] = time_horizon
    return result
