import logging
import re
from typing import Any, Dict, List

import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


POSITIVE_WORDS = ["growth", "improve", "recover", "steady", "beat", "upgrade", "strong"]
NEGATIVE_WORDS = ["decline", "pressure", "volatility", "risk", "weak", "downgrade", "drawdown"]


def _score_text(text: str) -> float:
    text = text or ""
    positive_hits = sum(word in text for word in POSITIVE_WORDS)
    negative_hits = sum(word in text for word in NEGATIVE_WORDS)
    raw = positive_hits - negative_hits
    return max(-1.0, min(1.0, raw / 3))


def _traditional_analysis(news_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not news_list:
        return {"error": "No news data"}

    clean_texts = []
    for news in news_list:
        text = f"{news.get('title', '')} {news.get('content', '')}"
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            clean_texts.append(text)

    if not clean_texts:
        return {"error": "No usable text after cleaning"}

    sentiments = np.array([_score_text(text) for text in clean_texts], dtype=float)
    return {
        "news_count": len(clean_texts),
        "news_list": news_list[:10],
        "sentiment_analysis": {
            "sentiment_mean": round(float(sentiments.mean()), 4),
            "sentiment_volatility": round(float(sentiments.std()), 4),
            "positive_ratio": round(float((sentiments > 0.2).mean()), 4),
            "negative_ratio": round(float((sentiments < -0.2).mean()), 4),
            "neutral_ratio": round(float((np.abs(sentiments) <= 0.2).mean()), 4),
        },
    }


def _build_rule_based_llm(news_list: List[Dict[str, Any]], traditional: Dict[str, Any]) -> Dict[str, Any]:
    sentiment_mean = traditional.get("sentiment_analysis", {}).get("sentiment_mean", 0)
    negative_ratio = traditional.get("sentiment_analysis", {}).get("negative_ratio", 0)

    if negative_ratio >= 0.5:
        risk_level = "high"
    elif negative_ratio >= 0.25:
        risk_level = "medium"
    else:
        risk_level = "low"

    if sentiment_mean > 0.2:
        overall_sentiment = "positive"
    elif sentiment_mean < -0.2:
        overall_sentiment = "negative"
    else:
        overall_sentiment = "neutral"

    market_impact = "high" if len(news_list) >= 8 else "medium" if len(news_list) >= 4 else "low"

    return {
        "sentiment_analysis": {
            "overall_sentiment": overall_sentiment,
            "sentiment_score": round(float(sentiment_mean), 4),
            "confidence": 0.65,
            "sentiment_trend": "stable",
        },
        "topic_analysis": {
            "main_topics": ["earnings", "market volatility", "capital risk"],
            "risk_indicators": ["volatility", "soft sentiment", "short drawdown"],
            "market_impact": market_impact,
        },
        "risk_assessment": {
            "risk_level": risk_level,
            "risk_factors": ["short volatility", "sentiment impact"],
            "recommendations": ["manage position size", "watch follow-up trend signals"],
            "time_horizon": "short",
        },
    }


def process_news_data(news_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    traditional = _traditional_analysis(news_list)
    if "error" in traditional:
        return {
            "llm_analysis": None,
            "traditional_analysis": traditional,
            "analysis_method": "traditional",
            "news_count": len(news_list),
        }

    return {
        "llm_analysis": _build_rule_based_llm(news_list, traditional),
        "traditional_analysis": traditional,
        "analysis_method": "rule_based",
        "news_count": len(news_list),
    }
