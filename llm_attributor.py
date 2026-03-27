import os
from typing import Any, Dict, List

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


class LLMAttributor:
    def __init__(self):
        self.api_key = os.getenv("QWEN_API_KEY")
        self.model = "qwen-max"
        self.client = None

        if self.api_key and OpenAI is not None:
            try:
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                )
            except Exception:
                self.client = None

    def is_available(self) -> bool:
        return True

    def has_remote_client(self) -> bool:
        return self.client is not None

    def generate_attribution(
        self,
        entity_info: Dict[str, Any],
        prediction_result: Dict[str, Any],
        feature_importance: List[Dict[str, Any]],
        raw_data_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self.client is not None:
            try:
                return self._generate_with_llm(entity_info, prediction_result, feature_importance, raw_data_summary)
            except Exception:
                pass
        return self._generate_rule_based(entity_info, prediction_result, feature_importance, raw_data_summary)

    def _generate_with_llm(
        self,
        entity_info: Dict[str, Any],
        prediction_result: Dict[str, Any],
        feature_importance: List[Dict[str, Any]],
        raw_data_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        prompt = (
            "Based on the following financial risk result, return a JSON object with "
            "executive_summary, risk_level_assessment, key_risk_drivers, market_context, "
            "mitigation_suggestions, and outlook."
            f"\nEntity: {entity_info}\nPrediction: {prediction_result}\nFeatures: {feature_importance[:5]}\nSummary: {raw_data_summary}"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a rigorous financial risk analyst. Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1000,
        )
        content = response.choices[0].message.content.strip()
        import json
        import re

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", content)
            if not match:
                raise
            parsed = json.loads(match.group())

        return {
            "success": True,
            "attribution": parsed,
            "model_used": self.model,
            "usage": {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            },
        }

    def _generate_rule_based(
        self,
        entity_info: Dict[str, Any],
        prediction_result: Dict[str, Any],
        feature_importance: List[Dict[str, Any]],
        raw_data_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        top_features = feature_importance[:3]
        risk_level = prediction_result.get("latest_risk_level", "medium")
        probability = prediction_result.get("latest_risk_probability", 0)
        sentiment = raw_data_summary.get("sentiment", "neutral")

        drivers = []
        for item in top_features:
            importance = float(item.get("importance", 0))
            impact = "high" if importance >= 0.12 else "medium" if importance >= 0.06 else "low"
            drivers.append(
                {
                    "factor": item.get("feature", "unknown_feature"),
                    "impact": impact,
                    "explanation": f"This feature has importance {importance:.3f} and contributes meaningfully to short-term risk detection.",
                }
            )

        summary = (
            f"{entity_info.get('name', 'The asset')} is currently rated {risk_level} with a risk probability of "
            f"{probability}%. The signal appears to be driven by trend, volatility, and sentiment together."
        )

        attribution = {
            "executive_summary": summary,
            "risk_level_assessment": {
                "current_level": risk_level,
                "probability": probability,
                "assessment": "This result is suitable for demo and analysis, not direct live trading advice.",
            },
            "key_risk_drivers": drivers,
            "market_context": f"The app prefers live data and falls back to local demo data when needed. Current sentiment is {sentiment}.",
            "mitigation_suggestions": [
                "Reduce single-asset exposure and enforce position sizing.",
                "Track moving averages, volatility, and volume for follow-up confirmation.",
                "Use a paid or validated data provider for production-grade decisions.",
            ],
            "outlook": "If volatility cools and sentiment improves, the model score should ease. If drawdowns continue on strong volume, risk may rise further.",
        }

        return {
            "success": True,
            "attribution": attribution,
            "model_used": "rule_based_fallback",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
