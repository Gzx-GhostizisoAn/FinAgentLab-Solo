import asyncio
import logging
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from data_collector import AVAILABLE_PROVIDERS, SECTOR_STANDARD, configure_data_source, get_data_source_status, pull_standard_data
from feature_engineering import FeatureEngineer
from llm_attributor import LLMAttributor
from model_trainer import RiskModelTrainer


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

feature_engineer = FeatureEngineer()
model_trainer = RiskModelTrainer()
llm_attributor = LLMAttributor()
last_prediction_data: Optional[Dict[str, Any]] = None


def _json_safe_feature_result(feature_result: Dict[str, Any]) -> Dict[str, Any]:
    if not feature_result:
        return {}

    safe_result = dict(feature_result)
    safe_result.pop("full_features", None)
    return safe_result


def run_async_task(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def _validate_params(params: Optional[Dict[str, Any]]) -> Optional[str]:
    if not params:
        return "Request body is empty"
    if not params.get("entity_type"):
        return "Missing entity_type"
    if not params.get("time_horizon"):
        return "Missing time_horizon"
    if params.get("entity_type") == "listed_company" and not params.get("entity_code"):
        return "Stock mode requires entity_code"
    return None


def _latest_close(raw_data: Dict[str, Any]) -> Any:
    series = raw_data.get("price_data") or raw_data.get("market_core_data") or []
    return series[-1].get("Close", "unknown") if series else "unknown"


def _news_summary(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    summary = {"sentiment": "unknown", "llm_risk_level": "unknown", "market_impact": "unknown"}
    news_data = raw_data.get("news_data") or {}
    llm_analysis = news_data.get("llm_analysis") or {}
    sentiment = llm_analysis.get("sentiment_analysis", {})
    risk = llm_analysis.get("risk_assessment", {})

    if sentiment:
        mapping = {"positive": "positive", "negative": "negative", "neutral": "neutral"}
        summary["sentiment"] = mapping.get(sentiment.get("overall_sentiment"), sentiment.get("overall_sentiment", "unknown"))
    if risk:
        summary["llm_risk_level"] = risk.get("risk_level", "unknown")
        summary["market_impact"] = risk.get("market_impact", "unknown")
    return summary


@app.route("/")
def index():
    return render_template(
        "index.html",
        sector_list=SECTOR_STANDARD,
        provider_list=AVAILABLE_PROVIDERS,
        data_source_status=get_data_source_status(),
    )


@app.route("/api/config/data_source", methods=["GET"])
def data_source_status_api():
    return jsonify({"success": True, "status": get_data_source_status()})


@app.route("/api/config/data_source", methods=["POST"])
def configure_data_source_api():
    params = request.get_json(silent=True) or {}
    result = configure_data_source(params.get("provider", ""), params.get("api_key", ""))
    status_code = 200 if result.get("success") else 400
    payload = dict(result)
    payload["status"] = get_data_source_status()
    return jsonify(payload), status_code


@app.route("/api/pull_data", methods=["POST"])
def pull_data_api():
    try:
        params = request.get_json(silent=True)
        error = _validate_params(params)
        if error:
            return jsonify({"success": False, "error": error}), 400

        raw_data = run_async_task(
            pull_standard_data(
                params["entity_type"],
                params.get("entity_code", ""),
                params["time_horizon"],
                bool(params.get("pull_news", False)),
            )
        )
        if not raw_data.get("success"):
            return jsonify(raw_data), 500

        feature_result = feature_engineer.process_features(raw_data)
        return jsonify(
            {
                "success": True,
                "raw_data": raw_data,
                "feature_result": _json_safe_feature_result(feature_result),
            }
        )
    except Exception as exc:
        logger.exception("pull_data failed")
        return jsonify({"success": False, "error": f"Server error: {exc}"}), 500


@app.route("/api/train_model", methods=["POST"])
def train_model_api():
    try:
        params = request.get_json(silent=True)
        error = _validate_params(params)
        if error:
            return jsonify({"success": False, "error": error}), 400

        raw_data = run_async_task(
            pull_standard_data(params["entity_type"], params.get("entity_code", ""), params["time_horizon"], False)
        )
        if not raw_data.get("success"):
            return jsonify(raw_data), 500

        feature_result = feature_engineer.process_features(raw_data)
        if not feature_result.get("success"):
            return jsonify(feature_result), 500

        features_df = feature_result["full_features"]
        labeled_df = model_trainer.create_risk_label(features_df, params["entity_type"])
        X, y, feature_cols = model_trainer.prepare_training_data(labeled_df)
        train_result = model_trainer.train_model(X, y, feature_cols)
        if not train_result.get("success"):
            return jsonify(train_result), 400

        save_result = model_trainer.save_model(params.get("model_name", "risk_model"))
        return jsonify({"success": True, "train_result": train_result, "save_result": save_result})
    except Exception as exc:
        logger.exception("train_model failed")
        return jsonify({"success": False, "error": f"Training failed: {exc}"}), 500


@app.route("/api/predict_risk", methods=["POST"])
def predict_risk_api():
    global last_prediction_data

    try:
        params = request.get_json(silent=True)
        error = _validate_params(params)
        if error:
            return jsonify({"success": False, "error": error}), 400

        raw_data = run_async_task(
            pull_standard_data(params["entity_type"], params.get("entity_code", ""), params["time_horizon"], True)
        )
        if not raw_data.get("success"):
            return jsonify(raw_data), 500

        feature_result = feature_engineer.process_features(raw_data)
        if not feature_result.get("success"):
            return jsonify(feature_result), 500

        load_result = model_trainer.load_model(params.get("model_name", "risk_model"))
        if not load_result.get("success"):
            return jsonify(load_result), 400

        predict_result = model_trainer.predict_risk(feature_result["full_features"])
        if not predict_result.get("success"):
            return jsonify(predict_result), 500

        news_summary = _news_summary(raw_data)
        last_prediction_data = {
            "entity_info": {
                "name": raw_data.get("entity_name", "unknown"),
                "type": params["entity_type"],
                "code": raw_data.get("entity_code", "unknown"),
            },
            "prediction_result": predict_result,
            "feature_importance": predict_result.get("feature_importance", []),
            "raw_data_summary": {
                "data_count": raw_data.get("data_count", 0),
                "latest_close": _latest_close(raw_data),
                "latest_volatility": "see features",
                "sentiment": news_summary["sentiment"],
                "llm_risk_level": news_summary["llm_risk_level"],
                "market_impact": news_summary["market_impact"],
                "date_range": params["time_horizon"],
                "data_source": raw_data.get("data_source", "unknown"),
            },
        }

        return jsonify({"success": True, "entity_name": raw_data.get("entity_name"), "prediction_result": predict_result})
    except Exception as exc:
        logger.exception("predict_risk failed")
        return jsonify({"success": False, "error": f"Prediction failed: {exc}"}), 500


@app.route("/api/generate_attribution", methods=["POST"])
def generate_attribution_api():
    try:
        if not last_prediction_data:
            return jsonify({"success": False, "error": "Run prediction first"}), 400

        if not llm_attributor.has_remote_client():
            return jsonify(
                {
                    "success": False,
                    "error": "Qwen API is not available. Please set QWEN_API_KEY before generating attribution.",
                }
            ), 400

        result = llm_attributor.generate_attribution(
            last_prediction_data["entity_info"],
            last_prediction_data["prediction_result"],
            last_prediction_data["feature_importance"],
            last_prediction_data["raw_data_summary"],
        )
        return jsonify(result)
    except Exception as exc:
        logger.exception("generate_attribution failed")
        return jsonify({"success": False, "error": f"Attribution failed: {exc}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
