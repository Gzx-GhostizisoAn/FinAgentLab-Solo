from flask import Flask, render_template, request, jsonify
from data_collector import pull_standard_data, SECTOR_STANDARD
from feature_engineering import FeatureEngineer
from model_trainer import RiskModelTrainer
from llm_attributor import LLMAttributor
import asyncio
import os
import logging
from typing import Optional, Dict, Any

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 全局变量 - 在最前面定义
last_prediction_data: Optional[Dict[str, Any]] = None

# 初始化模块
feature_engineer = FeatureEngineer()
model_trainer = RiskModelTrainer()

try:
    llm_attributor = LLMAttributor()
except Exception as e:
    logger.error(f"LLM归因器初始化失败：{e}")
    llm_attributor = None


def run_async_task(coro):
    """
    安全地运行异步任务
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)


@app.route('/')
def index():
    return render_template('index.html', sector_list=SECTOR_STANDARD)


@app.route('/api/pull_data', methods=['POST'])
def pull_data_api():
    try:
        params = request.json
        if not params:
            return jsonify({"success": False, "error": "请求体为空"}), 400
        
        entity_type = params.get('entity_type')
        entity_code = params.get('entity_code', '')
        time_horizon = params.get('time_horizon')
        pull_news = params.get('pull_news', False)
        
        if not entity_type or not time_horizon:
            return jsonify({"success": False, "error": "缺少必要参数：entity_type 或 time_horizon"}), 400
        
        # 拉取原始数据
        raw_data = run_async_task(
            pull_standard_data(entity_type, entity_code, time_horizon, pull_news)
        )
        
        if not raw_data.get("success"):
            return jsonify({"success": False, "error": raw_data.get("error", "数据拉取失败")}), 500
        
        # 特征工程
        feature_result = feature_engineer.process_features(raw_data)
        
        return jsonify({
            "success": True,
            "raw_data": raw_data,
            "feature_result": feature_result
        })

    except Exception as e:
        logger.error(f"拉取数据失败：{e}", exc_info=True)
        return jsonify({"success": False, "error": f"服务器错误：{str(e)}"}), 500


@app.route('/api/train_model', methods=['POST'])
def train_model_api():
    try:
        params = request.json
        if not params:
            return jsonify({"success": False, "error": "请求体为空"}), 400
        
        entity_type = params.get('entity_type')
        entity_code = params.get('entity_code', '')
        time_horizon = params.get('time_horizon')
        model_name = params.get('model_name', 'risk_model')
        
        if not entity_type or not time_horizon:
            return jsonify({"success": False, "error": "缺少必要参数"}), 400
        
        # 拉取原始数据
        raw_data = run_async_task(
            pull_standard_data(entity_type, entity_code, time_horizon, pull_news=False)
        )
        
        if not raw_data.get("success"):
            return jsonify({"success": False, "error": "数据拉取失败"}), 500
        
        # 特征工程
        feature_result = feature_engineer.process_features(raw_data)
        
        if not feature_result.get("success"):
            return jsonify({"success": False, "error": "特征工程失败"}), 500
        
        features_df = feature_result.get("full_features")
        if features_df is None or features_df.empty:
            return jsonify({"success": False, "error": "特征数据为空"}), 500
        
        # 创建标签
        df_with_label = model_trainer.create_risk_label(features_df, entity_type)
        
        # 准备训练数据
        X, y, feature_cols = model_trainer.prepare_training_data(df_with_label)
        
        if X.empty or y.empty:
            return jsonify({"success": False, "error": "训练数据为空"}), 500
        
        # 训练模型
        train_result = model_trainer.train_model(X, y, feature_cols)
        
        # 保存模型
        save_result = model_trainer.save_model(model_name)
        
        return jsonify({
            "success": True,
            "train_result": train_result,
            "save_result": save_result
        })

    except Exception as e:
        logger.error(f"模型训练失败：{e}", exc_info=True)
        return jsonify({"success": False, "error": f"模型训练失败：{str(e)}"}), 500


@app.route('/api/predict_risk', methods=['POST'])
def predict_risk_api():
    global last_prediction_data
    try:
        params = request.json
        if not params:
            return jsonify({"success": False, "error": "请求体为空"}), 400
        
        entity_type = params.get('entity_type')
        entity_code = params.get('entity_code', '')
        time_horizon = params.get('time_horizon')
        model_name = params.get('model_name', 'risk_model')
        
        if not entity_type or not time_horizon:
            return jsonify({"success": False, "error": "缺少必要参数"}), 400
        
        # 拉取原始数据
        raw_data = run_async_task(
            pull_standard_data(entity_type, entity_code, time_horizon, pull_news=False)
        )
        
        if not raw_data.get("success"):
            return jsonify({"success": False, "error": "数据拉取失败"}), 500
        
        # 特征工程
        feature_result = feature_engineer.process_features(raw_data)
        
        if not feature_result.get("success"):
            return jsonify({"success": False, "error": "特征工程失败"}), 500
        
        features_df = feature_result.get("full_features")
        if features_df is None or features_df.empty:
            return jsonify({"success": False, "error": "特征数据为空"}), 500
        
        # 加载模型
        load_result = model_trainer.load_model(model_name)
        if not load_result.get("success"):
            return jsonify({"success": False, "error": "模型加载失败，请先训练模型"}), 400
        
        # 预测
        predict_result = model_trainer.predict_risk(features_df)
        
        if not predict_result.get("success"):
            return jsonify({"success": False, "error": predict_result.get("error", "预测失败")}), 500
        
        # 保存预测数据用于后续归因
        price_data = raw_data.get("price_data", [{}])
        latest_close = price_data[-1].get("Close", "未知") if price_data else "未知"
        
        last_prediction_data = {
            "entity_info": {
                "name": raw_data.get("entity_name", "未知"),
                "type": entity_type,
                "code": raw_data.get("entity_code", "未知")
            },
            "prediction_result": predict_result,
            "feature_importance": predict_result.get("feature_importance", []),
            "raw_data_summary": {
                "data_count": raw_data.get("data_count", 0),
                "latest_close": latest_close,
                "latest_volatility": "未知",
                "sentiment": "未知",
                "date_range": time_horizon
            }
        }
        
        return jsonify({
            "success": True,
            "entity_name": raw_data.get("entity_name", "未知"),
            "prediction_result": predict_result
        })

    except Exception as e:
        logger.error(f"预测失败：{e}", exc_info=True)
        return jsonify({"success": False, "error": f"预测失败：{str(e)}"}), 500


@app.route('/api/generate_attribution', methods=['POST'])
def generate_attribution_api():
    try:
        if not llm_attributor:
            return jsonify({
                "success": False,
                "error": "LLM归因器未初始化，请检查环境变量 QWEN_API_KEY 是否设置"
            }), 500
        
        if not last_prediction_data:
            return jsonify({
                "success": False,
                "error": "请先进行风险预测"
            }), 400
        
        # 生成归因
        attribution_result = llm_attributor.generate_attribution(
            last_prediction_data["entity_info"],
            last_prediction_data["prediction_result"],
            last_prediction_data["feature_importance"],
            last_prediction_data["raw_data_summary"]
        )
        
        return jsonify(attribution_result)

    except Exception as e:
        logger.error(f"LLM归因失败：{e}", exc_info=True)
        return jsonify({"success": False, "error": f"LLM归因失败：{str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)