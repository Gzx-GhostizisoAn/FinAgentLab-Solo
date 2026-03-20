from flask import Flask, render_template, request, jsonify
from data_collector import pull_standard_data, SECTOR_STANDARD
from feature_engineering import FeatureEngineer
from model_trainer import RiskModelTrainer
from llm_attributor import LLMAttributor  
import asyncio
import os

app = Flask(__name__)
feature_engineer = FeatureEngineer()
model_trainer = RiskModelTrainer()


try:
    llm_attributor = LLMAttributor()
except Exception as e:
    print(f"LLM归因器初始化失败：{e}")
    llm_attributor = None


@app.route('/')
def index():
    return render_template('index.html', sector_list=SECTOR_STANDARD)


@app.route('/api/pull_data', methods=['POST'])
def pull_data_api():
    try:
        params = request.json
        entity_type = params['entity_type']
        entity_code = params.get('entity_code', '')
        time_horizon = params['time_horizon']
        pull_news = params.get('pull_news', False)

   
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        raw_data = loop.run_until_complete(
            pull_standard_data(entity_type, entity_code, time_horizon, pull_news)
        )

   
        feature_result = feature_engineer.process_features(raw_data)

        return jsonify({
            "success": True,
            "raw_data": raw_data,
            "feature_result": feature_result
        })

    except Exception as e:
        return jsonify({"success": False, "error": f"服务器错误：{str(e)}"}), 500


@app.route('/api/train_model', methods=['POST'])
def train_model_api():
    try:
        params = request.json
        entity_type = params['entity_type']
        entity_code = params.get('entity_code', '')
        time_horizon = params['time_horizon']
        model_name = params.get('model_name', 'risk_model')

   
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        raw_data = loop.run_until_complete(
            pull_standard_data(entity_type, entity_code, time_horizon, pull_news=False)
        )
        feature_result = feature_engineer.process_features(raw_data)
        
        if not feature_result.get("success"):
            return jsonify({"success": False, "error": "特征工程失败"})
        
     
        features_df = feature_result["full_features"]
        
    
        df_with_label = model_trainer.create_risk_label(features_df, entity_type)
        
     
        X, y, feature_cols = model_trainer.prepare_training_data(df_with_label)
        
      
        train_result = model_trainer.train_model(X, y, feature_cols)
        
      
        save_result = model_trainer.save_model(model_name)
        
        return jsonify({
            "success": True,
            "train_result": train_result,
            "save_result": save_result
        })

    except Exception as e:
        return jsonify({"success": False, "error": f"模型训练失败：{str(e)}"}), 500


@app.route('/api/predict_risk', methods=['POST'])
def predict_risk_api():
    try:
        params = request.json
        entity_type = params['entity_type']
        entity_code = params.get('entity_code', '')
        time_horizon = params['time_horizon']
        model_name = params.get('model_name', 'risk_model')

    
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        raw_data = loop.run_until_complete(
            pull_standard_data(entity_type, entity_code, time_horizon, pull_news=False)
        )
        feature_result = feature_engineer.process_features(raw_data)
        
        if not feature_result.get("success"):
            return jsonify({"success": False, "error": "特征工程失败"})
        
       
        features_df = feature_result["full_features"]
        
    
        load_result = model_trainer.load_model(model_name)
        if not load_result.get("success"):
            return jsonify({"success": False, "error": "模型加载失败，请先训练模型"})
        
        predict_result = model_trainer.predict_risk(features_df)
        
      
        global last_prediction_data
        last_prediction_data = {
            "entity_info": {
                "name": raw_data.get("entity_name"),
                "type": entity_type,
                "code": raw_data.get("entity_code")
            },
            "prediction_result": predict_result,
            "feature_importance": predict_result.get("feature_importance", []),
            "raw_data_summary": {
                "data_count": raw_data.get("data_count", 0),
                "latest_close": raw_data.get("price_data", [{}])[-1].get("Close", "未知") if raw_data.get("price_data") else "未知",
                "latest_volatility": "未知",
                "sentiment": "未知",
                "date_range": f"{raw_data.get('time_horizon', '未知')}"
            }
        }
        
        return jsonify({
            "success": True,
            "entity_name": raw_data.get("entity_name"),
            "prediction_result": predict_result
        })

    except Exception as e:
        return jsonify({"success": False, "error": f"预测失败：{str(e)}"}), 500


@app.route('/api/generate_attribution', methods=['POST'])
def generate_attribution_api():
    try:
        if not llm_attributor:
            return jsonify({
                "success": False, 
                "error": "LLM归因器未初始化，请检查环境变量 QWEN_API_KEY 是否设置"
            }), 500
        
      
        global last_prediction_data
        if not last_prediction_data:
            return jsonify({
                "success": False, 
                "error": "请先进行风险预测"
            }), 400
        
     
        attribution_result = llm_attributor.generate_attribution(
            last_prediction_data["entity_info"],
            last_prediction_data["prediction_result"],
            last_prediction_data["feature_importance"],
            last_prediction_data["raw_data_summary"]
        )
        
        return jsonify(attribution_result)

    except Exception as e:
        return jsonify({"success": False, "error": f"LLM归因失败：{str(e)}"}), 500

last_prediction_data = None

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)