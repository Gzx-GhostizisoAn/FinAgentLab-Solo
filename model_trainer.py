import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve, auc

class RiskModelTrainer:
   
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.model = None
        self.feature_cols = None
        self.label_col = "risk_label"
        
      
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
   
    def create_risk_label(self, features_df: pd.DataFrame, entity_type: str) -> pd.DataFrame:
       
        df = features_df.copy()
        
       
        if entity_type == "listed_company":
          
            df["future_return_5d"] = df["Close"].pct_change(5).shift(-5)
            df[self.label_col] = (df["future_return_5d"] < -0.05).astype(int)
        elif entity_type == "sector":
            
            if "cumulative_return_vs_spy" in df.columns:
                df["future_relative_return_20d"] = df["cumulative_return_vs_spy"].diff(20).shift(-20)
                df[self.label_col] = (df["future_relative_return_20d"] < -0.03).astype(int)
            else:
                df["future_return_20d"] = df["Close"].pct_change(20).shift(-20)
                df[self.label_col] = (df["future_return_20d"] < -0.04).astype(int)
        else:  
           
            df["future_return_10d"] = df["Close"].pct_change(10).shift(-10)
            df[self.label_col] = (df["future_return_10d"] < -0.03).astype(int)
        
        
        df = df.dropna(subset=[self.label_col]).reset_index(drop=True)
        return df
    
   
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, list]:
      
        
        exclude_cols = [
            "Date", "Open", "High", "Low", "Close", "Volume",
            "future_return_5d", "future_return_20d", "future_return_10d",
            "future_relative_return_20d", "cumulative_return_vs_spy",
            self.label_col
        ]
        
        
        feature_cols = [
            col for col in df.columns 
            if col not in exclude_cols 
            and pd.api.types.is_numeric_dtype(df[col])
        ]
        
       
        df_clean = df[feature_cols + [self.label_col]].dropna().reset_index(drop=True)
        
        X = df_clean[feature_cols]
        y = df_clean[self.label_col]
        
        return X, y, feature_cols
    
   
    def train_model(self, X: pd.DataFrame, y: pd.Series, feature_cols: list) -> Dict[str, Any]:
       
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1
        
      
        self.model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric=["auc", "logloss"],
            scale_pos_weight=scale_pos_weight,
            learning_rate=0.05,
            max_depth=4,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=50
        )
        
        
        self.feature_cols = feature_cols
        
        
        y_pred_prob = self.model.predict_proba(X_test)[:, 1]
        y_pred = self.model.predict(X_test)
        
        
        roc_auc = roc_auc_score(y_test, y_pred_prob)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
        pr_auc = auc(recall, precision)
        
      
        feature_importance = pd.DataFrame({
            "feature": feature_cols,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False).head(15)
        
        return {
            "success": True,
            "model": self.model,
            "feature_cols": feature_cols,
            "metrics": {
                "roc_auc": round(roc_auc, 4),
                "pr_auc": round(pr_auc, 4),
                "classification_report": classification_report(y_test, y_pred, output_dict=True)
            },
            "feature_importance": feature_importance.to_dict("records"),
            "train_size": len(X_train),
            "test_size": len(X_test)
        }
    
   
    def save_model(self, model_name: str = "risk_model"):
      
        if self.model is None:
            return {"success": False, "error": "没有训练好的模型"}
        
        model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
        feature_path = os.path.join(self.model_dir, f"{model_name}_features.pkl")
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.feature_cols, feature_path)
        
        return {
            "success": True,
            "model_path": model_path,
            "feature_path": feature_path
        }
  
    def load_model(self, model_name: str = "risk_model"):
        
        model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
        feature_path = os.path.join(self.model_dir, f"{model_name}_features.pkl")
        
        if not os.path.exists(model_path) or not os.path.exists(feature_path):
            return {"success": False, "error": "模型文件不存在"}
        
        self.model = joblib.load(model_path)
        self.feature_cols = joblib.load(feature_path)
        
        return {"success": True}
  
    def predict_risk(self, features_df: pd.DataFrame) -> Dict[str, Any]:
       
        if self.model is None:
           
            load_result = self.load_model()
            if not load_result["success"]:
                return {"success": False, "error": "没有可用的模型，请先训练"}
        
   
        try:
       
            missing_cols = [col for col in self.feature_cols if col not in features_df.columns]
            if missing_cols:
                return {"success": False, "error": f"缺少特征列：{missing_cols}"}
            
          
            X_pred = features_df[self.feature_cols].fillna(0)
            
           
            risk_prob = self.model.predict_proba(X_pred)[:, 1]
            risk_pred = self.model.predict(X_pred)
            
         
            latest_risk_prob = float(risk_prob[-1])
            latest_risk_pred = int(risk_pred[-1])
            
           
            if latest_risk_prob < 0.3:
                risk_level = "低"
            elif latest_risk_prob < 0.6:
                risk_level = "中"
            elif latest_risk_prob < 0.85:
                risk_level = "高"
            else:
                risk_level = "极高"
            
          
            feature_importance = pd.DataFrame({
                "feature": self.feature_cols,
                "importance": self.model.feature_importances_
            }).sort_values("importance", ascending=False).head(10)
            
            return {
                "success": True,
                "latest_risk_probability": round(latest_risk_prob * 100, 2),
                "latest_risk_level": risk_level,
                "latest_risk_prediction": latest_risk_pred,
                "risk_probability_history": risk_prob.tolist(),
                "feature_importance": feature_importance.to_dict("records")
            }
        
        except Exception as e:
            return {"success": False, "error": f"预测失败：{str(e)}"}