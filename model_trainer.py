import os
from typing import Any, Dict, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, classification_report, precision_recall_curve, roc_auc_score


class RiskModelTrainer:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.model = None
        self.feature_cols = None
        self.label_col = "risk_label"
        os.makedirs(model_dir, exist_ok=True)

    def create_risk_label(self, features_df: pd.DataFrame, entity_type: str) -> pd.DataFrame:
        df = features_df.copy()
        if entity_type == "listed_company":
            df["future_return_5d"] = df["Close"].shift(-5) / df["Close"] - 1
            df[self.label_col] = (df["future_return_5d"] < -0.04).astype(int)
        elif entity_type == "sector":
            if "cumulative_return_vs_spy" in df.columns:
                df["future_relative_return_20d"] = df["cumulative_return_vs_spy"].shift(-20) - df["cumulative_return_vs_spy"]
                df[self.label_col] = (df["future_relative_return_20d"] < -0.03).astype(int)
            else:
                df["future_return_20d"] = df["Close"].shift(-20) / df["Close"] - 1
                df[self.label_col] = (df["future_return_20d"] < -0.035).astype(int)
        else:
            df["future_return_10d"] = df["Close"].shift(-10) / df["Close"] - 1
            df[self.label_col] = (df["future_return_10d"] < -0.025).astype(int)

        return df.dropna().reset_index(drop=True)

    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, list]:
        exclude_cols = {
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "future_return_5d",
            "future_return_10d",
            "future_return_20d",
            "future_relative_return_20d",
            self.label_col,
        }
        feature_cols = [
            col
            for col in df.columns
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
        ]

        clean_df = df[feature_cols + [self.label_col]].dropna().reset_index(drop=True)
        return clean_df[feature_cols], clean_df[self.label_col], feature_cols

    def train_model(self, X: pd.DataFrame, y: pd.Series, feature_cols: list) -> Dict[str, Any]:
        if len(X) < 40:
            return {"success": False, "error": "Need at least 40 samples for training"}

        split_idx = max(int(len(X) * 0.8), 1)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        if y_train.nunique() < 2:
            return {"success": False, "error": "Training labels contain only one class"}
        if len(X_test) == 0 or y_test.nunique() < 2:
            return {"success": False, "error": "Test set is too small for evaluation"}

        self.model = RandomForestClassifier(
            n_estimators=240,
            max_depth=6,
            min_samples_leaf=3,
            random_state=42,
            class_weight="balanced_subsample",
        )
        self.model.fit(X_train, y_train)
        self.feature_cols = feature_cols

        y_pred_prob = self.model.predict_proba(X_test)[:, 1]
        y_pred = self.model.predict(X_test)

        roc_auc = roc_auc_score(y_test, y_pred_prob)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
        pr_auc = auc(recall, precision)

        feature_importance = (
            pd.DataFrame({"feature": feature_cols, "importance": self.model.feature_importances_})
            .sort_values("importance", ascending=False)
            .head(15)
        )

        return {
            "success": True,
            "metrics": {
                "roc_auc": round(float(roc_auc), 4),
                "pr_auc": round(float(pr_auc), 4),
                "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
            },
            "feature_importance": feature_importance.to_dict("records"),
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
        }

    def save_model(self, model_name: str = "risk_model") -> Dict[str, Any]:
        if self.model is None:
            return {"success": False, "error": "No trained model to save"}

        model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
        feature_path = os.path.join(self.model_dir, f"{model_name}_features.pkl")
        joblib.dump(self.model, model_path)
        joblib.dump(self.feature_cols, feature_path)
        return {"success": True, "model_path": model_path, "feature_path": feature_path}

    def load_model(self, model_name: str = "risk_model") -> Dict[str, Any]:
        model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
        feature_path = os.path.join(self.model_dir, f"{model_name}_features.pkl")
        if not os.path.exists(model_path) or not os.path.exists(feature_path):
            return {"success": False, "error": "Model files not found"}

        self.model = joblib.load(model_path)
        self.feature_cols = joblib.load(feature_path)
        return {"success": True}

    def predict_risk(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        if self.model is None:
            load_result = self.load_model()
            if not load_result.get("success"):
                return {"success": False, "error": "No available model, train first"}

        missing_cols = [col for col in self.feature_cols if col not in features_df.columns]
        if missing_cols:
            return {"success": False, "error": f"Missing feature columns: {missing_cols}"}

        X_pred = features_df[self.feature_cols].fillna(0)
        risk_prob = self.model.predict_proba(X_pred)[:, 1]
        risk_pred = self.model.predict(X_pred)
        latest_risk_prob = float(risk_prob[-1])

        if latest_risk_prob < 0.3:
            risk_level = "low"
        elif latest_risk_prob < 0.55:
            risk_level = "medium"
        elif latest_risk_prob < 0.8:
            risk_level = "high"
        else:
            risk_level = "extreme"

        feature_importance = (
            pd.DataFrame({"feature": self.feature_cols, "importance": self.model.feature_importances_})
            .sort_values("importance", ascending=False)
            .head(10)
        )

        return {
            "success": True,
            "latest_risk_probability": round(latest_risk_prob * 100, 2),
            "latest_risk_level": risk_level,
            "latest_risk_prediction": int(risk_pred[-1]),
            "risk_probability_history": [round(float(item) * 100, 2) for item in risk_prob.tolist()],
            "feature_importance": feature_importance.to_dict("records"),
        }
