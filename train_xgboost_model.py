import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt


df = pd.read_csv("sector_ml_dataset.csv")

print("数据形状:", df.shape)



drop_cols = ["Date","Ticker","Risk_Label","Risk_Threshold"]

features = df.drop(columns=drop_cols)
target = df["Risk_Label"]

X = features.values
y = target.values

print("Feature数量:", X.shape[1])


split = int(len(df)*0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

print("Train size:",len(X_train))
print("Test size:",len(X_test))

model = xgb.XGBClassifier(

    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train,y_train)

pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:,1]


print("\nClassification Report")
print(classification_report(y_test,pred))

auc = roc_auc_score(y_test,proba)

print("\nAUC:",auc)

importance = model.feature_importances_

feature_names = features.columns

imp_df = pd.DataFrame({
    "feature":feature_names,
    "importance":importance
})

imp_df = imp_df.sort_values(by="importance",ascending=False)

print("\nTop Features:")
print(imp_df.head(15))

plt.figure(figsize=(8,6))
plt.barh(
    imp_df["feature"].head(15),
    imp_df["importance"].head(15)
)

plt.gca().invert_yaxis()
plt.title("Top 15 Feature Importance")
plt.show()