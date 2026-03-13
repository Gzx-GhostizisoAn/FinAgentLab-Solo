# Financial Agent

## 项目描述

这是一个金融代理项目，使用机器学习技术对金融数据进行分析和预测。主要功能包括市场数据爬取、特征工程和XGBoost模型训练。

## 文件结构

```
financial_agent/
├── readme.md                 # 项目说明文档
├── app/                      # 应用程序代码目录
├── data/                     # 数据文件目录
│   ├── sector_ml_dataset.csv              # 部门机器学习数据集
│   └── sector_risk_data_spy_xle_xlk_xlf.csv  # 部门风险数据（SPY, XLE, XLK, XLF）
└── scripts/                  # 脚本目录
    ├── data_crawl_market.py  # 市场数据爬取脚本
    ├── feature_engineering.py # 特征工程脚本
    └── train_xgboost_model.py # XGBoost模型训练脚本
```

## 安装

1. 确保安装了Python 3.7或更高版本。
2. 克隆或下载项目到本地。
3. 安装依赖包：

   ```bash
   pip install -r requirements.txt
   ```

   如果没有requirements.txt文件，请手动安装以下包：
   - pandas
   - numpy
   - scikit-learn
   - xgboost
   - requests (用于数据爬取)

## 使用

### 数据爬取
运行市场数据爬取脚本：

```bash
python scripts/data_crawl_market.py
```

### 特征工程
运行特征工程脚本：

```bash
python scripts/feature_engineering.py
```

### 模型训练
运行XGBoost模型训练脚本：

```bash
python scripts/train_xgboost_model.py
```

## 数据说明

- `sector_ml_dataset.csv`：用于机器学习的部门数据集。
- `sector_risk_data_spy_xle_xlk_xlf.csv`：包含SPY、XLE、XLK、XLF等指数的风险数据。

## 贡献

欢迎提交问题和拉取请求。请确保代码符合项目的编码规范。

## 许可证

[请添加许可证信息]
