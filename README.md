# FinAgentLab Solo

## Overview

This project is an open-source financial risk prediction and attribution system.
It is now designed around a **bring-your-own-data-provider-key** workflow.

The application supports:

1. data source configuration
2. live market data collection
3. feature engineering
4. model training
5. risk prediction
6. Qwen-based attribution

## Supported Commercial Data Providers

### 1. Twelve Data

- configure with `TWELVE_DATA_API_KEY`
- suitable for stock and ETF time series
- good for unified market data access

### 2. EODHD

- configure with `EODHD_API_KEY`
- suitable for end-of-day data, fundamentals, and news
- useful when you want structured provider news in the workflow

## Configuration Flow

The system now requires users to:

1. open the data provider configuration section
2. choose a provider
3. input their own API key
4. save the configuration
5. run the remaining workflow

Without a configured provider, pull/train/predict actions are disabled in the UI and blocked in the backend.

## Important Design Decision

This repository does **not** bundle or redistribute paid data.

- users provide their own provider credentials
- the app only acts as an orchestration and analysis layer
- the repository owner does not need to operate a central data service

This keeps the project more realistic and more sustainable for open-source use.

## Current Architecture

### Data layer

- provider switch layer in `data_collector.py`
- runtime provider configuration API in `app.py`
- provider-specific normalization into one common internal schema

### Model layer

- unified feature pipeline in `feature_engineering.py`
- random-forest risk model in `model_trainer.py`

### Attribution layer

- Qwen-compatible attribution in `llm_attributor.py`
- attribution requires `QWEN_API_KEY`

## Run

```bash
pip install -r requirements.txt
python app.py
```

Open:

```text
http://localhost:5000
```

## Environment Variables

Optional startup defaults:

```text
TWELVE_DATA_API_KEY=your_key
EODHD_API_KEY=your_key
QWEN_API_KEY=your_key
```

If neither data-source key is present at startup, users can still enter one from the web configuration page.

## Known Limitations

- provider coverage differs by market and endpoint
- some market combinations may be more stable on one provider than another
- Twelve Data support in this version focuses on time-series/profile access
- EODHD support in this version is better for fundamentals and news
- provider quotas, throttling, or network failures will surface as explicit errors

## Future Improvements

### Multi-provider fallback

Add automatic provider failover:

- primary provider request
- health check
- fallback provider request
- cached last-successful snapshot

### Richer provider adapters

Extend normalization for:

- more fundamentals fields
- broader market indices
- macro series
- sector and ETF metadata

### LangChain orchestration

LangChain is a good next step for:

- provider/tool routing
- multi-source news summarization
- structured extraction from provider news
- attribution workflow orchestration

It is not a replacement for the underlying market data provider itself.
