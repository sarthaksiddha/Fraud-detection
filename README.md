# Real-Time Fraud Detection Pipeline

A comprehensive real-time transaction monitoring and fraud detection system using streaming analytics, machine learning, and real-time alerting.

## Project Overview

This project implements an end-to-end fraud detection pipeline that can process financial transactions in real-time and identify potentially fraudulent activities. The system uses advanced machine learning techniques combined with real-time streaming data processing to provide immediate fraud detection capabilities.

### Key Features

- Real-time transaction processing using Apache Kafka
- Advanced feature engineering for fraud detection
- Machine learning models for fraud prediction
- Real-time scoring service with REST API
- Model monitoring and drift detection
- Performance metrics and alerting system
- Redis-based transaction history caching
- Prometheus metrics integration

## System Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Transaction │    │   Feature    │    │     ML       │
│   Stream     │───▶│  Engineering │───▶│   Prediction │
└──────────────┘    └──────────────┘    └──────────────┘
        │                                       │
        │                                       │
        ▼                                       ▼
┌──────────────┐                        ┌──────────────┐
│    Redis     │                        │   Alerts &   │
│    Cache     │                        │  Monitoring  │
└──────────────┘                        └──────────────┘
```

## Components

1. **Data Ingestion** (`data_ingestion/`)
   - Kafka producer for transaction streaming
   - Transaction data validation and preprocessing

2. **Feature Engineering** (`feature_engineering/`)
   - Real-time feature calculation
   - Historical feature aggregation
   - Feature scaling and transformation

3. **ML Training** (`ml_training/`)
   - Model training pipeline
   - Multiple model support (Random Forest, Gradient Boosting)
   - Model evaluation and selection
   - Feature importance analysis

4. **Scoring Service** (`scoring_service/`)
   - FastAPI-based REST API
   - Real-time prediction serving
   - Transaction history management
   - Response caching

5. **Monitoring** (`monitoring/`)
   - Model performance monitoring
   - Feature drift detection
   - Prometheus metrics
   - Alert generation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sarthaksiddha/Fraud-detection.git
cd Fraud-detection
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Install and start required services:
```bash
# Start Kafka
docker-compose up -d kafka

# Start Redis
docker-compose up -d redis

# Start Prometheus
docker-compose up -d prometheus
```

[Rest of README content as before...]