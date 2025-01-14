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

## Usage

1. **Start the Feature Engineering Pipeline**:
```bash
python feature_engineering/feature_pipeline.py
```

2. **Train the ML Model**:
```bash
python ml_training/model_trainer.py
```

3. **Start the Scoring Service**:
```bash
python scoring_service/predictor.py
```

4. **Start the Model Monitor**:
```bash
python monitoring/model_monitor.py
```

5. **Send Test Transactions**:
```python
import requests

transaction = {
    "transaction_id": "TX123",
    "timestamp": "2024-01-14T10:00:00",
    "account_id": "ACC456",
    "amount": 1000.0,
    "country": "US",
    "merchant_category": 5411
}

response = requests.post("http://localhost:8000/predict", json=transaction)
print(response.json())
```

## Configuration

The system can be configured through various configuration files:

- `config/model_config.yaml`: ML model parameters and training settings
- `config/feature_config.yaml`: Feature engineering parameters
- `config/monitoring_config.yaml`: Monitoring thresholds and settings
- `docker-compose.yaml`: Service configuration

## Monitoring

Access monitoring dashboards:

- Prometheus metrics: `http://localhost:9090`
- Model monitoring dashboard: `http://localhost:8001`

## Development

1. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

2. Run tests:
```bash
pytest tests/
```

3. Run linting:
```bash
flake8 .
```

## Project Structure

```
real-time-fraud-detection/
├── data_ingestion/
│   └── kafka_producer.py
├── feature_engineering/
│   └── feature_pipeline.py
├── ml_training/
│   ├── model_trainer.py
│   └── config/
│       └── model_config.yaml
├── scoring_service/
│   └── predictor.py
├── monitoring/
│   └── model_monitor.py
├── config/
│   └── prometheus.yml
├── docker-compose.yaml
├── requirements.txt
└── README.md
```

## Dependencies

- Python 3.8+
- Apache Kafka
- Redis
- Prometheus
- FastAPI
- Scikit-learn
- Pandas
- NumPy

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.