#!/usr/bin/env python3
# feature_engineering/feature_pipeline.py

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import StandardScaler
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineering:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = [
            'amount',
            'hour_of_day',
            'is_weekend',
            'transaction_frequency_1h',
            'transaction_frequency_24h',
            'average_amount_24h',
            'amount_std_24h',
            'international_frequency_24h'
        ]

    def create_features(self, transaction: Dict[str, Any], historical_transactions: List[Dict[str, Any]] = None) -> Dict[str, float]:
        if historical_transactions is None:
            historical_transactions = []
            
        features = {
            'amount': transaction['amount'],
            'is_international': 1 if transaction['country'] != 'US' else 0
        }
        
        # Add temporal features
        timestamp = datetime.fromisoformat(transaction['timestamp'])
        features.update({
            'hour_of_day': timestamp.hour,
            'is_weekend': 1 if timestamp.weekday() >= 5 else 0
        })
        
        # Add historical features
        if historical_transactions:
            features.update(self._calculate_historical_features(transaction, historical_transactions))
        else:
            features.update({
                'transaction_frequency_1h': 0,
                'transaction_frequency_24h': 0,
                'average_amount_24h': 0,
                'amount_std_24h': 0,
                'international_frequency_24h': 0
            })
        
        return features

    def _calculate_historical_features(self, current_transaction: Dict[str, Any], historical_transactions: List[Dict[str, Any]]) -> Dict[str, float]:
        current_time = datetime.fromisoformat(current_transaction['timestamp'])
        
        # Calculate time windows
        one_hour_ago = current_time - timedelta(hours=1)
        one_day_ago = current_time - timedelta(hours=24)
        
        # Filter transactions by time windows
        transactions_1h = [
            tx for tx in historical_transactions 
            if datetime.fromisoformat(tx['timestamp']) >= one_hour_ago
        ]
        transactions_24h = [
            tx for tx in historical_transactions 
            if datetime.fromisoformat(tx['timestamp']) >= one_day_ago
        ]
        
        if not transactions_24h:
            return {
                'transaction_frequency_1h': 0,
                'transaction_frequency_24h': 0,
                'average_amount_24h': 0,
                'amount_std_24h': 0,
                'international_frequency_24h': 0
            }
        
        amounts_24h = [tx['amount'] for tx in transactions_24h]
        
        return {
            'transaction_frequency_1h': len(transactions_1h),
            'transaction_frequency_24h': len(transactions_24h),
            'average_amount_24h': np.mean(amounts_24h),
            'amount_std_24h': np.std(amounts_24h) if len(amounts_24h) > 1 else 0,
            'international_frequency_24h': sum(1 for tx in transactions_24h if tx['country'] != 'US') / len(transactions_24h)
        }

    def fit_scaler(self, features_list: List[Dict[str, float]]):
        df = pd.DataFrame(features_list)
        self.scaler.fit(df[self.feature_columns])
        joblib.dump(self.scaler, 'models/feature_scaler.pkl')

    def transform_features(self, features: Dict[str, float]) -> np.ndarray:
        df = pd.DataFrame([features])
        return self.scaler.transform(df[self.feature_columns])