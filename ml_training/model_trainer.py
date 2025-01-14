#!/usr/bin/env python3
# ml_training/model_trainer.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import json
from datetime import datetime
import sys
import os

# Add parent directory to path to import feature engineering
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_engineering.feature_pipeline import FeatureEngineering

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudModelTrainer:
    def __init__(self, model_dir: str = 'models'):
        self.model_dir = model_dir
        self.feature_pipeline = FeatureEngineering()
        os.makedirs(model_dir, exist_ok=True)
        
        self.models = {
            'random_forest': {
                'class': RandomForestClassifier,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'random_state': 42,
                    'class_weight': 'balanced',
                    'n_jobs': -1
                }
            },
            'gradient_boosting': {
                'class': GradientBoostingClassifier,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 5,
                    'learning_rate': 0.1,
                    'random_state': 42
                }
            }
        }

    def prepare_training_data(self, transactions: List[Dict[str, Any]], labels: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        features_list = self.feature_pipeline.process_batch(transactions)
        self.feature_pipeline.fit_scaler(features_list)
        
        X = np.vstack([
            self.feature_pipeline.transform_features(features)
            for features in features_list
        ])
        
        y = np.array(labels)
        
        return X, y

    def train_and_evaluate(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict[str, Dict[str, Any]]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        results = {}
        best_auc = 0
        best_model = None
        best_model_name = None
        
        for model_name, model_config in self.models.items():
            logger.info(f"Training {model_name}...")
            
            model = model_config['class'](**model_config['params'])
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            auc_score = roc_auc_score(y_test, y_prob)
            class_report = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            results[model_name] = {
                'auc_score': auc_score,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix.tolist(),
                'feature_importance': None
            }
            
            if hasattr(model, 'feature_importances_'):
                results[model_name]['feature_importance'] = {
                    feature: importance
                    for feature, importance in zip(
                        self.feature_pipeline.feature_columns,
                        model.feature_importances_
                    )
                }
            
            if auc_score > best_auc:
                best_auc = auc_score
                best_model = model
                best_model_name = model_name
                
            logger.info(f"{model_name} - AUC: {auc_score:.4f}")
        
        if best_model is not None:
            self.save_model(best_model, best_model_name)
            
        return results

    def save_model(self, model: Any, model_name: str):
        model_path = os.path.join(self.model_dir, f"{model_name}_model.pkl")
        joblib.dump(model, model_path)
        
        metadata = {
            'model_name': model_name,
            'feature_columns': self.feature_pipeline.feature_columns,
            'training_date': datetime.now().isoformat(),
            'model_params': self.models[model_name]['params']
        }
        
        metadata_path = os.path.join(self.model_dir, f"{model_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Saved {model_name} model and metadata")

    def load_best_model(self) -> Tuple[Any, Dict[str, Any]]:
        model_files = [f for f in os.listdir(self.model_dir) if f.endswith('_model.pkl')]
        if not model_files:
            raise FileNotFoundError("No trained models found")
            
        latest_model = max(model_files, key=lambda x: os.path.getmtime(
            os.path.join(self.model_dir, x)
        ))
        
        model_path = os.path.join(self.model_dir, latest_model)
        metadata_path = os.path.join(
            self.model_dir,
            latest_model.replace('_model.pkl', '_metadata.json')
        )
        
        model = joblib.load(model_path)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        return model, metadata

if __name__ == "__main__":
    # Generate sample data for demonstration
    np.random.seed(42)
    
    n_samples = 1000
    n_fraudulent = 50  # 5% fraud rate
    
    sample_transactions = []
    labels = []
    
    # Generate legitimate transactions
    for i in range(n_samples - n_fraudulent):
        tx = {
            'transaction_id': f'TX{i}',
            'timestamp': '2024-01-14T10:00:00',
            'account_id': f'ACC{i % 100}',
            'amount': np.random.lognormal(4, 1),
            'country': 'US',
            'merchant_category': np.random.randint(1000, 9999)
        }
        sample_transactions.append(tx)
        labels.append(0)  # Legitimate
        
    # Generate fraudulent transactions
    for i in range(n_fraudulent):
        tx = {
            'transaction_id': f'TX{n_samples - n_fraudulent + i}',
            'timestamp': '2024-01-14T10:00:00',
            'account_id': f'ACC{i % 10}',
            'amount': np.random.lognormal(6, 2),
            'country': np.random.choice(['US', 'FR', 'GB']),
            'merchant_category': np.random.randint(1000, 9999)
        }
        sample_transactions.append(tx)
        labels.append(1)  # Fraudulent
    
    trainer = FraudModelTrainer()
    X, y = trainer.prepare_training_data(sample_transactions, labels)
    results = trainer.train_and_evaluate(X, y)
    
    for model_name, metrics in results.items():
        print(f"\nResults for {model_name}:")
        print(f"AUC Score: {metrics['auc_score']:.4f}")
        print("\nClassification Report:")
        print(json.dumps(metrics['classification_report'], indent=2))