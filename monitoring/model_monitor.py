#!/usr/bin/env python3
# monitoring/model_monitor.py

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging
from datetime import datetime, timedelta
from kafka import KafkaConsumer
import json
import time
from scipy import stats
from prometheus_client import start_http_server, Gauge, Counter, Histogram
import threading
from collections import deque
import pickle
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelMonitor:
    def __init__(self, 
                 reference_data_path: str,
                 metrics_port: int = 8001,
                 kafka_bootstrap_servers: str = 'localhost:9092'):
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        
        # Load reference statistics
        with open(reference_data_path, 'rb') as f:
            self.reference_stats = pickle.load(f)
        
        # Initialize Prometheus metrics
        self.prediction_counter = Counter(
            'fraud_predictions_total',
            'Total number of fraud predictions'
        )
        self.fraud_ratio_gauge = Gauge(
            'fraud_ratio',
            'Ratio of fraudulent transactions'
        )
        self.prediction_latency = Histogram(
            'prediction_latency_seconds',
            'Prediction latency in seconds'
        )
        self.feature_drift_gauge = Gauge(
            'feature_drift',
            'Feature drift score',
            ['feature_name']
        )
        
        # Start Prometheus metrics server
        start_http_server(metrics_port)
        
        # Initialize sliding windows for monitoring
        self.prediction_window = deque(maxlen=1000)
        self.feature_windows = {
            feature: deque(maxlen=1000)
            for feature in self.reference_stats['features'].keys()
        }

        # Start monitoring thread
        self.stop_flag = False
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()

    def _monitor_loop(self):
        consumer = KafkaConsumer(
            'fraud_predictions',
            bootstrap_servers=self.kafka_bootstrap_servers,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            auto_offset_reset='latest',
            group_id='model_monitor'
        )
        
        try:
            for message in consumer:
                if self.stop_flag:
                    break
                    
                prediction_data = message.value
                self._process_prediction(prediction_data)
                self._update_metrics()
                
        except Exception as e:
            logger.error(f"Error in monitoring loop: {str(e)}")
        finally:
            consumer.close()

    def _process_prediction(self, prediction_data: Dict[str, Any]):
        # Update counters
        self.prediction_counter.inc()
        
        # Store prediction result
        self.prediction_window.append({
            'timestamp': prediction_data['prediction_timestamp'],
            'is_fraudulent': prediction_data['is_fraudulent'],
            'probability': prediction_data['fraud_probability']
        })
        
        # Store feature values
        features = prediction_data['features_used']
        for feature_name, value in features.items():
            if feature_name in self.feature_windows:
                self.feature_windows[feature_name].append(value)

    def _update_metrics(self):
        self._update_fraud_ratio()
        self._check_feature_drift()
        self._check_performance_metrics()

    def _update_fraud_ratio(self):
        if self.prediction_window:
            fraud_ratio = sum(
                1 for p in self.prediction_window if p['is_fraudulent']
            ) / len(self.prediction_window)
            self.fraud_ratio_gauge.set(fraud_ratio)

    def _check_feature_drift(self):
        for feature_name, values in self.feature_windows.items():
            if len(values) >= 100:  # Minimum sample size
                current_stats = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'skew': stats.skew(values)
                }
                
                # Calculate drift score
                ref_stats = self.reference_stats['features'][feature_name]
                drift_score = abs(
                    (current_stats['mean'] - ref_stats['mean']) / ref_stats['std']
                )
                
                self.feature_drift_gauge.labels(feature_name).set(drift_score)
                
                if drift_score > 2.0:
                    logger.warning(
                        f"Significant drift detected in feature {feature_name}: "
                        f"score = {drift_score:.2f}"
                    )

    def _check_performance_metrics(self):
        if len(self.prediction_window) >= 100:
            recent_predictions = list(self.prediction_window)[-100:]
            
            timestamps = [
                datetime.fromisoformat(p['timestamp'])
                for p in recent_predictions
            ]
            latencies = [
                (t2 - t1).total_seconds()
                for t1, t2 in zip(timestamps[:-1], timestamps[1:])
            ]
            
            if latencies:
                avg_latency = np.mean(latencies)
                self.prediction_latency.observe(avg_latency)
                
                if avg_latency > 1.0:
                    logger.warning(
                        f"High prediction latency detected: {avg_latency:.2f}s"
                    )

    def generate_monitoring_report(self) -> Dict[str, Any]:
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_predictions': self.prediction_counter._value.get(),
            'fraud_ratio': self.fraud_ratio_gauge._value.get(),
            'feature_drift': {
                feature: self.feature_drift_gauge.labels(feature)._value.get()
                for feature in self.reference_stats['features'].keys()
            },
            'performance_metrics': {
                'avg_latency': np.mean([
                    sample[0] for sample in self.prediction_latency._samples()
                ])
            }
        }
        return report

    def stop(self):
        self.stop_flag = True
        self.monitor_thread.join()
        logger.info("Model monitoring stopped")

if __name__ == "__main__":
    # Create sample reference statistics
    reference_stats = {
        'features': {
            'amount': {'mean': 100.0, 'std': 50.0, 'skew': 0.5},
            'hour_of_day': {'mean': 12.0, 'std': 6.0, 'skew': 0.0},
            'transaction_frequency_24h': {'mean': 5.0, 'std': 3.0, 'skew': 0.8}
        }
    }
    
    # Save reference statistics
    with open('reference_stats.pkl', 'wb') as f:
        pickle.dump(reference_stats, f)
    
    # Initialize and run monitor
    monitor = ModelMonitor('reference_stats.pkl')
    
    try:
        while True:
            # Generate monitoring report every hour
            time.sleep(3600)
            report = monitor.generate_monitoring_report()
            logger.info(f"Monitoring report: {json.dumps(report, indent=2)}")
    except KeyboardInterrupt:
        monitor.stop()