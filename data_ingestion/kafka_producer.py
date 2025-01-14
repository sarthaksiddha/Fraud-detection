#!/usr/bin/env python3
# data_ingestion/kafka_producer.py

from kafka import KafkaProducer
import json
import logging
from datetime import datetime
import time
import random
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransactionProducer:
    """
    Simulates real-time transaction data and publishes to Kafka topic
    """
    
    def __init__(self, bootstrap_servers: str, topic_name: str):
        """
        Initialize the transaction producer
        
        Args:
            bootstrap_servers: Kafka broker addresses
            topic_name: Topic to publish transactions to
        """
        self.topic_name = topic_name
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

    def generate_transaction(self) -> Dict[str, Any]:
        """
        Generate a synthetic transaction
        
        Returns:
            Dict containing transaction details
        """
        transaction_types = ['PAYMENT', 'TRANSFER', 'WITHDRAWAL', 'DEPOSIT']
        countries = ['US', 'UK', 'FR', 'DE', 'IT', 'ES']
        
        transaction = {
            'transaction_id': f"TX-{int(time.time() * 1000)}",
            'timestamp': datetime.now().isoformat(),
            'account_id': f"ACC-{random.randint(1000, 9999)}",
            'transaction_type': random.choice(transaction_types),
            'amount': round(random.uniform(10, 10000), 2),
            'currency': 'USD',
            'country': random.choice(countries),
            'merchant_category': random.randint(1000, 9999),
        }
        return transaction

    def send_transaction(self, transaction: Dict[str, Any]):
        """
        Send transaction to Kafka topic
        
        Args:
            transaction: Transaction data to send
        """
        try:
            future = self.producer.send(self.topic_name, transaction)
            self.producer.flush()  # Ensure message is sent
            future.get(timeout=10)  # Wait for message to be delivered
            logger.info(f"Sent transaction: {transaction['transaction_id']}")
        except Exception as e:
            logger.error(f"Error sending transaction: {str(e)}")

    def simulate_transactions(self, interval: float = 1.0):
        """
        Continuously generate and send transactions
        
        Args:
            interval: Time between transactions in seconds
        """
        try:
            while True:
                transaction = self.generate_transaction()
                self.send_transaction(transaction)
                time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("Stopping transaction simulation...")
            self.producer.close()

if __name__ == "__main__":
    # Configuration
    KAFKA_BOOTSTRAP_SERVERS = 'localhost:9092'  # Update with your Kafka broker address
    KAFKA_TOPIC = 'transactions'
    
    # Create and run producer
    producer = TransactionProducer(KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC)
    producer.simulate_transactions(interval=2.0)  # Generate transaction every 2 seconds