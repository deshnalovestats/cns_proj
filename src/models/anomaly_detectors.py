"""
Anomaly Detection Models
- Isolation Forest
- One-Class SVM
- Autoencoder
- K-Means Clustering
- Gaussian Mixture Model (GMM)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import json

class IsolationForestDetector:
    def __init__(self, n_estimators=100, contamination=0.1, max_samples=256):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def train(self, X_train):
        """Train on normal data"""
        print("Training Isolation Forest...")
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled)
        self.is_fitted = True
        print("Isolation Forest training complete.")
        
    def predict(self, X):
        """Predict anomalies. Returns 1 for anomaly, 0 for normal"""
        if not self.is_fitted:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        # Convert: -1 (anomaly) -> 1, 1 (normal) -> 0
        return (predictions == -1).astype(int)
    
    def predict_score(self, X):
        """Return anomaly scores (lower = more anomalous)"""
        if not self.is_fitted:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        scores = self.model.score_samples(X_scaled)
        # Normalize to 0-1 (higher = more anomalous)
        scores_normalized = 1 / (1 + np.exp(scores))
        return scores_normalized
    
    def save(self, path):
        """Save model"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted
        }, path)
        print(f"Isolation Forest saved to {path}")
        
    def load(self, path):
        """Load model"""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_fitted = data['is_fitted']
        print(f"Isolation Forest loaded from {path}")


class OneClassSVMDetector:
    def __init__(self, kernel='rbf', nu=0.1, gamma='auto'):
        self.model = OneClassSVM(
            kernel=kernel,
            nu=nu,
            gamma=gamma
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def train(self, X_train):
        """Train on normal data"""
        print("Training One-Class SVM...")
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled)
        self.is_fitted = True
        print("One-Class SVM training complete.")
        
    def predict(self, X):
        """Predict anomalies. Returns 1 for anomaly, 0 for normal"""
        if not self.is_fitted:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        # Convert: -1 (anomaly) -> 1, 1 (normal) -> 0
        return (predictions == -1).astype(int)
    
    def predict_score(self, X):
        """Return anomaly scores"""
        if not self.is_fitted:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        scores = self.model.decision_function(X_scaled)
        # Normalize to 0-1 (higher = more anomalous)
        scores_normalized = 1 / (1 + np.exp(scores))
        return scores_normalized
    
    def save(self, path):
        """Save model"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted
        }, path)
        print(f"One-Class SVM saved to {path}")
        
    def load(self, path):
        """Load model"""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_fitted = data['is_fitted']
        print(f"One-Class SVM loaded from {path}")


class AutoencoderDetector:
    def __init__(self, input_dim, encoding_dim=32, hidden_layers=[128, 64, 32]):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers
        self.scaler = StandardScaler()
        self.model = None
        self.threshold = None
        self.is_fitted = False
        
    def build_model(self):
        """Build autoencoder architecture"""
        # Encoder
        encoder_input = layers.Input(shape=(self.input_dim,))
        x = encoder_input
        
        for units in self.hidden_layers:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.Dropout(0.2)(x)
        
        encoded = layers.Dense(self.encoding_dim, activation='relu', name='encoding')(x)
        
        # Decoder
        x = encoded
        for units in reversed(self.hidden_layers):
            x = layers.Dense(units, activation='relu')(x)
            x = layers.Dropout(0.2)(x)
        
        decoded = layers.Dense(self.input_dim, activation='linear')(x)
        
        # Autoencoder model
        autoencoder = keras.Model(encoder_input, decoded, name='autoencoder')
        autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        self.model = autoencoder
        return autoencoder
    
    def train(self, X_train, epochs=50, batch_size=32, validation_split=0.1):
        """Train autoencoder on normal data"""
        print("Training Autoencoder...")
        
        X_scaled = self.scaler.fit_transform(X_train)
        
        if self.model is None:
            self.build_model()
        
        # Early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        history = self.model.fit(
            X_scaled, X_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=1
        )
        
        # Calculate reconstruction error threshold
        reconstructions = self.model.predict(X_scaled)
        mse = np.mean(np.square(X_scaled - reconstructions), axis=1)
        self.threshold = np.percentile(mse, 95)  # 95th percentile
        
        self.is_fitted = True
        print(f"Autoencoder training complete. Threshold: {self.threshold:.4f}")
        
        return history
    
    def predict(self, X):
        """Predict anomalies. Returns 1 for anomaly, 0 for normal"""
        if not self.is_fitted:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        reconstructions = self.model.predict(X_scaled, verbose=0)
        mse = np.mean(np.square(X_scaled - reconstructions), axis=1)
        
        return (mse > self.threshold).astype(int)
    
    def predict_score(self, X):
        """Return anomaly scores (reconstruction error)"""
        if not self.is_fitted:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        reconstructions = self.model.predict(X_scaled, verbose=0)
        mse = np.mean(np.square(X_scaled - reconstructions), axis=1)
        
        # Normalize scores
        scores_normalized = np.clip(mse / (self.threshold * 2), 0, 1)
        return scores_normalized
    
    def save(self, path):
        """Save model"""
        self.model.save(f"{path}_model.keras")
        joblib.dump({
            'scaler': self.scaler,
            'threshold': self.threshold,
            'input_dim': self.input_dim,
            'encoding_dim': self.encoding_dim,
            'hidden_layers': self.hidden_layers,
            'is_fitted': self.is_fitted
        }, f"{path}_metadata.pkl")
        print(f"Autoencoder saved to {path}")
        
    def load(self, path):
        """Load model"""
        self.model = keras.models.load_model(f"{path}_model.keras")
        metadata = joblib.load(f"{path}_metadata.pkl")
        self.scaler = metadata['scaler']
        self.threshold = metadata['threshold']
        self.input_dim = metadata['input_dim']
        self.encoding_dim = metadata['encoding_dim']
        self.hidden_layers = metadata['hidden_layers']
        self.is_fitted = metadata['is_fitted']
        print(f"Autoencoder loaded from {path}")


class KMeansDetector:
    def __init__(self, n_clusters=5, contamination=0.1):
        """
        K-Means based anomaly detector
        Args:
            n_clusters: Number of clusters to form
            contamination: Expected proportion of anomalies (for threshold calculation)
        """
        self.n_clusters = n_clusters
        self.contamination = contamination
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.threshold = None
        
    def train(self, X_train):
        """Train on normal data"""
        print(f"Training K-Means with {self.n_clusters} clusters...")
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled)
        
        # Calculate distances to nearest cluster center
        distances = self.model.transform(X_scaled).min(axis=1)
        
        # Set threshold based on contamination level
        self.threshold = np.percentile(distances, 100 * (1 - self.contamination))
        
        self.is_fitted = True
        print(f"K-Means training complete. Threshold: {self.threshold:.4f}")
        
    def predict(self, X):
        """Predict anomalies. Returns 1 for anomaly, 0 for normal"""
        if not self.is_fitted:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        distances = self.model.transform(X_scaled).min(axis=1)
        
        # Points far from all clusters are anomalies
        return (distances > self.threshold).astype(int)
    
    def predict_score(self, X):
        """Return anomaly scores (distance to nearest cluster)"""
        if not self.is_fitted:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        distances = self.model.transform(X_scaled).min(axis=1)
        
        # Normalize scores to 0-1 range
        scores_normalized = np.clip(distances / (self.threshold * 2), 0, 1)
        return scores_normalized
    
    def save(self, path):
        """Save model"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'threshold': self.threshold,
            'n_clusters': self.n_clusters,
            'contamination': self.contamination,
            'is_fitted': self.is_fitted
        }, path)
        print(f"K-Means saved to {path}")
        
    def load(self, path):
        """Load model"""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.threshold = data['threshold']
        self.n_clusters = data['n_clusters']
        self.contamination = data['contamination']
        self.is_fitted = data['is_fitted']
        print(f"K-Means loaded from {path}")


class GMMDetector:
    def __init__(self, n_components=5, contamination=0.1, covariance_type='full'):
        """
        Gaussian Mixture Model based anomaly detector
        Args:
            n_components: Number of Gaussian components
            contamination: Expected proportion of anomalies (for threshold calculation)
            covariance_type: Type of covariance parameters ('full', 'tied', 'diag', 'spherical')
        """
        self.n_components = n_components
        self.contamination = contamination
        self.covariance_type = covariance_type
        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.threshold = None
        
    def train(self, X_train):
        """Train on normal data"""
        print(f"Training GMM with {self.n_components} components...")
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled)
        
        # Calculate log-likelihood scores
        log_likelihoods = self.model.score_samples(X_scaled)
        
        # Set threshold based on contamination level (lower log-likelihood = anomaly)
        self.threshold = np.percentile(log_likelihoods, 100 * self.contamination)
        
        self.is_fitted = True
        print(f"GMM training complete. Threshold: {self.threshold:.4f}")
        
    def predict(self, X):
        """Predict anomalies. Returns 1 for anomaly, 0 for normal"""
        if not self.is_fitted:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        log_likelihoods = self.model.score_samples(X_scaled)
        
        # Points with low log-likelihood are anomalies
        return (log_likelihoods < self.threshold).astype(int)
    
    def predict_score(self, X):
        """Return anomaly scores (inverse normalized log-likelihood)"""
        if not self.is_fitted:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        log_likelihoods = self.model.score_samples(X_scaled)
        
        # Convert to anomaly scores (lower likelihood = higher anomaly score)
        # Normalize to 0-1 range
        scores = self.threshold - log_likelihoods
        scores_normalized = np.clip(scores / (abs(self.threshold) * 2), 0, 1)
        return scores_normalized
    
    def save(self, path):
        """Save model"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'threshold': self.threshold,
            'n_components': self.n_components,
            'contamination': self.contamination,
            'covariance_type': self.covariance_type,
            'is_fitted': self.is_fitted
        }, path)
        print(f"GMM saved to {path}")
        
    def load(self, path):
        """Load model"""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.threshold = data['threshold']
        self.n_components = data['n_components']
        self.contamination = data['contamination']
        self.covariance_type = data['covariance_type']
        self.is_fitted = data['is_fitted']
        print(f"GMM loaded from {path}")


if __name__ == "__main__":
    # Test models with synthetic data
    print("Testing Anomaly Detection Models...")
    
    # Generate synthetic data
    np.random.seed(42)
    X_normal = np.random.randn(1000, 20)
    X_anomaly = np.random.randn(100, 20) * 3 + 5
    
    # Test Isolation Forest
    print("\n" + "="*50)
    print("Testing Isolation Forest")
    print("="*50)
    iso = IsolationForestDetector()
    iso.train(X_normal)
    
    normal_preds = iso.predict(X_normal[:10])
    anomaly_preds = iso.predict(X_anomaly[:10])
    print(f"Normal predictions: {normal_preds}")
    print(f"Anomaly predictions: {anomaly_preds}")
    
    # Test One-Class SVM
    print("\n" + "="*50)
    print("Testing One-Class SVM")
    print("="*50)
    svm = OneClassSVMDetector()
    svm.train(X_normal)
    
    normal_preds = svm.predict(X_normal[:10])
    anomaly_preds = svm.predict(X_anomaly[:10])
    print(f"Normal predictions: {normal_preds}")
    print(f"Anomaly predictions: {anomaly_preds}")
    
    # Test Autoencoder
    print("\n" + "="*50)
    print("Testing Autoencoder")
    print("="*50)
    ae = AutoencoderDetector(input_dim=20, encoding_dim=8, hidden_layers=[16, 12, 8])
    ae.train(X_normal, epochs=10, batch_size=32)
    
    normal_preds = ae.predict(X_normal[:10])
    anomaly_preds = ae.predict(X_anomaly[:10])
    print(f"Normal predictions: {normal_preds}")
    print(f"Anomaly predictions: {anomaly_preds}")
    
    print("\nAll tests passed!")
