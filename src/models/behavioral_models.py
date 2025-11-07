"""
Behavioral Sequence Models
- LSTM for sequence analysis
- Random Forest for feature-based classification
- Gradient Boosting for advanced classification
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib

class LSTMSequenceDetector:
    def __init__(self, input_dim, sequence_length=20, hidden_size=128, 
                 num_layers=2, dropout=0.3):
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def build_model(self):
        """Build LSTM architecture"""
        model = keras.Sequential([
            layers.Input(shape=(self.sequence_length, self.input_dim)),
            
            # First LSTM layer
            layers.LSTM(self.hidden_size, return_sequences=True if self.num_layers > 1 else False),
            layers.Dropout(self.dropout),
            
            # Additional LSTM layers
            *[layers.LSTM(self.hidden_size, return_sequences=(i < self.num_layers - 2))
              for i in range(self.num_layers - 1)],
            *[layers.Dropout(self.dropout) for _ in range(self.num_layers - 1)],
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(self.dropout),
            layers.Dense(32, activation='relu'),
            layers.Dropout(self.dropout),
            
            # Output layer
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        return model
    
    def prepare_sequences(self, df, feature_cols, sequence_length=None):
        """Prepare sequence data from dataframe"""
        if sequence_length is None:
            sequence_length = self.sequence_length
            
        df = df.sort_values(['session_id', 'timestamp'])
        
        sequences = []
        labels = []
        
        for session_id, group in df.groupby('session_id'):
            session_features = group[feature_cols].values
            session_labels = group['is_attack'].values
            
            for i in range(len(session_features)):
                # Get sequence up to current point
                start_idx = max(0, i - sequence_length + 1)
                sequence = session_features[start_idx:i+1]
                
                # Pad if necessary
                if len(sequence) < sequence_length:
                    padding = np.zeros((sequence_length - len(sequence), len(feature_cols)))
                    sequence = np.vstack([padding, sequence])
                
                sequences.append(sequence)
                labels.append(session_labels[i])
        
        return np.array(sequences), np.array(labels)
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=30, batch_size=64):
        """Train LSTM model"""
        print("Training LSTM Sequence Detector...")
        
        # Scale features for each time step
        n_samples, n_timesteps, n_features = X_train.shape
        X_train_reshaped = X_train.reshape(-1, n_features)
        X_train_scaled = self.scaler.fit_transform(X_train_reshaped)
        X_train_scaled = X_train_scaled.reshape(n_samples, n_timesteps, n_features)
        
        if X_val is not None:
            n_val_samples = X_val.shape[0]
            X_val_reshaped = X_val.reshape(-1, n_features)
            X_val_scaled = self.scaler.transform(X_val_reshaped)
            X_val_scaled = X_val_scaled.reshape(n_val_samples, n_timesteps, n_features)
            validation_data = (X_val_scaled, y_val)
        else:
            validation_data = None
        
        if self.model is None:
            self.build_model()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=3
            )
        ]
        
        history = self.model.fit(
            X_train_scaled, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_fitted = True
        print("LSTM training complete.")
        
        return history
    
    def predict(self, X):
        """Predict attack probability"""
        if not self.is_fitted:
            raise ValueError("Model not trained yet")
        
        n_samples, n_timesteps, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)
        
        predictions = self.model.predict(X_scaled, verbose=0)
        return (predictions.flatten() > 0.5).astype(int)
    
    def predict_proba(self, X):
        """Predict attack probability scores"""
        if not self.is_fitted:
            raise ValueError("Model not trained yet")
        
        n_samples, n_timesteps, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)
        
        probabilities = self.model.predict(X_scaled, verbose=0)
        return probabilities.flatten()
    
    def save(self, path):
        """Save model"""
        self.model.save(f"{path}_model.keras")
        joblib.dump({
            'scaler': self.scaler,
            'input_dim': self.input_dim,
            'sequence_length': self.sequence_length,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'is_fitted': self.is_fitted
        }, f"{path}_metadata.pkl")
        print(f"LSTM model saved to {path}")
        
    def load(self, path):
        """Load model"""
        self.model = keras.models.load_model(f"{path}_model.keras")
        metadata = joblib.load(f"{path}_metadata.pkl")
        self.scaler = metadata['scaler']
        self.input_dim = metadata['input_dim']
        self.sequence_length = metadata['sequence_length']
        self.hidden_size = metadata['hidden_size']
        self.num_layers = metadata['num_layers']
        self.dropout = metadata['dropout']
        self.is_fitted = metadata['is_fitted']
        print(f"LSTM model loaded from {path}")


class RandomForestDetector:
    def __init__(self, n_estimators=200, max_depth=20, min_samples_split=5):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_importances_ = None
        
    def train(self, X_train, y_train):
        """Train Random Forest"""
        print("Training Random Forest...")
        
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.feature_importances_ = self.model.feature_importances_
        
        self.is_fitted = True
        print("Random Forest training complete.")
        
    def predict(self, X):
        """Predict attacks"""
        if not self.is_fitted:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Predict attack probabilities"""
        if not self.is_fitted:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)
        return probabilities[:, 1]  # Probability of attack class
    
    def get_feature_importance(self, feature_names):
        """Get feature importance rankings"""
        if self.feature_importances_ is None:
            return None
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save(self, path):
        """Save model"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_importances': self.feature_importances_,
            'is_fitted': self.is_fitted
        }, path)
        print(f"Random Forest saved to {path}")
        
    def load(self, path):
        """Load model"""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_importances_ = data['feature_importances']
        self.is_fitted = data['is_fitted']
        print(f"Random Forest loaded from {path}")


class GradientBoostingDetector:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=5, min_samples_split=5):
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_importances_ = None
        
    def train(self, X_train, y_train):
        """Train Gradient Boosting"""
        print("Training Gradient Boosting...")
        
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.feature_importances_ = self.model.feature_importances_
        
        self.is_fitted = True
        print("Gradient Boosting training complete.")
        
    def predict(self, X):
        """Predict attacks"""
        if not self.is_fitted:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Predict attack probabilities"""
        if not self.is_fitted:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)
        return probabilities[:, 1]  # Probability of attack class
    
    def get_feature_importance(self, feature_names):
        """Get feature importance rankings"""
        if self.feature_importances_ is None:
            return None
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save(self, path):
        """Save model"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_importances': self.feature_importances_,
            'is_fitted': self.is_fitted
        }, path)
        print(f"Gradient Boosting saved to {path}")
        
    def load(self, path):
        """Load model"""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_importances_ = data['feature_importances']
        self.is_fitted = data['is_fitted']
        print(f"Gradient Boosting loaded from {path}")


if __name__ == "__main__":
    print("Testing Behavioral Models...")
    
    # Generate synthetic sequence data
    np.random.seed(42)
    n_samples = 1000
    sequence_length = 20
    n_features = 15
    
    X_seq = np.random.randn(n_samples, sequence_length, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    # Test LSTM
    print("\n" + "="*50)
    print("Testing LSTM")
    print("="*50)
    lstm = LSTMSequenceDetector(
        input_dim=n_features,
        sequence_length=sequence_length,
        hidden_size=64,
        num_layers=2
    )
    lstm.train(X_seq[:800], y[:800], X_seq[800:], y[800:], epochs=5, batch_size=32)
    
    preds = lstm.predict(X_seq[:10])
    probs = lstm.predict_proba(X_seq[:10])
    print(f"Predictions: {preds}")
    print(f"Probabilities: {probs}")
    
    # Test Random Forest
    print("\n" + "="*50)
    print("Testing Random Forest")
    print("="*50)
    X_flat = np.random.randn(1000, 20)
    y_flat = np.random.randint(0, 2, 1000)
    
    rf = RandomForestDetector(n_estimators=50)
    rf.train(X_flat[:800], y_flat[:800])
    
    preds = rf.predict(X_flat[800:810])
    probs = rf.predict_proba(X_flat[800:810])
    print(f"Predictions: {preds}")
    print(f"Probabilities: {probs}")
    
    print("\nAll tests passed!")
