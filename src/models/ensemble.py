"""
Ensemble Detection System
Combines multiple models for improved detection
"""
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List

class EnsembleDetector:
    def __init__(self, voting_strategy='weighted', threshold=0.6):
        """
        Args:
            voting_strategy: 'weighted' or 'majority'
            threshold: Decision threshold for classification
        """
        self.voting_strategy = voting_strategy
        self.threshold = threshold
        self.models = {}
        self.weights = {}
        
    def add_model(self, name, model, weight=1.0):
        """Add a model to the ensemble"""
        self.models[name] = model
        self.weights[name] = weight
        print(f"Added model '{name}' with weight {weight}")
        
    def normalize_weights(self):
        """Normalize weights to sum to 1"""
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
    def predict_single_model(self, model_name, X, return_proba=True):
        """Get predictions from a single model"""
        model = self.models[model_name]
        
        try:
            if return_proba:
                if hasattr(model, 'predict_proba'):
                    return model.predict_proba(X)
                elif hasattr(model, 'predict_score'):
                    return model.predict_score(X)
                else:
                    # Fall back to binary predictions
                    return model.predict(X).astype(float)
            else:
                return model.predict(X)
        except Exception as e:
            print(f"Error predicting with {model_name}: {e}")
            return None
    
    def predict(self, X, X_sequence=None):
        """
        Ensemble prediction
        
        Args:
            X: Feature matrix for traditional ML models
            X_sequence: Sequence data for LSTM model (optional)
        
        Returns:
            predictions: Binary predictions (0/1)
            scores: Confidence scores
            individual_scores: Dict of scores from each model
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        # Normalize weights
        self.normalize_weights()
        
        # Collect predictions from all models
        individual_scores = {}
        
        for model_name, model in self.models.items():
            try:
                # Use sequence data for LSTM, regular features for others
                if 'lstm' in model_name.lower() and X_sequence is not None:
                    scores = self.predict_single_model(model_name, X_sequence, return_proba=True)
                else:
                    scores = self.predict_single_model(model_name, X, return_proba=True)
                
                if scores is not None:
                    individual_scores[model_name] = scores
            except Exception as e:
                print(f"Skipping model {model_name} due to error: {e}")
                continue
        
        if not individual_scores:
            raise ValueError("No models produced valid predictions")
        
        # Combine predictions based on strategy
        if self.voting_strategy == 'weighted':
            # Weighted average of scores
            # First, normalize all scores to 1D arrays
            normalized_scores = {}
            for model_name, scores in individual_scores.items():
                # Convert 2D probability arrays to 1D anomaly scores
                if scores.ndim == 2:
                    if scores.shape[1] == 2:
                        # Binary classification: take probability of positive class (attack)
                        normalized_scores[model_name] = scores[:, 1]
                    else:
                        # Multi-dimensional: take mean or max
                        normalized_scores[model_name] = scores.max(axis=1)
                else:
                    # Already 1D
                    normalized_scores[model_name] = scores
            
            # Get number of samples
            first_scores = next(iter(normalized_scores.values()))
            n_samples = len(first_scores)
            ensemble_scores = np.zeros(n_samples)

            # Weighted sum
            for model_name, scores in normalized_scores.items():
                weight = self.weights.get(model_name, 0)
                ensemble_scores += weight * scores

        
        elif self.voting_strategy == 'majority':
            # Majority voting - normalize scores first
            normalized_scores = {}
            for model_name, scores in individual_scores.items():
                if scores.ndim == 2:
                    if scores.shape[1] == 2:
                        normalized_scores[model_name] = scores[:, 1]
                    else:
                        normalized_scores[model_name] = scores.max(axis=1)
                else:
                    normalized_scores[model_name] = scores
            
            # Get number of samples
            first_scores = next(iter(normalized_scores.values()))
            n_samples = len(first_scores)
            votes = np.zeros(n_samples)
            
            for model_name, scores in normalized_scores.items():
                binary_preds = (scores > 0.5).astype(int)
                votes += binary_preds
            
            ensemble_scores = votes / len(normalized_scores)
        
        else:
            raise ValueError(f"Unknown voting strategy: {self.voting_strategy}")
        
        # Apply threshold
        predictions = (ensemble_scores > self.threshold).astype(int)
        
        # Return normalized scores instead of original individual_scores
        return predictions, ensemble_scores, normalized_scores
    
    def predict_with_explanation(self, X, X_sequence=None, feature_names=None):
        """
        Predict with detailed explanation of the decision
        
        Returns:
            result: Dict with prediction, scores, and explanations
        """
        predictions, ensemble_scores, individual_scores = self.predict(X, X_sequence)
        
        results = []
        for i in range(len(predictions)):
            result = {
                'prediction': int(predictions[i]),
                'ensemble_score': float(ensemble_scores[i]),
                'confidence': float(abs(ensemble_scores[i] - 0.5) * 2),  # 0-1 scale
                'individual_scores': {
                    name: float(scores[i]) 
                    for name, scores in individual_scores.items()
                },
                'decision': 'ATTACK' if predictions[i] == 1 else 'NORMAL',
                'risk_level': self._get_risk_level(ensemble_scores[i])
            }
            results.append(result)
        
        return results
    
    def _get_risk_level(self, score):
        """Determine risk level from score"""
        if score < 0.3:
            return 'LOW'
        elif score < 0.6:
            return 'MEDIUM'
        elif score < 0.8:
            return 'HIGH'
        else:
            return 'CRITICAL'
    
    def evaluate(self, X, y_true, X_sequence=None):
        """Evaluate ensemble performance"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, roc_auc_score, confusion_matrix
        )
        
        predictions, scores, individual_scores = self.predict(X, X_sequence)
        
        metrics = {
            'accuracy': accuracy_score(y_true, predictions),
            'precision': precision_score(y_true, predictions, zero_division=0),
            'recall': recall_score(y_true, predictions, zero_division=0),
            'f1_score': f1_score(y_true, predictions, zero_division=0),
            'roc_auc': roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else 0,
            'confusion_matrix': confusion_matrix(y_true, predictions).tolist()
        }
        
        # Individual model performance
        individual_metrics = {}
        for model_name, model_scores in individual_scores.items():
            model_predictions = (model_scores > 0.5).astype(int)
            individual_metrics[model_name] = {
                'accuracy': accuracy_score(y_true, model_predictions),
                'precision': precision_score(y_true, model_predictions, zero_division=0),
                'recall': recall_score(y_true, model_predictions, zero_division=0),
                'f1_score': f1_score(y_true, model_predictions, zero_division=0),
            }
        
        metrics['individual_models'] = individual_metrics
        
        return metrics
    
    def save(self, path):
        """Save ensemble configuration"""
        config = {
            'voting_strategy': self.voting_strategy,
            'threshold': self.threshold,
            'weights': self.weights,
            'model_names': list(self.models.keys())
        }
        joblib.dump(config, path)
        print(f"Ensemble configuration saved to {path}")
    
    def load_config(self, path):
        """Load ensemble configuration (models must be loaded separately)"""
        config = joblib.load(path)
        self.voting_strategy = config['voting_strategy']
        self.threshold = config['threshold']
        self.weights = config['weights']
        print(f"Ensemble configuration loaded from {path}")
        return config['model_names']


class AttackDetectionPipeline:
    """Complete detection pipeline with preprocessing and ensemble"""
    
    def __init__(self, ensemble_detector, feature_engineer):
        self.ensemble = ensemble_detector
        self.feature_engineer = feature_engineer
        
    def detect_attacks(self, session_logs_df, return_details=False):
        """
        Detect attacks in session logs
        
        Args:
            session_logs_df: DataFrame with raw session logs
            return_details: If True, return detailed explanations
        
        Returns:
            results_df: DataFrame with detection results
        """
        # Feature engineering
        print("Engineering features...")
        df_features = self.feature_engineer.engineer_features(session_logs_df)
        
        # Extract ML features
        X = self.feature_engineer.get_ml_features(df_features)
        
        # Prepare sequence data if LSTM is in ensemble
        X_sequence = None
        if any('lstm' in name.lower() for name in self.ensemble.models.keys()):
            print("Preparing sequence features for LSTM...")
            # This is a simplified version - in practice, you'd use proper sequence preparation
            # For now, we'll skip LSTM in real-time detection
            pass
        
        # Make predictions
        print("Making predictions...")
        if return_details:
            results = self.ensemble.predict_with_explanation(X.values, X_sequence)
            results_df = pd.DataFrame(results)
        else:
            predictions, scores, _ = self.ensemble.predict(X.values, X_sequence)
            results_df = pd.DataFrame({
                'prediction': predictions,
                'score': scores,
                'decision': ['ATTACK' if p == 1 else 'NORMAL' for p in predictions]
            })
        
        # Add original data
        results_df = pd.concat([
            session_logs_df.reset_index(drop=True),
            results_df
        ], axis=1)
        
        return results_df
    
    def generate_alert(self, detection_result):
        """Generate alert for detected attacks"""
        if detection_result['prediction'] == 1:
            alert = {
                'timestamp': detection_result.get('timestamp', 'N/A'),
                'session_id': detection_result.get('session_id', 'N/A'),
                'user_id': detection_result.get('user_id', 'N/A'),
                'attack_type': detection_result.get('attack_type', 'UNKNOWN'),
                'risk_level': detection_result.get('risk_level', 'UNKNOWN'),
                'confidence': detection_result.get('confidence', 0),
                'ip_address': detection_result.get('ip_address', 'N/A'),
                'location': f"{detection_result.get('city', 'N/A')}, {detection_result.get('country', 'N/A')}",
                'message': f"Potential session attack detected for user {detection_result.get('user_id', 'N/A')}",
                'recommended_action': self._get_recommended_action(detection_result)
            }
            return alert
        return None
    
    def _get_recommended_action(self, detection_result):
        """Recommend action based on detection"""
        risk_level = detection_result.get('risk_level', 'UNKNOWN')
        
        if risk_level == 'CRITICAL':
            return "IMMEDIATE: Terminate session and require re-authentication"
        elif risk_level == 'HIGH':
            return "Require additional authentication (2FA) for sensitive actions"
        elif risk_level == 'MEDIUM':
            return "Monitor session closely and log all actions"
        else:
            return "Continue monitoring"


if __name__ == "__main__":
    print("Testing Ensemble Detector...")
    
    # Create mock models for testing
    class MockModel:
        def __init__(self, name):
            self.name = name
            
        def predict_proba(self, X):
            return np.random.rand(len(X))
    
    # Create ensemble
    ensemble = EnsembleDetector(voting_strategy='weighted', threshold=0.6)
    
    # Add mock models
    ensemble.add_model('isolation_forest', MockModel('if'), weight=0.2)
    ensemble.add_model('one_class_svm', MockModel('svm'), weight=0.15)
    ensemble.add_model('autoencoder', MockModel('ae'), weight=0.25)
    ensemble.add_model('random_forest', MockModel('rf'), weight=0.4)
    
    # Test prediction
    X_test = np.random.randn(10, 20)
    predictions, scores, individual_scores = ensemble.predict(X_test)
    
    print(f"Predictions: {predictions}")
    print(f"Scores: {scores}")
    print(f"Individual scores: {list(individual_scores.keys())}")
    
    # Test with explanation
    results = ensemble.predict_with_explanation(X_test)
    print(f"\nSample result with explanation:")
    print(results[0])
    
    print("\nEnsemble test passed!")
