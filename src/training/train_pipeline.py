"""
Training Pipeline
Train all models and save artifacts
"""
import pandas as pd
import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split
import yaml
import joblib

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from preprocessing.data_generator import SessionDataGenerator
from preprocessing.feature_engineering import FeatureEngineer
from models.anomaly_detectors import (
    IsolationForestDetector, 
    OneClassSVMDetector, 
    AutoencoderDetector,
    KMeansDetector,
    GMMDetector
)
from models.behavioral_models import (
    LSTMSequenceDetector,
    RandomForestDetector,
    GradientBoostingDetector
)
from models.dbscan_detector import DBSCANAnomalyDetector
from models.ensemble import EnsembleDetector

def load_config(config_path='config.yaml'):
    """Load configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def generate_or_load_data(config):
    """Generate or load training data"""
    data_path = 'data/raw/session_logs.csv'
    
    if os.path.exists(data_path):
        print(f"Loading existing data from {data_path}...")
        df = pd.read_csv(data_path)
    else:
        print("Generating new training data...")
        generator = SessionDataGenerator()
        df = generator.generate_dataset(
            num_normal=config['data_generation']['num_normal_sessions'],
            num_hijack=config['data_generation']['num_hijack_sessions'],
            num_fixation=config['data_generation']['num_fixation_sessions']
        )
        df.to_csv(data_path, index=False)
    
    print(f"Dataset loaded: {len(df)} events, {df['session_id'].nunique()} sessions")
    return df

def prepare_features(df, config):
    """Engineer features"""
    processed_path = 'data/processed/session_logs_features.csv'
    
    if os.path.exists(processed_path):
        print(f"Loading processed features from {processed_path}...")
        df_features = pd.read_csv(processed_path)
        # Recreate the engineer for consistent feature extraction
        engineer = FeatureEngineer()
    else:
        print("Engineering features...")
        engineer = FeatureEngineer()
        df_features = engineer.engineer_features(df)
        df_features.to_csv(processed_path, index=False)
    
    # Extract ML features
    X = engineer.get_ml_features(df_features)
    y = df_features['is_attack']
    
    print(f"Features prepared: {X.shape[1]} features, {len(X)} samples")
    print(f"Attack rate: {y.mean():.2%}")
    
    # Save feature engineer
    joblib.dump(engineer, 'models/feature_engineer.pkl')
    
    return X, y, df_features, engineer

def train_anomaly_detectors(X_train_normal, config):
    """Train anomaly detection models (unsupervised)"""
    print("\n" + "="*70)
    print("TRAINING ANOMALY DETECTION MODELS")
    print("="*70)
    
    models = {}
    
    # Isolation Forest (Fast)
    print("\n1. Training Isolation Forest...")
    iso_config = config['models']['isolation_forest']
    iso = IsolationForestDetector(
        n_estimators=iso_config['n_estimators'],
        contamination=0.2,  # Improved: increased from 0.1 to match 20% attack rate
        max_samples=iso_config['max_samples']
    )
    iso.train(X_train_normal.values)
    iso.save('models/isolation_forest.pkl')
    models['isolation_forest'] = iso
    
    # DBSCAN Clustering (Fast)
    print("\n2. Training DBSCAN Clustering...")
    print("Using optimized hyperparameters from tuning results...")
    # Best parameters from tuning: eps=2.0, min_samples=5
    # Provides good balance: 91.3% recall, 63% accuracy, 16.2% precision
    dbscan = DBSCANAnomalyDetector(eps=2.0, min_samples=5, auto_tune=False)
    
    # Train with optimized parameters
    dbscan.fit(X_train_normal.values)
    dbscan.save_model('models/dbscan.pkl')
    models['dbscan'] = dbscan
    print(f"DBSCAN trained with optimal parameters: eps={dbscan.eps}, min_samples={dbscan.min_samples}")
    print(f"Identified {dbscan.n_clusters_} clusters, noise ratio: {dbscan.noise_ratio_:.2%}")
    
    # K-Means Clustering (Fast)
    print("\n3. Training K-Means Clustering...")
    kmeans_config = config['models'].get('kmeans', {'n_clusters': 8, 'contamination': 0.15})
    kmeans = KMeansDetector(
        n_clusters=kmeans_config.get('n_clusters', 8),
        contamination=kmeans_config.get('contamination', 0.15)
    )
    kmeans.train(X_train_normal.values)
    kmeans.save('models/kmeans.pkl')
    models['kmeans'] = kmeans
    print(f"K-Means trained with {kmeans.n_clusters} clusters")
    
    # GMM (Gaussian Mixture Model) (Fast)
    print("\n4. Training Gaussian Mixture Model...")
    gmm_config = config['models'].get('gmm', {'n_components': 8, 'contamination': 0.15})
    gmm = GMMDetector(
        n_components=gmm_config.get('n_components', 8),
        contamination=gmm_config.get('contamination', 0.15),
        covariance_type=gmm_config.get('covariance_type', 'full')
    )
    gmm.train(X_train_normal.values)
    gmm.save('models/gmm.pkl')
    models['gmm'] = gmm
    print(f"GMM trained with {gmm.n_components} components")
    
    # One-Class SVM (Slower)
    print("\n5. Training One-Class SVM...")
    svm_config = config['models']['one_class_svm']
    svm = OneClassSVMDetector(
        kernel=svm_config['kernel'],
        nu=0.15,  # Improved: increased from 0.1 to 0.15 for better sensitivity
        gamma=svm_config['gamma']
    )
    svm.train(X_train_normal.values)
    svm.save('models/one_class_svm.pkl')
    models['one_class_svm'] = svm
    
    # Autoencoder (Slowest - Deep Learning)
    print("\n6. Training Autoencoder (this may take a few minutes)...")
    ae_config = config['models']['autoencoder']
    ae = AutoencoderDetector(
        input_dim=X_train_normal.shape[1],
        encoding_dim=ae_config['encoding_dim'],
        hidden_layers=ae_config['hidden_layers']
    )
    ae.train(
        X_train_normal.values,
        epochs=ae_config['epochs'],
        batch_size=ae_config['batch_size']
    )
    ae.save('models/autoencoder')
    models['autoencoder'] = ae
    
    return models

def train_supervised_models(X_train, y_train, X_val, y_val, df_train, config):
    """Train supervised models"""
    print("\n" + "="*70)
    print("TRAINING SUPERVISED MODELS")
    print("="*70)
    
    models = {}
    
    # Random Forest (Fast)
    print("\n1. Training Random Forest...")
    rf_config = config['models']['random_forest']
    rf = RandomForestDetector(
        n_estimators=rf_config['n_estimators'],
        max_depth=rf_config['max_depth'],
        min_samples_split=rf_config['min_samples_split']
    )
    rf.train(X_train.values, y_train.values)
    rf.save('models/random_forest.pkl')
    models['random_forest'] = rf
    
    # Print feature importance
    feature_importance = rf.get_feature_importance(X_train.columns.tolist())
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    feature_importance.to_csv('outputs/reports/feature_importance.csv', index=False)
    
    # Gradient Boosting (Moderate speed)
    print("\n2. Training Gradient Boosting...")
    gb_config = config['models'].get('gradient_boosting', {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'min_samples_split': 5
    })
    gb = GradientBoostingDetector(
        n_estimators=gb_config.get('n_estimators', 100),
        learning_rate=gb_config.get('learning_rate', 0.1),
        max_depth=gb_config.get('max_depth', 5),
        min_samples_split=gb_config.get('min_samples_split', 5)
    )
    gb.train(X_train.values, y_train.values)
    gb.save('models/gradient_boosting.pkl')
    models['gradient_boosting'] = gb
    
    # Print feature importance
    gb_importance = gb.get_feature_importance(X_train.columns.tolist())
    print("\nGradient Boosting - Top 10 Most Important Features:")
    print(gb_importance.head(10))
    
    # LSTM removed - too resource intensive and has compatibility issues
    print("\n3. LSTM Sequence Model skipped (removed to reduce training time)")
    
    return models

def create_ensemble(all_models, config):
    """Create ensemble from trained models"""
    print("\n" + "="*70)
    print("CREATING ENSEMBLE")
    print("="*70)
    
    ensemble_config = config['ensemble']
    ensemble = EnsembleDetector(
        voting_strategy=ensemble_config['voting_strategy'],
        threshold=ensemble_config['threshold']
    )
    
    # Add models with weights
    weights = ensemble_config['weights']
    for model_name, model in all_models.items():
        weight = weights.get(model_name, 1.0)
        ensemble.add_model(model_name, model, weight=weight)
    
    ensemble.save('models/ensemble_config.pkl')
    
    return ensemble

def evaluate_models(ensemble, X_test, y_test, config):
    """Evaluate all models"""
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)
    
    metrics = ensemble.evaluate(X_test.values, y_test.values)
    
    print(f"\nEnsemble Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    
    print(f"\nConfusion Matrix:")
    cm = np.array(metrics['confusion_matrix'])
    print(f"  TN: {cm[0][0]:5d}  FP: {cm[0][1]:5d}")
    print(f"  FN: {cm[1][0]:5d}  TP: {cm[1][1]:5d}")
    
    print(f"\nIndividual Model Performance:")
    for model_name, model_metrics in metrics['individual_models'].items():
        print(f"\n  {model_name}:")
        print(f"    Accuracy:  {model_metrics['accuracy']:.4f}")
        print(f"    Precision: {model_metrics['precision']:.4f}")
        print(f"    Recall:    {model_metrics['recall']:.4f}")
        print(f"    F1 Score:  {model_metrics['f1_score']:.4f}")
    
    # Save metrics
    import json
    with open('outputs/reports/evaluation_metrics.json', 'w') as f:
        # Convert numpy types to native Python types
        metrics_serializable = {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1_score']),
            'roc_auc': float(metrics['roc_auc']),
            'confusion_matrix': [[int(x) for x in row] for row in metrics['confusion_matrix']],
            'individual_models': {
                k: {kk: float(vv) for kk, vv in v.items()}
                for k, v in metrics['individual_models'].items()
            }
        }
        json.dump(metrics_serializable, f, indent=2)
    
    return metrics

def main():
    """Main training pipeline"""
    print("="*70)
    print("SESSION TOKEN ABUSE DETECTION - TRAINING PIPELINE")
    print("="*70)
    
    # Load configuration
    config = load_config('config.yaml')
    
    # Generate/load data
    df = generate_or_load_data(config)
    
    # Prepare features
    X, y, df_features, engineer = prepare_features(df, config)
    
    # Split data
    print("\nSplitting data...")
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Second split: separate train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )
    
    # Get corresponding df rows
    df_train = df_features.loc[X_train.index]
    df_val = df_features.loc[X_val.index]
    df_test = df_features.loc[X_test.index]
    
    print(f"Train set: {len(X_train)} samples ({y_train.sum()} attacks)")
    print(f"Val set:   {len(X_val)} samples ({y_val.sum()} attacks)")
    print(f"Test set:  {len(X_test)} samples ({y_test.sum()} attacks)")
    
    # For anomaly detection models, use only normal data
    X_train_normal = X_train[y_train == 0]
    print(f"Normal training samples for anomaly detectors: {len(X_train_normal)}")
    
    # Train anomaly detectors
    anomaly_models = train_anomaly_detectors(X_train_normal, config)
    
    # Train supervised models
    supervised_models = train_supervised_models(
        X_train, y_train, X_val, y_val, df_train, config
    )
    
    # Combine all models
    all_models = {**anomaly_models, **supervised_models}
    
    # Create ensemble
    ensemble = create_ensemble(all_models, config)
    
    # Evaluate
    metrics = evaluate_models(ensemble, X_test, y_test, config)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print("\nModels saved to 'models/' directory")
    print("Evaluation results saved to 'outputs/reports/' directory")
    print("\nYou can now use the inference pipeline to detect attacks in real-time!")

if __name__ == "__main__":
    main()
