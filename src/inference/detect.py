"""
Real-time Inference and Detection Pipeline
"""
import pandas as pd
import numpy as np
import sys
import os
import joblib
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from preprocessing.feature_engineering import FeatureEngineer
from models.anomaly_detectors import (
    IsolationForestDetector,
    OneClassSVMDetector,
    AutoencoderDetector,
    KMeansDetector,
    GMMDetector
)
from models.behavioral_models import (
    RandomForestDetector,
    LSTMSequenceDetector,
    GradientBoostingDetector
)
from models.ensemble import EnsembleDetector, AttackDetectionPipeline

class RealTimeDetector:
    """Real-time session attack detector"""
    
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.ensemble = None
        self.feature_engineer = None
        self.pipeline = None
        self.load_models()
        
    def load_models(self):
        """Load all trained models"""
        print("Loading trained models...")
        
        # Load feature engineer
        self.feature_engineer = joblib.load(f'{self.models_dir}/feature_engineer.pkl')
        print("✓ Feature engineer loaded")
        
        # Create ensemble
        self.ensemble = EnsembleDetector()
        
        # Load anomaly detectors
        try:
            iso = IsolationForestDetector()
            iso.load(f'{self.models_dir}/isolation_forest.pkl')
            self.ensemble.add_model('isolation_forest', iso, weight=0.2)
            print("✓ Isolation Forest loaded")
        except Exception as e:
            print(f"✗ Could not load Isolation Forest: {e}")
        
        try:
            svm = OneClassSVMDetector()
            svm.load(f'{self.models_dir}/one_class_svm.pkl')
            self.ensemble.add_model('one_class_svm', svm, weight=0.15)
            print("✓ One-Class SVM loaded")
        except Exception as e:
            print(f"✗ Could not load One-Class SVM: {e}")
        
        try:
            ae = AutoencoderDetector(input_dim=1)  # Will be updated on load
            ae.load(f'{self.models_dir}/autoencoder')
            self.ensemble.add_model('autoencoder', ae, weight=0.25)
            print("✓ Autoencoder loaded")
        except Exception as e:
            print(f"✗ Could not load Autoencoder: {e}")
        
        # Load supervised models
        try:
            rf = RandomForestDetector()
            rf.load(f'{self.models_dir}/random_forest.pkl')
            self.ensemble.add_model('random_forest', rf, weight=0.4)
            print("✓ Random Forest loaded")
        except Exception as e:
            print(f"✗ Could not load Random Forest: {e}")
        
        try:
            gb = GradientBoostingDetector()
            gb.load(f'{self.models_dir}/gradient_boosting.pkl')
            self.ensemble.add_model('gradient_boosting', gb, weight=0.4)
            print("✓ Gradient Boosting loaded")
        except Exception as e:
            print(f"✗ Could not load Gradient Boosting: {e}")
        
        # Load clustering models
        try:
            kmeans = KMeansDetector()
            kmeans.load(f'{self.models_dir}/kmeans.pkl')
            self.ensemble.add_model('kmeans', kmeans, weight=0.2)
            print("✓ K-Means loaded")
        except Exception as e:
            print(f"✗ Could not load K-Means: {e}")
        
        try:
            gmm = GMMDetector()
            gmm.load(f'{self.models_dir}/gmm.pkl')
            self.ensemble.add_model('gmm', gmm, weight=0.25)
            print("✓ GMM loaded")
        except Exception as e:
            print(f"✗ Could not load GMM: {e}")
        
        # Load DBSCAN
        try:
            from models.dbscan_detector import DBSCANAnomalyDetector
            dbscan = DBSCANAnomalyDetector()
            dbscan.load_model(f'{self.models_dir}/dbscan_detector.pkl')
            self.ensemble.add_model('dbscan', dbscan, weight=0.15)
            print("✓ DBSCAN loaded")
        except Exception as e:
            print(f"✗ Could not load DBSCAN: {e}")
        
        # LSTM is optional due to complexity
        try:
            lstm = LSTMSequenceDetector(input_dim=1, sequence_length=20)
            lstm.load(f'{self.models_dir}/lstm_sequence')
            self.ensemble.add_model('lstm', lstm, weight=0.25)
            print("✓ LSTM loaded")
        except Exception as e:
            print(f"  (LSTM not loaded - this is optional)")
        
        # Load ensemble configuration
        try:
            self.ensemble.load_config(f'{self.models_dir}/ensemble_config.pkl')
            print("✓ Ensemble configuration loaded")
        except Exception as e:
            print(f"  Using default ensemble configuration")
        
        # Create pipeline
        self.pipeline = AttackDetectionPipeline(self.ensemble, self.feature_engineer)
        
        print("\nModels loaded successfully!")
        print(f"Active models: {list(self.ensemble.models.keys())}")
        
    def detect(self, session_logs_df, return_details=True):
        """
        Detect attacks in session logs
        
        Args:
            session_logs_df: DataFrame with session events
            return_details: Include detailed explanations
        
        Returns:
            results_df: DataFrame with detection results
        """
        print(f"\nAnalyzing {len(session_logs_df)} session events...")
        
        results_df = self.pipeline.detect_attacks(session_logs_df, return_details)
        
        # Count detections
        num_attacks = (results_df['prediction'] == 1).sum()
        attack_rate = num_attacks / len(results_df) * 100
        
        print(f"Detection complete:")
        print(f"  Total events: {len(results_df)}")
        print(f"  Attacks detected: {num_attacks} ({attack_rate:.2f}%)")
        
        return results_df
    
    def generate_alerts(self, results_df):
        """Generate alerts for detected attacks"""
        alerts = []
        
        for idx, row in results_df.iterrows():
            if row['prediction'] == 1:
                alert = self.pipeline.generate_alert(row.to_dict())
                if alert:
                    alerts.append(alert)
        
        return alerts
    
    def save_results(self, results_df, output_path):
        """Save detection results"""
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    
    def generate_report(self, results_df, alerts, output_path):
        """Generate detection report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_events': len(results_df),
            'attacks_detected': int((results_df['prediction'] == 1).sum()),
            'attack_rate': float((results_df['prediction'] == 1).mean()),
            'unique_sessions': int(results_df['session_id'].nunique()),
            'affected_users': results_df[results_df['prediction'] == 1]['user_id'].nunique(),
            'alerts': alerts
        }
        
        # Attack breakdown by type
        if 'attack_type' in results_df.columns:
            attack_types = results_df[results_df['prediction'] == 1]['attack_type'].value_counts()
            report['attack_type_breakdown'] = attack_types.to_dict()
        
        # Risk level distribution
        if 'risk_level' in results_df.columns:
            risk_levels = results_df[results_df['prediction'] == 1]['risk_level'].value_counts()
            report['risk_level_distribution'] = risk_levels.to_dict()
        
        # Save report
        import json
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Report saved to {output_path}")
        return report


def detect_from_csv(input_csv, output_dir='outputs/detection'):
    """Detect attacks from CSV file"""
    print("="*70)
    print("SESSION TOKEN ABUSE DETECTION - INFERENCE")
    print("="*70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"\nLoading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} events from {df['session_id'].nunique()} sessions")
    
    # Initialize detector
    detector = RealTimeDetector()
    
    # Detect attacks
    results_df = detector.detect(df, return_details=True)
    
    # Generate alerts
    alerts = detector.generate_alerts(results_df)
    
    print(f"\nGenerated {len(alerts)} alerts")
    if alerts:
        print("\nSample alerts:")
        for i, alert in enumerate(alerts[:3]):
            print(f"\nAlert {i+1}:")
            print(f"  Session: {alert['session_id']}")
            print(f"  User: {alert['user_id']}")
            print(f"  Risk Level: {alert['risk_level']}")
            print(f"  Action: {alert['recommended_action']}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"{output_dir}/detection_results_{timestamp}.csv"
    detector.save_results(results_df, results_path)
    
    # Generate report
    report_path = f"{output_dir}/detection_report_{timestamp}.json"
    report = detector.generate_report(results_df, alerts, report_path)
    
    print("\n" + "="*70)
    print("DETECTION COMPLETE")
    print("="*70)
    print(f"\nSummary:")
    print(f"  Total Events: {report['total_events']}")
    print(f"  Attacks Detected: {report['attacks_detected']}")
    print(f"  Attack Rate: {report['attack_rate']*100:.2f}%")
    print(f"  Affected Users: {report['affected_users']}")
    print(f"\nResults saved to: {results_path}")
    print(f"Report saved to: {report_path}")
    
    return results_df, alerts, report


def detect_realtime_event(event_dict, detector=None):
    """Detect attack in a single event (streaming mode)"""
    if detector is None:
        detector = RealTimeDetector()
    
    # Convert single event to DataFrame
    df = pd.DataFrame([event_dict])
    
    # Detect
    result = detector.detect(df, return_details=True)
    
    return result.iloc[0].to_dict()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Session Token Abuse Detection - Inference')
    parser.add_argument('--input', type=str, default='data/raw/session_logs.csv',
                       help='Input CSV file with session logs')
    parser.add_argument('--output', type=str, default='outputs/detection',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Run detection
    detect_from_csv(args.input, args.output)
