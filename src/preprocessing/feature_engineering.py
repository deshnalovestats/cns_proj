"""
Feature Engineering for Session Token Abuse Detection
Extracts relevant features from session logs
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from geopy.distance import geodesic
import hashlib

class FeatureEngineer:
    def __init__(self):
        self.feature_names = []
        
    def extract_temporal_features(self, df):
        """Extract time-based features"""
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by session and time
        df = df.sort_values(['session_id', 'timestamp'])
        
        # Time since session start
        df['time_since_session_start'] = df.groupby('session_id')['timestamp'].transform(
            lambda x: (x - x.min()).dt.total_seconds()
        )
        
        # Time between actions
        df['time_since_last_action'] = df.groupby('session_id')['timestamp'].diff().dt.total_seconds()
        df['time_since_last_action'] = df['time_since_last_action'].fillna(0)
        
        # Hour of day, day of week
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 6)).astype(int)
        
        return df
    
    def extract_ip_features(self, df):
        """Extract IP-based features"""
        df = df.copy()
        
        # Count IP changes within session
        df['ip_changes_in_session'] = df.groupby('session_id')['ip_address'].transform(
            lambda x: x.nunique() - 1
        )
        
        # IP changed from previous action
        df['ip_changed'] = (
            df.groupby('session_id')['ip_address'].shift(1) != df['ip_address']
        ).astype(int)
        df['ip_changed'] = df.groupby('session_id')['ip_changed'].fillna(0).astype(int)
        
        # Cumulative IP changes
        df['cumulative_ip_changes'] = df.groupby('session_id')['ip_changed'].cumsum()
        
        return df
    
    def extract_geolocation_features(self, df):
        """Extract location-based features"""
        df = df.copy()
        
        # Country changes
        df['country_changes_in_session'] = df.groupby('session_id')['country'].transform(
            lambda x: x.nunique() - 1
        )
        
        # Country changed
        df['country_changed'] = (
            df.groupby('session_id')['country'].shift(1) != df['country']
        ).astype(int)
        df['country_changed'] = df.groupby('session_id')['country_changed'].fillna(0).astype(int)
        
        # Calculate distance from previous location
        def calculate_distance_series(group):
            distances = [0]
            for i in range(1, len(group)):
                prev_loc = (group.iloc[i-1]['latitude'], group.iloc[i-1]['longitude'])
                curr_loc = (group.iloc[i]['latitude'], group.iloc[i]['longitude'])
                try:
                    dist = geodesic(prev_loc, curr_loc).kilometers
                except:
                    dist = 0
                distances.append(dist)
            return distances
        
        df['distance_from_prev_km'] = 0.0
        for session_id, group in df.groupby('session_id'):
            if len(group) > 1:
                distances = calculate_distance_series(group)
                df.loc[group.index, 'distance_from_prev_km'] = distances
        
        # Impossible travel detection
        df['travel_speed_kmh'] = 0.0
        mask = df['time_since_last_action'] > 0
        df.loc[mask, 'travel_speed_kmh'] = (
            df.loc[mask, 'distance_from_prev_km'] / 
            (df.loc[mask, 'time_since_last_action'] / 3600)
        )
        
        # Flag impossible travel (> 800 km/h, speed of aircraft)
        df['impossible_travel'] = (df['travel_speed_kmh'] > 800).astype(int)
        
        return df
    
    def extract_device_features(self, df):
        """Extract device and browser features"""
        df = df.copy()
        
        # Device fingerprint changes
        df['device_changes_in_session'] = df.groupby('session_id')['device_fingerprint'].transform(
            lambda x: x.nunique() - 1
        )
        
        # Device changed
        df['device_changed'] = (
            df.groupby('session_id')['device_fingerprint'].shift(1) != df['device_fingerprint']
        ).astype(int)
        df['device_changed'] = df.groupby('session_id')['device_changed'].fillna(0).astype(int)
        
        # User agent changes
        df['ua_changes_in_session'] = df.groupby('session_id')['user_agent'].transform(
            lambda x: x.nunique() - 1
        )
        
        df['ua_changed'] = (
            df.groupby('session_id')['user_agent'].shift(1) != df['user_agent']
        ).astype(int)
        df['ua_changed'] = df.groupby('session_id')['ua_changed'].fillna(0).astype(int)
        
        return df
    
    def extract_behavioral_features(self, df):
        """Extract user behavior patterns"""
        df = df.copy()
        
        # Action counts
        df['action_count_in_session'] = df.groupby('session_id').cumcount() + 1
        
        # Encode action types
        action_encoded = pd.get_dummies(df['action'], prefix='action')
        df = pd.concat([df, action_encoded], axis=1)
        
        # Suspicious action sequences (e.g., change_password, edit_profile without normal browsing)
        sensitive_actions = ['change_password', 'edit_profile', 'payment', 'download_file']
        df['is_sensitive_action'] = df['action'].isin(sensitive_actions).astype(int)
        
        # Ratio of sensitive actions
        df['sensitive_action_ratio'] = df.groupby('session_id')['is_sensitive_action'].transform('mean')
        
        # Action variety (entropy)
        df['action_variety'] = df.groupby('session_id')['action'].transform('nunique')
        
        return df
    
    def extract_session_features(self, df):
        """Extract session-level aggregate features"""
        df = df.copy()
        
        # Session duration
        session_stats = df.groupby('session_id').agg({
            'timestamp': ['min', 'max'],
            'action': 'count',
            'ip_address': 'nunique',
            'country': 'nunique',
            'device_fingerprint': 'nunique',
            'user_agent': 'nunique'
        })
        
        session_stats.columns = ['_'.join(col).strip() for col in session_stats.columns.values]
        session_stats['session_duration_seconds'] = (
            session_stats['timestamp_max'] - session_stats['timestamp_min']
        ).dt.total_seconds()
        
        session_stats = session_stats.reset_index()
        
        # Merge back
        df = df.merge(
            session_stats[['session_id', 'session_duration_seconds', 
                          'ip_address_nunique', 'country_nunique',
                          'device_fingerprint_nunique', 'user_agent_nunique']],
            on='session_id',
            how='left'
        )
        
        return df
    
    def create_sequence_features(self, df, sequence_length=20):
        """Create sequence features for LSTM/Transformer models"""
        df = df.copy()
        df = df.sort_values(['session_id', 'timestamp'])
        
        sequences = []
        labels = []
        
        for session_id, group in df.groupby('session_id'):
            # Create sequences for this session
            session_data = group.to_dict('records')
            
            for i in range(len(session_data)):
                # Get sequence up to current point
                start_idx = max(0, i - sequence_length + 1)
                sequence = session_data[start_idx:i+1]
                
                # Pad if necessary
                while len(sequence) < sequence_length:
                    sequence.insert(0, {k: 0 for k in sequence[0].keys()})
                
                sequences.append(sequence)
                labels.append(session_data[i]['is_attack'])
        
        return sequences, labels
    
    def engineer_features(self, df):
        """Apply all feature engineering steps"""
        print("Starting feature engineering...")
        
        print("Extracting temporal features...")
        df = self.extract_temporal_features(df)
        
        print("Extracting IP features...")
        df = self.extract_ip_features(df)
        
        print("Extracting geolocation features...")
        df = self.extract_geolocation_features(df)
        
        print("Extracting device features...")
        df = self.extract_device_features(df)
        
        print("Extracting behavioral features...")
        df = self.extract_behavioral_features(df)
        
        print("Extracting session features...")
        df = self.extract_session_features(df)
        
        print(f"Feature engineering complete. Total columns: {len(df.columns)}")
        
        return df
    
    def get_ml_features(self, df):
        """Get features suitable for traditional ML models"""
        feature_cols = [
            'time_since_session_start', 'time_since_last_action',
            'hour_of_day', 'day_of_week', 'is_weekend', 'is_night',
            'ip_changes_in_session', 'ip_changed', 'cumulative_ip_changes',
            'country_changes_in_session', 'country_changed',
            'distance_from_prev_km', 'travel_speed_kmh', 'impossible_travel',
            'device_changes_in_session', 'device_changed',
            'ua_changes_in_session', 'ua_changed',
            'action_count_in_session', 'is_sensitive_action',
            'sensitive_action_ratio', 'action_variety',
            'session_duration_seconds', 'ip_address_nunique',
            'country_nunique', 'device_fingerprint_nunique',
            'user_agent_nunique'
        ]
        
        # Add action one-hot encoded columns
        action_cols = [col for col in df.columns if col.startswith('action_')]
        feature_cols.extend(action_cols)
        
        # Filter to existing columns
        available_cols = [col for col in feature_cols if col in df.columns]
        
        return df[available_cols].fillna(0)

if __name__ == "__main__":
    # Load data
    print("Loading data...")
    df = pd.read_csv('data/raw/session_logs.csv')
    
    # Engineer features
    engineer = FeatureEngineer()
    df_features = engineer.engineer_features(df)
    
    # Save processed data
    df_features.to_csv('data/processed/session_logs_features.csv', index=False)
    print("\nProcessed data saved to data/processed/session_logs_features.csv")
    
    # Get ML features
    X = engineer.get_ml_features(df_features)
    y = df_features['is_attack']
    
    print(f"\nML Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Attack rate: {y.mean():.2%}")
    
    print("\nSample features:")
    print(X.head())
