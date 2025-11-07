"""
DBSCAN-based Anomaly Detector for Session Token Abuse Detection

DBSCAN identifies anomalies as noise points that don't belong to any dense cluster.
This is ideal for session abuse detection where attacks are isolated events.
"""

import numpy as np
import pickle
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


class DBSCANAnomalyDetector:
    """
    DBSCAN-based anomaly detector for session token abuse.
    
    Advantages:
    - No assumption about cluster shapes
    - Automatically determines number of clusters
    - Identifies outliers as noise points (-1 label)
    - Works well with varying density clusters
    
    Parameters:
    -----------
    eps : float
        Maximum distance between two samples for one to be considered
        as in the neighborhood of the other. Auto-tuned if None.
    min_samples : int
        Number of samples in a neighborhood for a point to be considered
        as a core point. Typically 2 * n_features.
    metric : str
        Distance metric (default: 'euclidean')
    """
    
    def __init__(self, eps=None, min_samples=None, metric='euclidean', 
                 auto_tune=True):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.auto_tune = auto_tune
        self.scaler = StandardScaler()
        self.dbscan = None
        self.labels_ = None
        self.n_clusters_ = None
        self.noise_ratio_ = None
        
    def _auto_tune_eps(self, X_scaled):
        """
        Auto-tune eps parameter using k-distance graph.
        
        The optimal eps is the "elbow" in the k-distance curve.
        """
        # Use k = min_samples for k-NN
        k = self.min_samples
        
        # Fit k-NN to find distances to k-th nearest neighbor
        nbrs = NearestNeighbors(n_neighbors=k, metric=self.metric)
        nbrs.fit(X_scaled)
        distances, indices = nbrs.kneighbors(X_scaled)
        
        # Sort distances to k-th neighbor
        k_distances = np.sort(distances[:, k-1])
        
        # Find elbow point (maximum curvature)
        # Use simple gradient approach
        gradients = np.gradient(k_distances)
        elbow_idx = np.argmax(gradients)
        optimal_eps = k_distances[elbow_idx]
        
        print(f"Auto-tuned eps: {optimal_eps:.4f}")
        
        # Plot k-distance graph for visualization
        self._plot_k_distance(k_distances, elbow_idx)
        
        return optimal_eps
    
    def _plot_k_distance(self, k_distances, elbow_idx):
        """Plot k-distance graph with elbow point."""
        plt.figure(figsize=(10, 6))
        plt.plot(k_distances, linewidth=2)
        plt.axhline(y=k_distances[elbow_idx], color='r', linestyle='--', 
                   label=f'Optimal eps: {k_distances[elbow_idx]:.4f}')
        plt.xlabel('Data Points (sorted by distance)', fontsize=12)
        plt.ylabel(f'{self.min_samples}-NN Distance', fontsize=12)
        plt.title('K-Distance Graph for DBSCAN eps Selection', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('outputs/visualizations/dbscan_eps_tuning.png', dpi=300)
        plt.close()
    
    def fit(self, X, y=None):
        """
        Fit DBSCAN on normal data (unsupervised).
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data (preferably only normal sessions)
        y : array-like, optional
            Not used (unsupervised), kept for API consistency
        """
        # Store training data for predict_proba
        self.X_train_ = X
        
        # Scale features (DBSCAN is sensitive to scales)
        X_scaled = self.scaler.fit_transform(X)
        
        # Auto-tune parameters if requested
        if self.min_samples is None:
            # Rule of thumb: min_samples = 2 * n_features
            self.min_samples = min(2 * X.shape[1], 10)
            print(f"Auto-set min_samples: {self.min_samples}")
        
        if self.eps is None and self.auto_tune:
            self.eps = self._auto_tune_eps(X_scaled)
        elif self.eps is None:
            # Default heuristic
            self.eps = 0.5
            print(f"Using default eps: {self.eps}")
        
        # Fit DBSCAN
        self.dbscan = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric,
            n_jobs=-1
        )
        
        self.labels_ = self.dbscan.fit_predict(X_scaled)
        
        # Analyze clustering results
        self.n_clusters_ = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        self.noise_ratio_ = np.sum(self.labels_ == -1) / len(self.labels_)
        
        print(f"\nDBSCAN Clustering Results:")
        print(f"  Number of clusters: {self.n_clusters_}")
        print(f"  Noise points: {np.sum(self.labels_ == -1)} ({self.noise_ratio_*100:.2f}%)")
        print(f"  Silhouette score: {self._calculate_silhouette(X_scaled):.4f}")
        
        # Visualize clusters
        self._visualize_clusters(X_scaled)
        
        return self
    
    def _calculate_silhouette(self, X_scaled):
        """Calculate silhouette score (only for clustered points)."""
        # Filter out noise points for silhouette calculation
        mask = self.labels_ != -1
        if np.sum(mask) < 2 or self.n_clusters_ < 2:
            return 0.0
        return silhouette_score(X_scaled[mask], self.labels_[mask])
    
    def _visualize_clusters(self, X_scaled):
        """Visualize DBSCAN clusters using t-SNE."""
        from sklearn.manifold import TSNE
        
        # Use t-SNE for 2D visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_scaled)
        
        plt.figure(figsize=(12, 8))
        
        # Plot each cluster
        unique_labels = set(self.labels_)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                # Noise points (anomalies) in black
                color = 'black'
                marker = 'x'
                label_name = 'Noise (Anomalies)'
            else:
                marker = 'o'
                label_name = f'Cluster {label}'
            
            mask = self.labels_ == label
            plt.scatter(
                X_tsne[mask, 0], X_tsne[mask, 1],
                c=[color], marker=marker, s=100, alpha=0.6,
                edgecolors='k', label=label_name
            )
        
        plt.title('DBSCAN Clustering (t-SNE Visualization)', fontsize=14)
        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('outputs/visualizations/dbscan_clusters.png', dpi=300)
        plt.close()
    
    def predict(self, X):
        """
        Predict anomalies (noise points).
        
        Returns:
        --------
        predictions : array-like
            1 for anomalies (noise), 0 for normal (clustered)
        """
        if self.dbscan is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        
        # Use DBSCAN's fit_predict (refit on new data)
        # Note: DBSCAN doesn't have a traditional predict() method
        # We use the trained parameters to cluster new data
        dbscan_test = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric,
            n_jobs=-1
        )
        labels = dbscan_test.fit_predict(X_scaled)
        
        # Convert labels: -1 (noise) -> 1 (anomaly), others -> 0 (normal)
        predictions = (labels == -1).astype(int)
        
        return predictions
    
    def predict_proba(self, X):
        """
        Predict anomaly probabilities based on distance to nearest cluster.
        
        Returns:
        --------
        probabilities : array-like, shape (n_samples, 2)
            Probability of [normal, anomaly] for each sample
        """
        X_scaled = self.scaler.transform(X)
        
        # Calculate distance to nearest cluster center
        # Approximate cluster centers as mean of core points
        core_sample_indices = self.dbscan.core_sample_indices_
        X_train_scaled = self.scaler.transform(self.X_train_)  # Store in fit()
        
        cluster_centers = []
        for label in range(self.n_clusters_):
            mask = (self.labels_ == label) & np.isin(np.arange(len(self.labels_)), 
                                                     core_sample_indices)
            if np.sum(mask) > 0:
                cluster_centers.append(X_train_scaled[mask].mean(axis=0))
        
        if len(cluster_centers) == 0:
            # No clusters found, return default probabilities
            return np.column_stack([np.ones(len(X)) * 0.5, np.ones(len(X)) * 0.5])
        
        cluster_centers = np.array(cluster_centers)
        
        # Calculate minimum distance to any cluster center
        from scipy.spatial.distance import cdist
        distances = cdist(X_scaled, cluster_centers, metric=self.metric)
        min_distances = distances.min(axis=1)
        
        # Convert distance to probability (sigmoid transformation)
        # Points far from clusters have high anomaly probability
        anomaly_proba = 1 / (1 + np.exp(-5 * (min_distances - self.eps)))
        normal_proba = 1 - anomaly_proba
        
        return np.column_stack([normal_proba, anomaly_proba])
    
    def save_model(self, filepath):
        """Save trained model to disk."""
        model_data = {
            'eps': self.eps,
            'min_samples': self.min_samples,
            'metric': self.metric,
            'scaler': self.scaler,
            'dbscan': self.dbscan,
            'labels': self.labels_,
            'n_clusters': self.n_clusters_,
            'noise_ratio': self.noise_ratio_,
            'X_train': self.X_train_
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"DBSCAN model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """Load trained model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        detector = cls(
            eps=model_data['eps'],
            min_samples=model_data['min_samples'],
            metric=model_data['metric'],
            auto_tune=False
        )
        detector.scaler = model_data['scaler']
        detector.dbscan = model_data['dbscan']
        detector.labels_ = model_data['labels']
        detector.n_clusters_ = model_data['n_clusters']
        detector.noise_ratio_ = model_data['noise_ratio']
        detector.X_train_ = model_data.get('X_train', None)
        
        print(f"DBSCAN model loaded from {filepath}")
        return detector


def train_dbscan_detector(X_train, y_train, save_path='models/dbscan_detector.pkl'):
    """
    Train DBSCAN anomaly detector.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels (only normal data will be used)
    save_path : str
        Path to save trained model
    
    Returns:
    --------
    detector : DBSCANAnomalyDetector
        Trained DBSCAN detector
    """
    print("\n" + "="*60)
    print("Training DBSCAN Anomaly Detector")
    print("="*60)
    
    # Train only on normal data (unsupervised anomaly detection)
    X_normal = X_train[y_train == 0]
    
    print(f"Training on {len(X_normal)} normal sessions...")
    
    # Initialize and train detector with auto-tuning
    detector = DBSCANAnomalyDetector(
        eps=None,  # Auto-tune
        min_samples=None,  # Auto-tune
        metric='euclidean',
        auto_tune=True
    )
    
    detector.fit(X_normal)
    
    # Save model
    detector.save_model(save_path)
    
    return detector


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    # Load processed features
    df = pd.read_csv('data/processed/session_logs_features.csv')
    
    # Prepare data
    feature_cols = [col for col in df.columns if col not in 
                   ['session_id', 'user_id', 'timestamp', 'is_attack', 'attack_type']]
    X = df[feature_cols].values
    y = df['is_attack'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Train DBSCAN detector
    detector = train_dbscan_detector(X_train, y_train)
    
    # Evaluate
    from sklearn.metrics import classification_report, confusion_matrix
    
    y_pred = detector.predict(X_test)
    
    print("\n" + "="*60)
    print("DBSCAN Evaluation Results")
    print("="*60)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                               target_names=['Normal', 'Attack']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
