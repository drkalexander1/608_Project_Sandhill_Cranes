import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

class NodeGenerator:
    """
    Base class/Interface for node generation strategies.
    """
    def generate_nodes(self, df):
        """
        Generate nodes from the dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe with lat/lon/observation_count.
            
        Returns:
            tuple: (nodes_df, labeled_df)
                   nodes_df: DataFrame with node attributes (id, lat, lon, total_cranes, etc.)
                   labeled_df: Input df with an added 'node_id' column.
        """
        raise NotImplementedError

class CountyNodeGenerator(NodeGenerator):
    """
    Generates nodes based on FIPS county codes (standard approach).
    """
    def generate_nodes(self, df):
        print("Generating nodes based on Counties...")
        
        # Ensure county_code exists
        if 'county_code' not in df.columns:
            raise ValueError("county_code column missing for CountyNodeGenerator")
            
        labeled_df = df.copy()
        labeled_df['node_id'] = labeled_df['county_code']
        
        # Aggregate
        nodes_df = labeled_df.groupby('node_id').agg({
            'observation_count': 'sum',
            'latitude': 'mean',
            'longitude': 'mean',
            'observation_date': ['min', 'max'],
            'observer_id': 'nunique',
            'checklist_id': 'nunique'
        }).reset_index()
        
        # Flatten columns
        nodes_df.columns = [
            'node_id', 'total_cranes', 'lat', 'lon', 
            'first_observation', 'last_observation', 
            'num_observers', 'num_checklists'
        ]
        
        nodes_df['observation_duration'] = (
            nodes_df['last_observation'] - nodes_df['first_observation']
        ).dt.days
        
        print(f"Generated {len(nodes_df)} county-based nodes.")
        return nodes_df, labeled_df

class ClusterNodeGenerator(NodeGenerator):
    """
    Generates nodes using DBSCAN clustering on county-level aggregated data.
    This preserves more connections by clustering counties rather than raw observations.
    """
    def __init__(self, eps_km=100, min_samples=3):
        """
        Args:
            eps_km (float): Maximum distance between counties to cluster (larger = more aggregation)
            min_samples (int): Minimum counties needed to form a cluster (lower = more clusters)
        """
        self.eps_km = eps_km
        self.min_samples = min_samples
        
    def generate_nodes(self, df):
        print(f"Generating nodes using DBSCAN Clustering on counties (eps={self.eps_km}km)...")
        
        if df.empty:
            return pd.DataFrame(), df
        
        # First, aggregate by county (like CountyNodeGenerator does)
        if 'county_code' not in df.columns:
            raise ValueError("county_code column missing for ClusterNodeGenerator")
        
        # Aggregate observations by county first
        county_agg = df.groupby('county_code').agg({
            'observation_count': 'sum',
            'latitude': 'mean',
            'longitude': 'mean',
            'observation_date': ['min', 'max'],
            'observer_id': 'nunique',
            'checklist_id': 'nunique'
        }).reset_index()
        
        county_agg.columns = [
            'county_code', 'total_cranes', 'lat', 'lon', 
            'first_observation', 'last_observation', 
            'num_observers', 'num_checklists'
        ]
        
        print(f"Aggregated to {len(county_agg)} counties before clustering")
        
        # Now cluster the counties (not individual observations)
        coords = county_agg[['lat', 'lon']].values
        
        # Convert km to radians for haversine metric
        kms_per_radian = 6371.0088
        eps_rad = self.eps_km / kms_per_radian
        
        db = DBSCAN(eps=eps_rad, min_samples=self.min_samples, metric='haversine', algorithm='ball_tree')
        coords_rad = np.radians(coords)
        labels = db.fit_predict(coords_rad)
        
        county_agg['cluster_label'] = labels
        
        noise_count = (labels == -1).sum()
        print(f"DBSCAN found {len(set(labels)) - (1 if -1 in labels else 0)} clusters. Noise counties: {noise_count}")
        
        # Assign noise counties to their own clusters (don't drop them!)
        # This preserves all data
        if noise_count > 0:
            max_cluster = max([l for l in labels if l != -1]) if len(set(labels)) > 1 else -1
            noise_mask = labels == -1
            county_agg.loc[noise_mask, 'cluster_label'] = range(max_cluster + 1, max_cluster + 1 + noise_count.sum())
        
        county_agg['node_id'] = county_agg['cluster_label'].apply(lambda x: f"cluster_{x}")
        
        # Aggregate clusters to create final nodes
        nodes_df = county_agg.groupby('node_id').agg({
            'total_cranes': 'sum',
            'lat': 'mean',  # Centroid of cluster
            'lon': 'mean',
            'first_observation': 'min',
            'last_observation': 'max',
            'num_observers': 'sum',
            'num_checklists': 'sum'
        }).reset_index()
        
        nodes_df['observation_duration'] = (
            nodes_df['last_observation'] - nodes_df['first_observation']
        ).dt.days
        
        # Create labeled dataframe mapping counties to clusters
        labeled_df = df.copy()
        county_to_cluster = dict(zip(county_agg['county_code'], county_agg['node_id']))
        labeled_df['node_id'] = labeled_df['county_code'].map(county_to_cluster)
        
        print(f"Generated {len(nodes_df)} cluster-based nodes from {len(county_agg)} counties.")
        return nodes_df, labeled_df

