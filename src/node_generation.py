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
    Generates nodes using DBSCAN clustering on spatial coordinates.
    """
    def __init__(self, eps_km=30, min_samples=5):
        """
        Args:
            eps_km (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
            min_samples (int): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
        """
        self.eps_km = eps_km
        self.min_samples = min_samples
        
    def generate_nodes(self, df):
        print(f"Generating nodes using DBSCAN Clustering (eps={self.eps_km}km)...")
        
        if df.empty:
            return pd.DataFrame(), df
            
        coords = df[['latitude', 'longitude']].values
        
        # Convert km to radians for haversine metric
        kms_per_radian = 6371.0088
        eps_rad = self.eps_km / kms_per_radian
        
        # Run DBSCAN
        # We use haversine metric for geospatial data
        # ideally we should weigh by observation_count, but standard DBSCAN doesn't support weighted samples easily in all versions
        # We'll cluster the *checklists* (locations). 
        # To weigh it, we could repeat rows, but that's expensive.
        # Alternatively, we cluster the unique locations and then aggregate.
        
        # For now, let's cluster the observation points directly.
        # Note: If there are many points at the exact same location, they will be clustered together.
        
        db = DBSCAN(eps=eps_rad, min_samples=self.min_samples, metric='haversine', algorithm='ball_tree')
        
        # We need radians for haversine
        coords_rad = np.radians(coords)
        
        labels = db.fit_predict(coords_rad)
        
        labeled_df = df.copy()
        labeled_df['cluster_label'] = labels
        
        # Filter out noise (-1)
        # Option: Assign noise to nearest cluster? Or drop? 
        # Dropping noise means losing birds. Let's keep them as separate "noise" nodes or drop them?
        # For migration network, noise might be irrelevant sporadic sightings.
        # Let's drop noise for the node definition, but maybe we report it.
        
        noise_count = (labels == -1).sum()
        print(f"DBSCAN found {len(set(labels)) - (1 if -1 in labels else 0)} clusters. Noise points: {noise_count}")
        
        # Remove noise for node generation
        valid_df = labeled_df[labeled_df['cluster_label'] != -1].copy()
        
        if valid_df.empty:
             print("Warning: No clusters found. Try increasing eps_km or decreasing min_samples.")
             return pd.DataFrame(), labeled_df
             
        valid_df['node_id'] = valid_df['cluster_label'].apply(lambda x: f"cluster_{x}")
        
        # Aggregate to create node properties
        nodes_df = valid_df.groupby('node_id').agg({
            'observation_count': 'sum',
            'latitude': 'mean',
            'longitude': 'mean',
            'observation_date': ['min', 'max'],
            'observer_id': 'nunique',
            'checklist_id': 'nunique'
        }).reset_index()
        
        nodes_df.columns = [
            'node_id', 'total_cranes', 'lat', 'lon', 
            'first_observation', 'last_observation', 
            'num_observers', 'num_checklists'
        ]
        
        nodes_df['observation_duration'] = (
            nodes_df['last_observation'] - nodes_df['first_observation']
        ).dt.days
        
        # Update the labeled_df to have 'node_id' for valid ones, NaN for noise
        mask = labeled_df['cluster_label'] != -1
        labeled_df.loc[mask, 'node_id'] = labeled_df.loc[mask, 'cluster_label'].apply(lambda x: f"cluster_{x}")
        
        print(f"Generated {len(nodes_df)} cluster-based nodes.")
        return nodes_df, labeled_df

