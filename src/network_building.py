import networkx as nx
import pandas as pd
import numpy as np
from datetime import datetime

class NetworkBuilder:
    """
    Builds a directed graph from node data.
    """
    
    def __init__(self, max_distance_km=500, max_time_days=14):
        self.max_distance_km = max_distance_km
        self.max_time_days = max_time_days
        self.network = nx.DiGraph()
        
    def build_network(self, nodes_df):
        """
        Construct the network from nodes.
        
        Args:
            nodes_df (pd.DataFrame): Dataframe with node_id, lat, lon, first_observation, total_cranes.
            
        Returns:
            nx.DiGraph: The constructed network.
        """
        print("Building network structure...")
        self.network = nx.DiGraph()
        
        # Add nodes
        print(f"Adding {len(nodes_df)} nodes...")
        for _, row in nodes_df.iterrows():
            self.network.add_node(
                row['node_id'],
                **row.to_dict()
            )
            
        # Create edges
        print("Creating edges...")
        self._create_edges_vectorized(nodes_df)
        
        print(f"Network built: {self.network.number_of_nodes()} nodes, {self.network.number_of_edges()} edges.")
        return self.network

    def _create_edges_vectorized(self, nodes_df):
        """
        Vectorized approach to create edges. 
        Calculates all pairwise distances and time gaps, then filters.
        """
        n_nodes = len(nodes_df)
        if n_nodes == 0:
            return

        # Extract numpy arrays
        ids = nodes_df['node_id'].values
        lats = nodes_df['lat'].values
        lons = nodes_df['lon'].values
        times = nodes_df['first_observation'].values
        counts = nodes_df['total_cranes'].values
        
        # 1. Calculate Spatial Distances (Haversine Approximation)
        # Convert to radians
        lats_rad = np.radians(lats)
        lons_rad = np.radians(lons)
        
        # Broadcasting to get pairwise differences
        # shape: (n, n)
        dlat = lats_rad[:, np.newaxis] - lats_rad
        dlon = lons_rad[:, np.newaxis] - lons_rad
        
        # Haversine formula
        a = np.sin(dlat / 2)**2 + np.cos(lats_rad) * np.cos(lats_rad[:, np.newaxis]) * np.sin(dlon / 2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth radius ~6371 km
        dist_matrix = c * 6371.0
        
        # 2. Calculate Time Gaps (Days)
        # times is array of numpy.datetime64
        # shape: (n, n)
        time_matrix_days = (times[:, np.newaxis] - times).astype('timedelta64[D]').astype(float)
        # We want absolute time gap? Or directional?
        # Migration moves forward in time. So we probably want target time > source time.
        # But original code used abs(time_gap) and then checked flow based on counts?
        # Let's check original code:
        # time_gap = abs((county_i['first_observation'] - county_j['first_observation']).days)
        # It used abs.
        
        abs_time_matrix = np.abs(time_matrix_days)
        
        # 3. Identify Valid Edges
        # Constraints
        valid_dist = (dist_matrix <= self.max_distance_km) & (dist_matrix > 0) # >0 to avoid self-loops
        valid_time = abs_time_matrix <= self.max_time_days
        
        valid_edges_mask = valid_dist & valid_time
        
        # Indices of valid edges
        source_indices, target_indices = np.where(valid_edges_mask)
        
        # Iterate and add edges
        # This loop is much smaller than N*N
        print(f"Processing {len(source_indices)} potential edges...")
        
        for src_idx, tgt_idx in zip(source_indices, target_indices):
            src_id = ids[src_idx]
            tgt_id = ids[tgt_idx]
            
            dist = dist_matrix[src_idx, tgt_idx]
            time_gap = abs_time_matrix[src_idx, tgt_idx]
            
            # Calculate weights (ported from original code)
            total_cranes_i = counts[src_idx]
            total_cranes_j = counts[tgt_idx]
            
            denom = total_cranes_i + total_cranes_j
            if denom == 0:
                proportion_i = 0
                proportion_j = 0
            else:
                proportion_i = total_cranes_i / denom
                proportion_j = total_cranes_j / denom
            
            temporal_decay = np.exp(-time_gap / self.max_time_days)
            distance_decay = np.exp(-dist / self.max_distance_km)
            
            edge_weight = min(proportion_i, proportion_j) * temporal_decay * distance_decay
            
            self.network.add_edge(
                src_id,
                tgt_id,
                weight=edge_weight,
                distance=dist,
                time_gap=time_gap,
                flow_direction=f"{src_id} -> {tgt_id}"
            )

