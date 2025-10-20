"""
Efficient Sandhill Crane Migration Network Transformation V2
===========================================================

This version uses precomputed distance tables to avoid recalculating
distances for each year's network construction.
"""

import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
import warnings
from efficient_distance_table_builder import DistanceTableBuilder

warnings.filterwarnings('ignore')

class EfficientCraneNetworkBuilderV2:
    """
    Builds a migration network from sandhill crane observation data.
    Uses precomputed distance tables for efficiency.
    """
    
    def __init__(self, max_distance_km=500, max_time_days=14, distance_table_file='county_distance_table.pkl'):
        """
        Initialize the network builder.
        """
        self.max_distance_km = max_distance_km
        self.max_time_days = max_time_days
        self.network = nx.DiGraph()
        
        # Load distance table
        self.distance_builder = DistanceTableBuilder()
        if not self.distance_builder.load_distance_table(distance_table_file):
            print("Warning: Could not load distance table. Building new one...")
            self.distance_builder.build_distance_table(max_distance_km)
            self.distance_builder.save_distance_table(distance_table_file)
        
    def load_and_filter_data(self, parquet_file):
        """
        Load data and filter out zero-crane observations immediately.
        """
        print(f"Loading data from {parquet_file}...")
        df = pd.read_parquet(parquet_file)
        
        # Filter for sandhill crane observations only
        if 'scientific_name' in df.columns:
            df = df[df['scientific_name'] == 'Antigone canadensis']
        
        print(f"Total observations before filtering: {len(df):,}")
        
        # CRITICAL: Filter out zero-crane observations immediately
        df = df[df['observation_count'] > 0]
        print(f"Observations with cranes: {len(df):,}")
        
        # Convert date columns if needed
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df
    
    def aggregate_by_county(self, df):
        """
        Aggregate observations by FIPS county code.
        Now all counties will have >0 cranes.
        """
        print("Aggregating observations by county...")
        
        # Group by county and aggregate
        county_data = df.groupby('county_code').agg({
            'observation_count': 'sum',           # Total cranes per county
            'latitude': 'mean',                   # County center coordinates
            'longitude': 'mean',
            'observation_date': ['min', 'max'],   # First and last observation dates
            'observer_id': 'nunique',             # Number of unique observers
            'checklist_id': 'nunique'             # Number of checklists
        }).reset_index()
        
        # Flatten column names
        county_data.columns = [
            'county_code', 'total_cranes', 'lat', 'lon', 
            'first_observation', 'last_observation', 
            'num_observers', 'num_checklists'
        ]
        
        # Calculate observation duration
        county_data['observation_duration'] = (
            county_data['last_observation'] - county_data['first_observation']
        ).dt.days
        
        print(f"Aggregated to {len(county_data)} counties with cranes")
        return county_data
    
    def create_network_edges(self, county_data):
        """
        Create network edges based on temporal and geographic proximity.
        Uses precomputed distances for efficiency.
        """
        print("Creating network edges using precomputed distances...")
        
        n_counties = len(county_data)
        edges_created = 0
        
        for i in range(n_counties):
            county_i = county_data.iloc[i]
            
            for j in range(n_counties):
                if i == j:
                    continue
                    
                county_j = county_data.iloc[j]
                
                # Get distance from precomputed table
                distance = self.distance_builder.get_distance(
                    county_i['county_code'], 
                    county_j['county_code']
                )
                
                # Apply geographic constraint
                if distance > self.max_distance_km:
                    continue
                
                # Calculate temporal proximity
                time_gap = abs((county_i['first_observation'] - county_j['first_observation']).days)
                
                # Apply temporal constraint
                if time_gap > self.max_time_days:
                    continue
                
                # Calculate edge weight based on proportional flow
                total_cranes_i = county_i['total_cranes']
                total_cranes_j = county_j['total_cranes']
                
                # Proportional flow (conservation of mass)
                proportion_i = total_cranes_i / (total_cranes_i + total_cranes_j)
                proportion_j = total_cranes_j / (total_cranes_i + total_cranes_j)
                
                # Temporal decay factor
                temporal_decay = np.exp(-time_gap / self.max_time_days)
                
                # Distance decay factor
                distance_decay = np.exp(-distance / self.max_distance_km)
                
                # Edge weight
                edge_weight = min(proportion_i, proportion_j) * temporal_decay * distance_decay
                
                # Add edge to network
                self.network.add_edge(
                    county_i['county_code'],
                    county_j['county_code'],
                    weight=edge_weight,
                    distance=distance,
                    time_gap=time_gap,
                    flow_direction=f"{county_i['county_code']} -> {county_j['county_code']}"
                )
                
                edges_created += 1
        
        print(f"Created {edges_created} edges")
    
    def add_node_attributes(self, county_data):
        """
        Add node attributes to the network.
        """
        print("Adding node attributes...")
        
        for _, county in county_data.iterrows():
            self.network.add_node(
                county['county_code'],
                total_cranes=county['total_cranes'],
                lat=county['lat'],
                lon=county['lon'],
                num_observers=county['num_observers'],
                num_checklists=county['num_checklists'],
                observation_duration=county['observation_duration']
            )
    
    def build_network(self, parquet_file):
        """
        Build the complete migration network.
        """
        print("Building Efficient Sandhill Crane Migration Network V2...")
        print("=" * 60)
        
        # Load and filter data (removes zero-crane observations)
        df = self.load_and_filter_data(parquet_file)
        
        # Aggregate by county
        county_data = self.aggregate_by_county(df)
        
        # Create network edges (using precomputed distances)
        self.create_network_edges(county_data)
        
        # Add node attributes
        self.add_node_attributes(county_data)
        
        print("=" * 60)
        print(f"Network construction complete!")
        print(f"Nodes: {self.network.number_of_nodes()}")
        print(f"Edges: {self.network.number_of_edges()}")
        
        return self.network
    
    def save_network(self, output_file):
        """
        Save the network to a file.
        """
        print(f"Saving network to {output_file}...")
        
        # Save as GraphML for network analysis tools
        nx.write_graphml(self.network, output_file)
        
        # Also save as CSV for easy inspection
        edges_df = nx.to_pandas_edgelist(self.network)
        edges_df.to_csv(output_file.replace('.graphml', '_edges.csv'), index=False)
        
        nodes_df = pd.DataFrame.from_dict(dict(self.network.nodes(data=True)), orient='index')
        nodes_df.to_csv(output_file.replace('.graphml', '_nodes.csv'))
        
        print("Network saved successfully!")

def main():
    """
    Main function to build the efficient migration network.
    """
    # Initialize network builder
    builder = EfficientCraneNetworkBuilderV2(
        max_distance_km=500,    # 500km maximum migration distance
        max_time_days=14      # 14 days maximum time gap
    )
    
    # Build network for 2018 data
    network = builder.build_network('Antigone canadensis_all_checklist_2018.parquet')
    
    # Save network
    builder.save_network('efficient_sandhill_crane_migration_network_2018_v2.graphml')
    
    # Print network statistics
    print("\nEfficient Network Statistics:")
    print(f"Number of nodes: {network.number_of_nodes()}")
    print(f"Number of edges: {network.number_of_edges()}")
    print(f"Average degree: {2 * network.number_of_edges() / network.number_of_nodes():.2f}")
    
    # Find most connected counties
    degree_centrality = nx.degree_centrality(network)
    top_counties = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print("\nTop 10 Most Connected Counties (with cranes):")
    for county, centrality in top_counties:
        node_data = network.nodes[county]
        total_cranes = node_data.get('total_cranes', 0)
        print(f"  {county}: centrality={centrality:.3f}, cranes={total_cranes}")

if __name__ == "__main__":
    main()
