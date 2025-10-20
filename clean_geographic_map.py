"""
Clean geographic map for sandhill crane migration using geopandas.
Requires: geopandas, matplotlib
"""

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import contextily as ctx

def load_network_data():
    """
    Load the network and extract node data with coordinates.
    """
    print("Loading efficient migration network...")
    network = nx.read_graphml('efficient_sandhill_crane_migration_network_2018.graphml')
    print(f"Network loaded: {network.number_of_nodes()} nodes, {network.number_of_edges()} edges")
    
    # Extract node data
    nodes_data = []
    for node, data in network.nodes(data=True):
        nodes_data.append({
            'county_code': node,
            'lat': data.get('lat', 0),
            'lon': data.get('lon', 0),
            'total_cranes': data.get('total_cranes', 0),
            'num_observers': data.get('num_observers', 0)
        })
    
    nodes_df = pd.DataFrame(nodes_data)
    return network, nodes_df

def create_geographic_map(network, nodes_df):
    """
    Create a geographic map showing migration patterns.
    """
    print("Creating geographic migration map...")
    
    # Filter to significant counties (100+ cranes)
    significant_counties = nodes_df[nodes_df['total_cranes'] >= 100].copy()
    print(f"Showing {len(significant_counties)} counties with 100+ cranes")
    
    # Create GeoDataFrame
    geometry = [Point(xy) for xy in zip(significant_counties['lon'], significant_counties['lat'])]
    gdf = gpd.GeoDataFrame(significant_counties, geometry=geometry)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Plot counties
    gdf.plot(ax=ax, 
             markersize=gdf['total_cranes']/50,  # Size by crane count
             c=gdf['total_cranes'],              # Color by crane count
             cmap='Reds', 
             alpha=0.8, 
             edgecolor='black', 
             linewidth=0.5)
    
    # Add labels for top counties
    top_counties = significant_counties.nlargest(10, 'total_cranes')
    for _, row in top_counties.iterrows():
        ax.annotate(f"{row['county_code']}\n{row['total_cranes']:.0f}", 
                   (row['lon'], row['lat']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Draw major migration routes
    subgraph = network.subgraph(significant_counties['county_code'])
    route_count = 0
    for edge in subgraph.edges():
        if route_count > 20:  # Limit for readability
            break
        
        source_data = significant_counties[significant_counties['county_code'] == edge[0]]
        target_data = significant_counties[significant_counties['county_code'] == edge[1]]
        
        if not source_data.empty and not target_data.empty:
            weight = subgraph.edges[edge].get('weight', 0)
            if weight > 0.1:  # Only strong connections
                ax.plot([source_data['lon'].iloc[0], target_data['lon'].iloc[0]],
                       [source_data['lat'].iloc[0], target_data['lat'].iloc[0]],
                       'blue', alpha=0.6, linewidth=weight*3)
                route_count += 1
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Sandhill Crane Migration Network 2018\n(Size = Crane count, Color = Crane abundance)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('clean_migration_network_map.png', dpi=300, bbox_inches='tight')
    print("Geographic map saved as 'clean_migration_network_map.png'")
    pass

def show_major_sites(nodes_df):
    """
    Show major stopover sites.
    """
    print("\n=== MAJOR STOPOVER SITES ===")
    top_counties = nodes_df.nlargest(15, 'total_cranes')
    
    print("Rank | County Code        | Cranes | Observers | Lat      | Lon")
    print("-" * 70)
    
    for i, (_, row) in enumerate(top_counties.iterrows(), 1):
        print(f"{i:4d} | {row['county_code']:18s} | {row['total_cranes']:6.0f} | {row['num_observers']:9d} | {row['lat']:8.2f} | {row['lon']:8.2f}")
    pass

def main():
    """
    Main function.
    """
    network, nodes_df = load_network_data()
    show_major_sites(nodes_df)
    create_geographic_map(network, nodes_df)

if __name__ == "__main__":
    main()
