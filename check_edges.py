"""
Check why edges aren't being displayed in the map.
"""

import networkx as nx
import pandas as pd

def check_network_edges():
    """
    Analyze the network edges to see why they're not showing up.
    """
    print("Loading network to check edges...")
    network = nx.read_graphml('efficient_sandhill_crane_migration_network_2018.graphml')
    
    print(f"Total network: {network.number_of_nodes()} nodes, {network.number_of_edges()} edges")
    
    # Get node data
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
    
    # Filter to significant counties (same as in map)
    significant_counties = nodes_df[nodes_df['total_cranes'] >= 50].copy()
    print(f"Significant counties (50+ cranes): {len(significant_counties)}")
    
    # Create subgraph
    subgraph = network.subgraph(significant_counties['county_code'])
    print(f"Subgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")
    
    # Check edge weights
    edge_weights = [data['weight'] for _, _, data in subgraph.edges(data=True)]
    print(f"\nEdge weight statistics:")
    print(f"  Min: {min(edge_weights):.6f}")
    print(f"  Max: {max(edge_weights):.6f}")
    print(f"  Mean: {sum(edge_weights)/len(edge_weights):.6f}")
    print(f"  Median: {sorted(edge_weights)[len(edge_weights)//2]:.6f}")
    
    # Check how many edges meet the threshold
    threshold = 0.05
    strong_edges = [w for w in edge_weights if w > threshold]
    print(f"\nEdges with weight > {threshold}: {len(strong_edges)} out of {len(edge_weights)}")
    
    # Show strongest edges
    print(f"\nTop 10 strongest edges:")
    edges_with_weights = [(source, target, data['weight']) for source, target, data in subgraph.edges(data=True)]
    top_edges = sorted(edges_with_weights, key=lambda x: x[2], reverse=True)[:10]
    
    for i, (source, target, weight) in enumerate(top_edges, 1):
        print(f"{i:2d}. {source} -> {target}: weight={weight:.6f}")
    
    # Check if edges exist between significant counties
    print(f"\nChecking connectivity between significant counties...")
    connected_pairs = 0
    for i, county1 in enumerate(significant_counties['county_code']):
        for j, county2 in enumerate(significant_counties['county_code']):
            if i < j:  # Avoid duplicates
                if subgraph.has_edge(county1, county2):
                    weight = subgraph.edges[county1, county2]['weight']
                    if weight > threshold:
                        connected_pairs += 1
    
    print(f"Connected pairs with weight > {threshold}: {connected_pairs}")
    
    # Check geographic distribution
    print(f"\nGeographic distribution of significant counties:")
    print(f"  Latitude range: {significant_counties['lat'].min():.2f} to {significant_counties['lat'].max():.2f}")
    print(f"  Longitude range: {significant_counties['lon'].min():.2f} to {significant_counties['lon'].max():.2f}")
    
    # Check if counties are clustered geographically
    print(f"\nTop counties by crane count:")
    top_counties = significant_counties.nlargest(10, 'total_cranes')
    for _, row in top_counties.iterrows():
        print(f"  {row['county_code']}: {row['total_cranes']:.0f} cranes at ({row['lat']:.2f}, {row['lon']:.2f})")

if __name__ == "__main__":
    check_network_edges()
