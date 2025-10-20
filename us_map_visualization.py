"""
Interactive US map visualization for sandhill crane migration network using Folium.
"""

import networkx as nx
import pandas as pd
import folium
from folium import plugins
import numpy as np

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

def create_us_map(network, nodes_df):
    """
    Create an interactive US map with migration patterns using Folium.
    """
    print("Creating interactive US map with migration patterns...")
    
    # Filter to significant counties (50+ cranes for better visibility)
    significant_counties = nodes_df[nodes_df['total_cranes'] >= 50].copy()
    print(f"Showing {len(significant_counties)} counties with 50+ cranes")
    
    # Create base map centered on US
    m = folium.Map(
        location=[39.8283, -98.5795],  # Center of US
        zoom_start=4,
        tiles='OpenStreetMap'
    )
    
    # Add markers for significant counties
    for _, row in significant_counties.iterrows():
        # Determine marker color based on crane count
        if row['total_cranes'] > 1000:
            color = 'red'
        elif row['total_cranes'] > 500:
            color = 'orange'
        elif row['total_cranes'] > 200:
            color = 'yellow'
        else:
            color = 'green'
        
        # Determine marker size
        size = max(5, min(20, row['total_cranes'] / 100))
        
        # Create popup text
        popup_text = f"""
        <b>County:</b> {row['county_code']}<br>
        <b>Cranes:</b> {row['total_cranes']:.0f}<br>
        <b>Observers:</b> {row['num_observers']}<br>
        <b>Location:</b> {row['lat']:.3f}, {row['lon']:.3f}
        """
        
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=size,
            popup=folium.Popup(popup_text, max_width=200),
            color=color,
            fill=True,
            fillOpacity=0.7,
            weight=2
        ).add_to(m)
    
    # Draw migration routes
    subgraph = network.subgraph(significant_counties['county_code'])
    
    # Get edges sorted by weight (strongest first)
    edges_with_weights = [(source, target, data['weight']) for source, target, data in subgraph.edges(data=True)]
    sorted_edges = sorted(edges_with_weights, key=lambda x: x[2], reverse=True)
    
    # Draw all edges (no limit)
    for source, target, weight in sorted_edges:
        source_data = significant_counties[significant_counties['county_code'] == source]
        target_data = significant_counties[significant_counties['county_code'] == target]
        
        if not source_data.empty and not target_data.empty:
            if weight > 0.01:  # Lower threshold to show more connections
                folium.PolyLine(
                    locations=[[source_data['lat'].iloc[0], source_data['lon'].iloc[0]],
                              [target_data['lat'].iloc[0], target_data['lon'].iloc[0]]],
                    color='blue',
                    weight=weight*5,
                    opacity=0.4,
                    popup=f"Migration route: {source} → {target}<br>Weight: {weight:.3f}"
                ).add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 200px; height: 120px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p><b>Sandhill Crane Migration Network</b></p>
    <p><i class="fa fa-circle" style="color:red"></i> >1000 cranes</p>
    <p><i class="fa fa-circle" style="color:orange"></i> 500-1000 cranes</p>
    <p><i class="fa fa-circle" style="color:yellow"></i> 200-500 cranes</p>
    <p><i class="fa fa-circle" style="color:green"></i> 50-200 cranes</p>
    <p><i class="fa fa-minus" style="color:blue"></i> Migration routes</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map
    m.save('interactive_migration_network_map.html')
    print("Interactive map saved as 'interactive_migration_network_map.html'")
    print("Open the HTML file in your browser to view the interactive map!")
    
    return m

def show_network_stats(network, nodes_df):
    """
    Show network statistics.
    """
    print(f"\n=== NETWORK STATISTICS ===")
    print(f"Total nodes: {network.number_of_nodes()}")
    print(f"Total edges: {network.number_of_edges()}")
    print(f"Average degree: {2 * network.number_of_edges() / network.number_of_nodes():.2f}")
    
    # Geographic coverage
    print(f"\nGeographic coverage:")
    print(f"  Latitude range: {nodes_df['lat'].min():.2f} to {nodes_df['lat'].max():.2f}")
    print(f"  Longitude range: {nodes_df['lon'].min():.2f} to {nodes_df['lon'].max():.2f}")
    
    # Top counties
    top_counties = nodes_df.nlargest(10, 'total_cranes')
    print(f"\nTop 10 counties by crane count:")
    for i, (_, row) in enumerate(top_counties.iterrows(), 1):
        print(f"{i:2d}. {row['county_code']}: {row['total_cranes']:.0f} cranes")

def main():
    """
    Main function.
    """
    network, nodes_df = load_network_data()
    show_network_stats(network, nodes_df)
    create_us_map(network, nodes_df)

if __name__ == "__main__":
    main()
