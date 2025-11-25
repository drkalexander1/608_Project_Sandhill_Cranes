import folium
import matplotlib.pyplot as plt
import contextily as ctx
import networkx as nx
import os
from matplotlib.patches import Rectangle

class Visualizer:
    """
    Handles visualization of migration networks and analysis results.
    """
    
    @staticmethod
    def create_interactive_map(network, output_file, title="Migration Network"):
        """
        Create an interactive Folium map.
        """
        print(f"Creating interactive map: {output_file}...")
        
        # Center map (US approximate center)
        m = folium.Map(location=[39.8283, -98.5795], zoom_start=4, tiles='OpenStreetMap')
        
        # Add title
        title_html = f'''
             <h3 align="center" style="font-size:16px"><b>{title}</b></h3>
             '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Draw edges first
        for u, v, data in network.edges(data=True):
            weight = data.get('weight', 0)
            if weight > 0.01: # Threshold
                pos_u = (network.nodes[u]['lat'], network.nodes[u]['lon'])
                pos_v = (network.nodes[v]['lat'], network.nodes[v]['lon'])
                
                folium.PolyLine(
                    locations=[pos_u, pos_v],
                    color='blue',
                    weight=weight * 5,
                    opacity=0.5
                ).add_to(m)
                
        # Draw nodes
        for node, data in network.nodes(data=True):
            cranes = data.get('total_cranes', 0)
            radius = max(3, min(20, cranes / 100))
            
            color = 'green'
            if cranes > 1000: color = 'red'
            elif cranes > 500: color = 'orange'
            elif cranes > 200: color = 'yellow'
            
            folium.CircleMarker(
                location=[data['lat'], data['lon']],
                radius=radius,
                color=color,
                fill=True,
                fill_opacity=0.7,
                tooltip=f"Node: {node}<br>Cranes: {cranes}"
            ).add_to(m)
            
        m.save(output_file)
        print(f"Saved to {output_file}")

    @staticmethod
    def plot_min_cut(original_network, cut_edges, output_file, title="Min Cut Analysis"):
        """
        Plot the network with min-cut edges highlighted.
        """
        print(f"Plotting Min-Cut analysis to {output_file}...")
        
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.set_xlim(-125, -66)
        ax.set_ylim(24, 50)
        ax.set_aspect('equal')
        
        try:
            ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.OpenStreetMap.Mapnik)
        except:
            ax.set_facecolor('lightblue')
            ax.add_patch(Rectangle((-125, 24), 59, 26, facecolor='lightgreen', alpha=0.3))
            
        ax.set_title(title, fontsize=16)
        
        # Create a set of cut edges for fast lookup (u_out, v_in) -> (u, v)
        # The cut edges are from the flow network: u_in/u_out -> v_in/v_out
        # We need to map back to original nodes.
        
        # However, the cut might be on the internal edge (u_in, u_out) -> Node Bottleneck
        # or on the travel edge (u_out, v_in) -> Edge Bottleneck
        
        # Let's iterate and classify
        node_bottlenecks = []
        edge_bottlenecks = []
        
        for u_flow, v_flow in cut_edges:
            # Check if it's a node split edge: "X_in" -> "X_out"
            if u_flow.endswith('_in') and v_flow.endswith('_out') and u_flow[:-3] == v_flow[:-4]:
                node_bottlenecks.append(u_flow[:-3])
            # Check if it's a travel edge: "X_out" -> "Y_in"
            elif u_flow.endswith('_out') and v_flow.endswith('_in'):
                u_orig = u_flow[:-4]
                v_orig = v_flow[:-3]
                edge_bottlenecks.append((u_orig, v_orig))
                
        # Draw normal edges
        for u, v, data in original_network.edges(data=True):
            pos_u = (original_network.nodes[u]['lon'], original_network.nodes[u]['lat'])
            pos_v = (original_network.nodes[v]['lon'], original_network.nodes[v]['lat'])
            
            # Check if this is a bottleneck edge
            if (u, v) in edge_bottlenecks:
                ax.plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], color='red', linewidth=3, zorder=5, label='Bottleneck Edge')
            else:
                ax.plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], color='gray', alpha=0.3, linewidth=1)
                
        # Draw nodes
        for node, data in original_network.nodes(data=True):
            pos = (data['lon'], data['lat'])
            
            if node in node_bottlenecks:
                ax.scatter(pos[0], pos[1], c='red', s=100, zorder=10, edgecolors='black', label='Bottleneck Node')
            else:
                ax.scatter(pos[0], pos[1], c='blue', s=20, alpha=0.6, zorder=6)
                
        # Unique legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

