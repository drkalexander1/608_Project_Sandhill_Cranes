import folium
import matplotlib.pyplot as plt
import contextily as ctx
import networkx as nx
import pandas as pd
import numpy as np
import os
from matplotlib.patches import Rectangle

class Visualizer:
    """
    Handles visualization of migration networks and analysis results.
    """
    
    @staticmethod
    def create_interactive_map(network, output_file, title="Migration Network", min_cranes=50):
        """
        Create an interactive Folium map with filtering for significant nodes.
        
        Args:
            network: NetworkX graph
            output_file: Output file path
            title: Map title
            min_cranes: Minimum crane count to display node (default 50)
        """
        print(f"Creating interactive map: {output_file}...")
        
        # Filter to significant nodes (like original code)
        significant_nodes = {
            n: d for n, d in network.nodes(data=True) 
            if d.get('total_cranes', 0) >= min_cranes
        }
        
        if not significant_nodes:
            print(f"Warning: No nodes with {min_cranes}+ cranes found. Showing all nodes.")
            significant_nodes = dict(network.nodes(data=True))
        
        # Center map (US approximate center)
        m = folium.Map(location=[39.8283, -98.5795], zoom_start=4, tiles='OpenStreetMap')
        
        # Add title
        title_html = f'''
             <h3 align="center" style="font-size:16px"><b>{title}</b></h3>
             '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Create subgraph for significant nodes
        subgraph = network.subgraph(significant_nodes.keys())
        
        # Get edges sorted by weight (strongest first)
        edges_with_weights = [
            (source, target, data.get('weight', 0)) 
            for source, target, data in subgraph.edges(data=True)
        ]
        sorted_edges = sorted(edges_with_weights, key=lambda x: x[2], reverse=True)
        
        # Draw edges first (so they appear behind nodes)
        for source, target, weight in sorted_edges:
            if weight > 0.01:  # Threshold
                source_data = significant_nodes[source]
                target_data = significant_nodes[target]
                
                folium.PolyLine(
                    locations=[
                        [source_data['lat'], source_data['lon']],
                        [target_data['lat'], target_data['lon']]
                    ],
                    color='blue',
                    weight=max(1, min(5, weight * 10)),  # Better scaling
                    opacity=0.4
                ).add_to(m)
                
        # Draw nodes
        for node, data in significant_nodes.items():
            cranes = data.get('total_cranes', 0)
            radius = max(5, min(20, cranes / 100))  # Better size scaling
            
            color = 'green'
            if cranes > 1000: 
                color = 'red'
            elif cranes > 500: 
                color = 'orange'
            elif cranes > 200: 
                color = 'yellow'
            
            folium.CircleMarker(
                location=[data['lat'], data['lon']],
                radius=radius,
                color=color,
                fill=True,
                fill_opacity=0.7,
                weight=2,
                tooltip=f"Node: {node}<br>Cranes: {int(cranes)}"
            ).add_to(m)
        
        # Add legend
        legend_html = f'''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: 140px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>{title}</b></p>
        <p><i class="fa fa-circle" style="color:red"></i> >1000 cranes</p>
        <p><i class="fa fa-circle" style="color:orange"></i> 500-1000 cranes</p>
        <p><i class="fa fa-circle" style="color:yellow"></i> 200-500 cranes</p>
        <p><i class="fa fa-circle" style="color:green"></i> {min_cranes}-200 cranes</p>
        <p><i class="fa fa-minus" style="color:blue"></i> Migration routes</p>
        <p>Nodes: {len(significant_nodes)}</p>
        <p>Edges: {subgraph.number_of_edges()}</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
            
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
    
    @staticmethod
    def plot_flow_intensity(summary_data, output_dir="results/main"):
        """
        Plot flow rate intensity over time as separate bar charts.
        Creates 3 separate PNG files.
        
        Args:
            summary_data: List of dicts with 'year', 'season', 'max_flow', etc.
            output_dir: Output directory path
        """
        print(f"Creating flow intensity plots in {output_dir}...")
        
        if not summary_data:
            print("No data to plot")
            return
        
        df = pd.DataFrame(summary_data)
        
        # Filter out cluster node types - only keep county
        if 'node_type' in df.columns:
            df = df[df['node_type'] == 'county'].copy()
            print(f"Filtered to county node type: {len(df)} entries")
        
        # Aggregate by year/season/node_type if needed
        if 'component' in df.columns:
            # Sum across components for each year/season/node_type
            df = df.groupby(['year', 'season', 'node_type']).agg({
                'max_flow': 'sum',
                'nodes': 'first',
                'edges': 'first'
            }).reset_index()
        
        # Calculate flow rate metrics
        df['flow_rate_per_node'] = df['max_flow'] / df['nodes'].replace(0, np.nan)
        df['flow_rate_per_edge'] = df['max_flow'] / df['edges'].replace(0, np.nan)
        
        years = sorted(df['year'].unique())
        x = np.arange(len(years))
        width = 0.35
        
        # Filter to cluster nodes if available (they have flow data), otherwise use county
        plot_df = df.copy()
        if 'node_type' in plot_df.columns:
            if len(plot_df[plot_df['node_type'] == 'cluster']) > 0:
                plot_df = plot_df[plot_df['node_type'] == 'cluster'].copy()
                print("Using cluster node data for flow plots (has flow data)")
            elif len(plot_df[plot_df['node_type'] == 'county']) > 0:
                plot_df = plot_df[plot_df['node_type'] == 'county'].copy()
                print("Using county node data for flow plots (cluster data not available)")
        
        def get_values_filtered(season, metric):
            values = []
            for y in years:
                data = plot_df[(plot_df['year'] == y) & (plot_df['season'] == season)]
                if len(data) > 0:
                    values.append(data[metric].sum() if metric in ['max_flow'] else data[metric].mean())
                else:
                    values.append(0)
            return values
        
        # Plot 1: Max Flow
        fig, ax = plt.subplots(figsize=(12, 6))
        pre_flows = get_values_filtered('PreBreeding', 'max_flow')
        post_flows = get_values_filtered('PostBreeding', 'max_flow')
        bars1 = ax.bar(x - width/2, pre_flows, width, label='PreBreeding', alpha=0.8, color='steelblue')
        bars2 = ax.bar(x + width/2, post_flows, width, label='PostBreeding', alpha=0.8, color='coral')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}',
                            ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Max Flow (cranes)', fontsize=12)
        ax.set_title('Flow Rate Intensity Over Time', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(years)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_file = os.path.join(output_dir, "flow_intensity_max_flow.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved to {output_file}")
        
        # Plot 2: Flow Rate per Node
        fig, ax = plt.subplots(figsize=(12, 6))
        pre_rate_node = get_values_filtered('PreBreeding', 'flow_rate_per_node')
        post_rate_node = get_values_filtered('PostBreeding', 'flow_rate_per_node')
        bars1 = ax.bar(x - width/2, pre_rate_node, width, label='PreBreeding', alpha=0.8, color='steelblue')
        bars2 = ax.bar(x + width/2, post_rate_node, width, label='PostBreeding', alpha=0.8, color='coral')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2f}',
                            ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Flow Rate per Node (cranes/node)', fontsize=12)
        ax.set_title('Flow Rate Efficiency: Flow per Node', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(years)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        output_file = os.path.join(output_dir, "flow_intensity_per_node.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved to {output_file}")
        
        # Plot 3: Flow Rate per Edge
        fig, ax = plt.subplots(figsize=(12, 6))
        pre_rate_edge = get_values_filtered('PreBreeding', 'flow_rate_per_edge')
        post_rate_edge = get_values_filtered('PostBreeding', 'flow_rate_per_edge')
        bars1 = ax.bar(x - width/2, pre_rate_edge, width, label='PreBreeding', alpha=0.8, color='steelblue')
        bars2 = ax.bar(x + width/2, post_rate_edge, width, label='PostBreeding', alpha=0.8, color='coral')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.4f}',
                            ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Flow Rate per Edge (cranes/edge)', fontsize=12)
        ax.set_title('Flow Rate Efficiency: Flow per Edge', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(years)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        output_file = os.path.join(output_dir, "flow_intensity_per_edge.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved to {output_file}")
        
        # Create combined plot with all 3 metrics
        fig, axes = plt.subplots(3, 1, figsize=(14, 14))
        
        # Plot 1: Max Flow
        ax1 = axes[0]
        bars1 = ax1.bar(x - width/2, pre_flows, width, label='PreBreeding', alpha=0.8, color='steelblue')
        bars2 = ax1.bar(x + width/2, post_flows, width, label='PostBreeding', alpha=0.8, color='coral')
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}',
                            ha='center', va='bottom', fontsize=9)
        ax1.set_xlabel('Year', fontsize=12)
        ax1.set_ylabel('Max Flow (cranes)', fontsize=12)
        ax1.set_title('Flow Rate Intensity Over Time', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(years)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Flow Rate per Node
        ax2 = axes[1]
        bars1 = ax2.bar(x - width/2, pre_rate_node, width, label='PreBreeding', alpha=0.8, color='steelblue')
        bars2 = ax2.bar(x + width/2, post_rate_node, width, label='PostBreeding', alpha=0.8, color='coral')
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2f}',
                            ha='center', va='bottom', fontsize=9)
        ax2.set_xlabel('Year', fontsize=12)
        ax2.set_ylabel('Flow Rate per Node (cranes/node)', fontsize=12)
        ax2.set_title('Flow Rate Efficiency: Flow per Node', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(years)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Flow Rate per Edge
        ax3 = axes[2]
        bars1 = ax3.bar(x - width/2, pre_rate_edge, width, label='PreBreeding', alpha=0.8, color='steelblue')
        bars2 = ax3.bar(x + width/2, post_rate_edge, width, label='PostBreeding', alpha=0.8, color='coral')
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax3.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.4f}',
                            ha='center', va='bottom', fontsize=9)
        ax3.set_xlabel('Year', fontsize=12)
        ax3.set_ylabel('Flow Rate per Edge (cranes/edge)', fontsize=12)
        ax3.set_title('Flow Rate Efficiency: Flow per Edge', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(years)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_file = os.path.join(output_dir, "flow_intensity_combined.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved combined plot to {output_file}")
    
    @staticmethod
    def plot_network_density(summary_data, output_dir="results/main"):
        """
        Plot network density metrics over time as separate bar charts.
        Creates 4 separate PNG files.
        
        Args:
            summary_data: List of dicts with 'year', 'season', 'nodes', 'edges', 'density', etc.
            output_dir: Output directory path
        """
        print(f"Creating network density plots in {output_dir}...")
        
        if not summary_data:
            print("No data to plot")
            return
        
        df = pd.DataFrame(summary_data)
        
        # Filter out cluster node types - only keep county
        if 'node_type' in df.columns:
            df = df[df['node_type'] == 'county'].copy()
            print(f"Filtered to county node type: {len(df)} entries")
        
        # Aggregate by year/season/node_type if needed
        if 'component' in df.columns:
            # For density metrics, we want to aggregate properly
            # Density should be recalculated from aggregated nodes/edges
            df = df.groupby(['year', 'season', 'node_type']).agg({
                'nodes': 'first',  # Same across components
                'edges': 'first',  # Same across components
            }).reset_index()
        
        # Calculate density metrics if not present
        if 'density' not in df.columns:
            df['density'] = df['edges'] / (df['nodes'] * (df['nodes'] - 1)).replace(0, np.nan)
        if 'avg_degree' not in df.columns:
            df['avg_degree'] = (2 * df['edges']) / df['nodes'].replace(0, np.nan)
        
        years = sorted(df['year'].unique())
        x = np.arange(len(years))
        width = 0.35
        
        # Helper function to get values for a season
        def get_values(season, metric):
            values = []
            for y in years:
                data = df[(df['year'] == y) & (df['season'] == season)]
                if len(data) > 0:
                    # Average across node_types if multiple
                    values.append(data[metric].mean())
                else:
                    values.append(0)
            return values
        
        # Plot 1: Network Density
        fig, ax = plt.subplots(figsize=(12, 6))
        pre_density = get_values('PreBreeding', 'density')
        post_density = get_values('PostBreeding', 'density')
        bars1 = ax.bar(x - width/2, pre_density, width, label='PreBreeding', alpha=0.8, color='steelblue')
        bars2 = ax.bar(x + width/2, post_density, width, label='PostBreeding', alpha=0.8, color='coral')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Network Density', fontsize=12)
        ax.set_title('Network Density Over Time', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(years)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        output_file = os.path.join(output_dir, "network_density.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved to {output_file}")
        
        # Plot 2: Average Degree
        fig, ax = plt.subplots(figsize=(12, 6))
        pre_degree = get_values('PreBreeding', 'avg_degree')
        post_degree = get_values('PostBreeding', 'avg_degree')
        bars1 = ax.bar(x - width/2, pre_degree, width, label='PreBreeding', alpha=0.8, color='steelblue')
        bars2 = ax.bar(x + width/2, post_degree, width, label='PostBreeding', alpha=0.8, color='coral')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Average Degree', fontsize=12)
        ax.set_title('Average Node Degree Over Time', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(years)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        output_file = os.path.join(output_dir, "network_avg_degree.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved to {output_file}")
        
        # Plot 3: Number of Nodes
        fig, ax = plt.subplots(figsize=(12, 6))
        pre_nodes = get_values('PreBreeding', 'nodes')
        post_nodes = get_values('PostBreeding', 'nodes')
        bars1 = ax.bar(x - width/2, pre_nodes, width, label='PreBreeding', alpha=0.8, color='steelblue')
        bars2 = ax.bar(x + width/2, post_nodes, width, label='PostBreeding', alpha=0.8, color='coral')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Number of Nodes', fontsize=12)
        ax.set_title('Network Size (Nodes) Over Time', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(years)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        output_file = os.path.join(output_dir, "network_num_nodes.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved to {output_file}")
        
        # Plot 4: Number of Edges
        fig, ax = plt.subplots(figsize=(12, 6))
        pre_edges = get_values('PreBreeding', 'edges')
        post_edges = get_values('PostBreeding', 'edges')
        bars1 = ax.bar(x - width/2, pre_edges, width, label='PreBreeding', alpha=0.8, color='steelblue')
        bars2 = ax.bar(x + width/2, post_edges, width, label='PostBreeding', alpha=0.8, color='coral')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Number of Edges', fontsize=12)
        ax.set_title('Network Connectivity (Edges) Over Time', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(years)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        output_file = os.path.join(output_dir, "network_num_edges.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved to {output_file}")
        
        # Create combined plot with all 4 metrics
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Network Density
        ax1 = axes[0, 0]
        bars1 = ax1.bar(x - width/2, pre_density, width, label='PreBreeding', alpha=0.8, color='steelblue')
        bars2 = ax1.bar(x + width/2, post_density, width, label='PostBreeding', alpha=0.8, color='coral')
        ax1.set_xlabel('Year', fontsize=11)
        ax1.set_ylabel('Network Density', fontsize=11)
        ax1.set_title('Network Density Over Time', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(years)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Average Degree
        ax2 = axes[0, 1]
        bars1 = ax2.bar(x - width/2, pre_degree, width, label='PreBreeding', alpha=0.8, color='steelblue')
        bars2 = ax2.bar(x + width/2, post_degree, width, label='PostBreeding', alpha=0.8, color='coral')
        ax2.set_xlabel('Year', fontsize=11)
        ax2.set_ylabel('Average Degree', fontsize=11)
        ax2.set_title('Average Node Degree Over Time', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(years)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Number of Nodes
        ax3 = axes[1, 0]
        bars1 = ax3.bar(x - width/2, pre_nodes, width, label='PreBreeding', alpha=0.8, color='steelblue')
        bars2 = ax3.bar(x + width/2, post_nodes, width, label='PostBreeding', alpha=0.8, color='coral')
        ax3.set_xlabel('Year', fontsize=11)
        ax3.set_ylabel('Number of Nodes', fontsize=11)
        ax3.set_title('Network Size (Nodes) Over Time', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(years)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Number of Edges
        ax4 = axes[1, 1]
        bars1 = ax4.bar(x - width/2, pre_edges, width, label='PreBreeding', alpha=0.8, color='steelblue')
        bars2 = ax4.bar(x + width/2, post_edges, width, label='PostBreeding', alpha=0.8, color='coral')
        ax4.set_xlabel('Year', fontsize=11)
        ax4.set_ylabel('Number of Edges', fontsize=11)
        ax4.set_title('Network Connectivity (Edges) Over Time', fontsize=12, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(years)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_file = os.path.join(output_dir, "network_density_combined.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved combined plot to {output_file}")

