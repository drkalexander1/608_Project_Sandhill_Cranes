import os
import pandas as pd
import numpy as np
import networkx as nx
from src.data_processing import DataLoader
from src.season_detection import SeasonSplitter
from src.node_generation import CountyNodeGenerator, ClusterNodeGenerator
from src.network_building import NetworkBuilder
from src.analysis import FlowAnalyzer
from src.visualization import Visualizer

def main():
    print("Starting Sandhill Crane Migration Pipeline...")
    print("=============================================")
    
    # Configuration
    YEARS = [2018, 2019, 2020, 2021, 2022, 2023]
    USE_CLUSTERING = False  # Set to True to use ML-based clustering nodes, False for county-based
    SPLIT_SEASONS = True  # Set to True to split into Spring/Fall, False to use full year
    OUTPUT_DIR = "results"
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Initialize components
    splitter = SeasonSplitter() # Default dates
    network_builder = NetworkBuilder(max_distance_km=500, max_time_days=14)
    
    # Loop through years
    for year in YEARS:
        print(f"\nProcessing Year: {year}")
        print("-" * 30)
        
        file_path = f'Antigone canadensis_all_checklist_{year}.parquet'
        if not os.path.exists(file_path):
            print(f"File {file_path} not found. Skipping.")
            continue
            
        # 1. Load Data
        df = DataLoader.load_and_filter_data(file_path)
        if df.empty:
            continue
            
        # 2. Split Seasons (optional)
        if SPLIT_SEASONS:
            seasons = splitter.split_seasons(df)
            season_list = seasons.items()
        else:
            # Use full year data
            seasons = {'FullYear': df}
            season_list = seasons.items()
        
        for season_name, season_df in season_list:
            if season_df.empty:
                print(f"No data for {season_name} {year}. Skipping.")
                continue
                
            print(f"\nAnalyzing {season_name} {year}...")
            
            # 3. Generate Nodes
            # We can do both or just one. Let's do Clustering if enabled, else County.
            if USE_CLUSTERING:
                node_gen = ClusterNodeGenerator(min_cluster_size=3)  # HDBSCAN: only needs min_cluster_size
                nodes_df, _ = node_gen.generate_nodes(season_df)
                node_type = "cluster"
            else:
                node_gen = CountyNodeGenerator()
                nodes_df, _ = node_gen.generate_nodes(season_df)
                node_type = "county"
                
            if nodes_df.empty:
                print("No nodes generated.")
                continue
                
            # 4. Build Network
            network = network_builder.build_network(nodes_df)
            
            # Save GraphML
            output_base = os.path.join(OUTPUT_DIR, f"migration_network_{year}_{season_name}_{node_type}")
            nx.write_graphml(network, f"{output_base}.graphml")
            
            # 5. Flow Analysis (Max-Flow Min-Cut) - Per Flyway
            # Find weakly connected components (flyways)
            components = list(nx.weakly_connected_components(network))
            print(f"\nFound {len(components)} flyway(s)/connected component(s)")

            # Analyze each flyway separately
            all_cut_edges = []
            total_max_flow = 0

            for flyway_idx, flyway_nodes in enumerate(components):
                if len(flyway_nodes) < 2:
                    print(f"Flyway {flyway_idx + 1}: Skipping (only {len(flyway_nodes)} node)")
                    continue
                    
                print(f"\nAnalyzing Flyway {flyway_idx + 1} ({len(flyway_nodes)} nodes)...")
                flyway_subgraph = network.subgraph(flyway_nodes)
                
                # Create analyzer for this flyway
                flyway_analyzer = FlowAnalyzer(flyway_subgraph)
                
                # Define Source/Sink based on geography and season WITHIN this flyway
                lats = [d['lat'] for n, d in flyway_subgraph.nodes(data=True)]
                if not lats:
                    continue
                    
                min_lat, max_lat = min(lats), max(lats)
                lat_range = max_lat - min_lat
                
                if lat_range == 0:
                    print(f"  Warning: All nodes have same latitude. Skipping.")
                    continue
                
                # PreBreeding (before August): South -> North migration
                if season_name == 'PreBreeding':
                    source_cutoff = min_lat + (lat_range * 0.3) # Bottom 30%
                    sink_cutoff = max_lat - (lat_range * 0.3)   # Top 30%
                    
                    source_func = lambda d: d['lat'] <= source_cutoff
                    sink_func = lambda d: d['lat'] >= sink_cutoff
                    
                # PostBreeding (after August): North -> South migration
                else:  # PostBreeding or FullYear
                    source_cutoff = max_lat - (lat_range * 0.3) # Top 30%
                    sink_cutoff = min_lat + (lat_range * 0.3)   # Bottom 30%
                    
                    source_func = lambda d: d['lat'] >= source_cutoff
                    sink_func = lambda d: d['lat'] <= sink_cutoff
                
                # Check how many nodes match criteria
                source_nodes = [n for n, d in flyway_subgraph.nodes(data=True) if source_func(d)]
                sink_nodes = [n for n, d in flyway_subgraph.nodes(data=True) if sink_func(d)]
                print(f"  Source nodes: {len(source_nodes)}, Sink nodes: {len(sink_nodes)}")
                
                # Fallback: if no sources/sinks found, use extreme nodes
                if len(source_nodes) == 0:
                    print("  No source nodes found. Using extreme node(s) as fallback.")
                    if season_name == 'PreBreeding':
                        sorted_by_lat = sorted(flyway_subgraph.nodes(data=True), key=lambda x: x[1]['lat'])
                        source_node_id = sorted_by_lat[0][0]
                        source_lat = sorted_by_lat[0][1]['lat']
                    else:
                        sorted_by_lat = sorted(flyway_subgraph.nodes(data=True), key=lambda x: x[1]['lat'], reverse=True)
                        source_node_id = sorted_by_lat[0][0]
                        source_lat = sorted_by_lat[0][1]['lat']
                    source_nodes = [source_node_id]
                    source_func = lambda d, slat=source_lat: abs(d.get('lat', 0) - slat) < 0.001
                    
                if len(sink_nodes) == 0:
                    print("  No sink nodes found. Using extreme node(s) as fallback.")
                    if season_name == 'PreBreeding':
                        sorted_by_lat = sorted(flyway_subgraph.nodes(data=True), key=lambda x: x[1]['lat'], reverse=True)
                        sink_node_id = sorted_by_lat[0][0]
                        sink_lat = sorted_by_lat[0][1]['lat']
                    else:
                        sorted_by_lat = sorted(flyway_subgraph.nodes(data=True), key=lambda x: x[1]['lat'])
                        sink_node_id = sorted_by_lat[0][0]
                        sink_lat = sorted_by_lat[0][1]['lat']
                    sink_nodes = [sink_node_id]
                    sink_func = lambda d, slat=sink_lat: abs(d.get('lat', 0) - slat) < 0.001
                
                if len(source_nodes) == 0 or len(sink_nodes) == 0:
                    print(f"  Skipping flyway {flyway_idx + 1}: No valid sources/sinks")
                    continue
                
                # Build flow network for this flyway
                flyway_analyzer.build_flow_network(source_func, sink_func)
                
                try:
                    max_flow, flow_dict = flyway_analyzer.calculate_max_flow()
                    min_cut_val, cut_edges = flyway_analyzer.calculate_min_cut()
                    
                    total_max_flow += max_flow
                    all_cut_edges.extend(cut_edges)
                    
                    print(f"  Flyway {flyway_idx + 1}: Max Flow = {max_flow:.2f}, Min Cut = {min_cut_val:.2f}")
                    
                except Exception as e:
                    print(f"  Flow analysis failed for flyway {flyway_idx + 1}: {e}")
                    continue

            # 6. Visualize (using full network, but with combined results)
            if total_max_flow > 0:
                Visualizer.create_interactive_map(
                    network, 
                    f"{output_base}.html", 
                    title=f"Sandhill Crane Migration ({season_name} {year}) - Total Max Flow: {total_max_flow:.0f}"
                )
                
                Visualizer.plot_min_cut(
                    network, 
                    all_cut_edges, 
                    f"{output_base}_mincut.png", 
                    title=f"Min-Cut Bottlenecks ({season_name} {year}) - Total Flow: {total_max_flow:.0f}"
                )
            else:
                print(f"Warning: No valid flow found for {year} {season_name}. Skipping visualization.")
    print("\nPipeline execution complete!")

if __name__ == "__main__":
    main()

