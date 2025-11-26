import os
import pandas as pd
import numpy as np
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
    USE_CLUSTERING = True  # Toggle to use ML-based nodes
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
            
        # 2. Split Seasons
        seasons = splitter.split_seasons(df)
        
        for season_name, season_df in seasons.items():
            if season_df.empty:
                print(f"No data for {season_name} {year}. Skipping.")
                continue
                
            print(f"\nAnalyzing {season_name} {year}...")
            
            # 3. Generate Nodes
            # We can do both or just one. Let's do Clustering if enabled, else County.
            if USE_CLUSTERING:
                node_gen = ClusterNodeGenerator(eps_km=50, min_samples=10)
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
            import networkx as nx
            nx.write_graphml(network, f"{output_base}.graphml")
            
            # 5. Flow Analysis (Max-Flow Min-Cut)
            analyzer = FlowAnalyzer(network)
            
            # Define Source/Sink based on geography and season
            lats = [d['lat'] for n, d in network.nodes(data=True)]
            if not lats:
                continue
                
            min_lat, max_lat = min(lats), max(lats)
            lat_range = max_lat - min_lat
            
            if lat_range == 0:
                print(f"Warning: All nodes have same latitude. Skipping flow analysis.")
                continue
            
            # Spring: South -> North
            if season_name == 'Spring':
                # Use more lenient thresholds: bottom 30% and top 30%
                source_cutoff = min_lat + (lat_range * 0.3) # Bottom 30%
                sink_cutoff = max_lat - (lat_range * 0.3)   # Top 30%
                
                source_func = lambda d: d['lat'] <= source_cutoff
                sink_func = lambda d: d['lat'] >= sink_cutoff
                
            # Fall: North -> South
            else:
                # Use more lenient thresholds: top 30% and bottom 30%
                source_cutoff = max_lat - (lat_range * 0.3) # Top 30%
                sink_cutoff = min_lat + (lat_range * 0.3)   # Bottom 30%
                
                source_func = lambda d: d['lat'] >= source_cutoff
                sink_func = lambda d: d['lat'] <= sink_cutoff
                
            # Debug: Check how many nodes match criteria
            source_nodes = [n for n, d in network.nodes(data=True) if source_func(d)]
            sink_nodes = [n for n, d in network.nodes(data=True) if sink_func(d)]
            print(f"Source nodes: {len(source_nodes)}, Sink nodes: {len(sink_nodes)}")
            
            # Fallback: if no sources/sinks found, use extreme nodes
            if len(source_nodes) == 0:
                print("No source nodes found with criteria. Using extreme node(s) as fallback.")
                if season_name == 'Spring':
                    # Spring: use southernmost
                    sorted_by_lat = sorted(network.nodes(data=True), key=lambda x: x[1]['lat'])
                    source_node_id = sorted_by_lat[0][0]
                    source_lat = sorted_by_lat[0][1]['lat']
                else:
                    # Fall: use northernmost
                    sorted_by_lat = sorted(network.nodes(data=True), key=lambda x: x[1]['lat'], reverse=True)
                    source_node_id = sorted_by_lat[0][0]
                    source_lat = sorted_by_lat[0][1]['lat']
                source_nodes = [source_node_id]
                source_func = lambda d, slat=source_lat: abs(d.get('lat', 0) - slat) < 0.001  # Match by lat
                
            if len(sink_nodes) == 0:
                print("No sink nodes found with criteria. Using extreme node(s) as fallback.")
                if season_name == 'Spring':
                    # Spring: use northernmost
                    sorted_by_lat = sorted(network.nodes(data=True), key=lambda x: x[1]['lat'], reverse=True)
                    sink_node_id = sorted_by_lat[0][0]
                    sink_lat = sorted_by_lat[0][1]['lat']
                else:
                    # Fall: use southernmost
                    sorted_by_lat = sorted(network.nodes(data=True), key=lambda x: x[1]['lat'])
                    sink_node_id = sorted_by_lat[0][0]
                    sink_lat = sorted_by_lat[0][1]['lat']
                sink_nodes = [sink_node_id]
                sink_func = lambda d, slat=sink_lat: abs(d.get('lat', 0) - slat) < 0.001  # Match by lat
                
            print(f"Final: Source nodes: {len(source_nodes)}, Sink nodes: {len(sink_nodes)}")
            
            analyzer.build_flow_network(source_func, sink_func)
            
            try:
                max_flow, flow_dict = analyzer.calculate_max_flow()
                min_cut_val, cut_edges = analyzer.calculate_min_cut()
                
                # 6. Visualize
                Visualizer.create_interactive_map(
                    network, 
                    f"{output_base}.html", 
                    title=f"Sandhill Crane Migration ({season_name} {year}) - Max Flow: {max_flow}"
                )
                
                Visualizer.plot_min_cut(
                    network, 
                    cut_edges, 
                    f"{output_base}_mincut.png", 
                    title=f"Min-Cut Bottlenecks ({season_name} {year})"
                )
            except Exception as e:
                print(f"Analysis failed for {year} {season_name}: {e}")

    print("\nPipeline execution complete!")

if __name__ == "__main__":
    main()

