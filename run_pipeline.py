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
from src.bottleneck_predictor import BottleneckPredictor
from src.community_detection import CommunityDetector

def create_summary_table(bottlenecks_mincut, communities_summary, year, season, max_flow):
    """
    Create a summary table with top 10 bottlenecks from min-cut and community analysis.
    
    Args:
        bottlenecks_mincut: List of bottleneck dicts from min-cut analysis
        communities_summary: List of community dicts with bottleneck info
        year: Year
        season: Season name
        max_flow: Total max flow value
        
    Returns:
        pd.DataFrame: Summary table
    """
    summary_rows = []
    
    # Add header row with overall stats
    summary_rows.append({
        'rank': '',
        'type': 'OVERALL_STATS',
        'identifier': '',
        'lat': '',
        'lon': '',
        'metric': f'Max Flow: {max_flow:.2f}',
        'description': f'Total bottlenecks: {len(bottlenecks_mincut)} nodes'
    })
    
    # Top 10 Min-Cut Bottlenecks (ranked by capacity)
    if bottlenecks_mincut:
        bn_df = pd.DataFrame(bottlenecks_mincut)
        # Remove duplicates by node_id, keep highest capacity
        bn_df = bn_df.sort_values('capacity', ascending=False).drop_duplicates('node_id', keep='first')
        top_bottlenecks = bn_df.head(10)
        
        summary_rows.append({
            'rank': '',
            'type': 'MIN_CUT_BOTTLENECKS',
            'identifier': '',
            'lat': '',
            'lon': '',
            'metric': '',
            'description': f'Top {len(top_bottlenecks)} bottlenecks from min-cut analysis'
        })
        
        for idx, (_, bn) in enumerate(top_bottlenecks.iterrows(), 1):
            summary_rows.append({
                'rank': idx,
                'type': 'Min-Cut Node',
                'identifier': bn['node_id'],
                'lat': f"{bn['lat']:.4f}",
                'lon': f"{bn['lon']:.4f}",
                'metric': f"Capacity: {bn['capacity']:.0f}",
                'description': f"Component {bn.get('component', 'N/A')}"
            })
    else:
        summary_rows.append({
            'rank': '',
            'type': 'MIN_CUT_BOTTLENECKS',
            'identifier': '',
            'lat': '',
            'lon': '',
            'metric': '',
            'description': 'No bottlenecks found'
        })
    
    # Top 10 Communities by Bottleneck Density
    if communities_summary:
        comm_df = pd.DataFrame(communities_summary)
        # Sort by bottleneck density (or num_bottlenecks if density is same)
        comm_df = comm_df.sort_values(['bottleneck_density', 'num_bottlenecks'], ascending=False)
        top_communities = comm_df.head(10)
        
        summary_rows.append({
            'rank': '',
            'type': 'COMMUNITY_ANALYSIS',
            'identifier': '',
            'lat': '',
            'lon': '',
            'metric': '',
            'description': f'Top {len(top_communities)} communities by bottleneck density'
        })
        
        for idx, (_, comm) in enumerate(top_communities.iterrows(), 1):
            summary_rows.append({
                'rank': idx,
                'type': 'Community',
                'identifier': f"Comm_{comm['community_id']}",
                'lat': f"{comm.get('lat', 0):.4f}" if pd.notna(comm.get('lat')) else 'N/A',
                'lon': f"{comm.get('lon', 0):.4f}" if pd.notna(comm.get('lon')) else 'N/A',
                'metric': f"Density: {comm['bottleneck_density']:.3f}",
                'description': f"{comm['num_bottlenecks']} bottlenecks, {comm['num_nodes']} nodes (Comp {comm['component']})"
            })
    else:
        summary_rows.append({
            'rank': '',
            'type': 'COMMUNITY_ANALYSIS',
            'identifier': '',
            'lat': '',
            'lon': '',
            'metric': '',
            'description': 'No community data available'
        })
    
    summary_df = pd.DataFrame(summary_rows)
    return summary_df

def main():
    print("Starting Sandhill Crane Migration Pipeline...")
    print("=============================================")
    
    # Configuration
    YEARS = [2018, 2019, 2020, 2021, 2022, 2023]
    USE_CLUSTERING = False  # Set to True to use ML-based clustering nodes, False for county-based
    SPLIT_SEASONS = True  # Set to True to split into Spring/Fall, False to use full year
    OUTPUT_DIR = "results"
    MAIN_DIR = os.path.join(OUTPUT_DIR, "main")  # Important files: PNGs, summaries
    DETAILED_DIR = os.path.join(OUTPUT_DIR, "detailed")  # Detailed CSVs, GraphML
    
    # Create directories
    for dir_path in [OUTPUT_DIR, MAIN_DIR, DETAILED_DIR]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
    # Initialize components
    splitter = SeasonSplitter() # Default dates
    network_builder = NetworkBuilder(max_distance_km=500, max_time_days=14)
    bottleneck_predictor = BottleneckPredictor()
    community_detector = CommunityDetector()
    
    # Track summary statistics for visualization
    summary_statistics = []
    
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
            
            # Save GraphML to detailed folder
            output_base_detailed = os.path.join(DETAILED_DIR, f"migration_network_{year}_{season_name}_{node_type}")
            output_base_main = os.path.join(MAIN_DIR, f"migration_network_{year}_{season_name}_{node_type}")
            nx.write_graphml(network, f"{output_base_detailed}.graphml")
            
            # Track all bottlenecks and communities for summary table
            all_bottlenecks_mincut = []
            all_communities_summary = []
            
            # 5. Flow Analysis (Max-Flow Min-Cut) - Per Connected Component
            # NOTE: Analyzing by connected components ensures sources/sinks are in same component
            # TODO: Replace with proper geographic flyways (West Coast: lon < -110, Central: -110 to -90, East: lon > -90)
            
            # Find weakly connected components (ensures sources/sinks are connected)
            components = list(nx.weakly_connected_components(network))
            print(f"\nFound {len(components)} connected component(s)")
            
            # Analyze each connected component separately
            all_cut_edges = []
            total_max_flow = 0
            
            for comp_idx, comp_nodes in enumerate(components):
                if len(comp_nodes) < 2:
                    print(f"Component {comp_idx + 1}: Skipping (only {len(comp_nodes)} node)")
                    continue
                    
                print(f"\nAnalyzing Component {comp_idx + 1} ({len(comp_nodes)} nodes)...")
                comp_subgraph = network.subgraph(comp_nodes)
                
                # Create analyzer for this component
                comp_analyzer = FlowAnalyzer(comp_subgraph)
                
                # Define Source/Sink based on geography and season WITHIN this component
                lats = [d['lat'] for n, d in comp_subgraph.nodes(data=True)]
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
                source_nodes = [n for n, d in comp_subgraph.nodes(data=True) if source_func(d)]
                sink_nodes = [n for n, d in comp_subgraph.nodes(data=True) if sink_func(d)]
                print(f"  Source nodes: {len(source_nodes)}, Sink nodes: {len(sink_nodes)}")
                
                # Fallback: if no sources/sinks found, use extreme nodes
                if len(source_nodes) == 0:
                    print("  No source nodes found. Using extreme node(s) as fallback.")
                    if season_name == 'PreBreeding':
                        sorted_by_lat = sorted(comp_subgraph.nodes(data=True), key=lambda x: x[1]['lat'])
                        source_node_id = sorted_by_lat[0][0]
                        source_lat = sorted_by_lat[0][1]['lat']
                    else:
                        sorted_by_lat = sorted(comp_subgraph.nodes(data=True), key=lambda x: x[1]['lat'], reverse=True)
                        source_node_id = sorted_by_lat[0][0]
                        source_lat = sorted_by_lat[0][1]['lat']
                    source_nodes = [source_node_id]
                    source_func = lambda d, slat=source_lat: abs(d.get('lat', 0) - slat) < 0.001
                    
                if len(sink_nodes) == 0:
                    print("  No sink nodes found. Using extreme node(s) as fallback.")
                    if season_name == 'PreBreeding':
                        sorted_by_lat = sorted(comp_subgraph.nodes(data=True), key=lambda x: x[1]['lat'], reverse=True)
                        sink_node_id = sorted_by_lat[0][0]
                        sink_lat = sorted_by_lat[0][1]['lat']
                    else:
                        sorted_by_lat = sorted(comp_subgraph.nodes(data=True), key=lambda x: x[1]['lat'])
                        sink_node_id = sorted_by_lat[0][0]
                        sink_lat = sorted_by_lat[0][1]['lat']
                    sink_nodes = [sink_node_id]
                    sink_func = lambda d, slat=sink_lat: abs(d.get('lat', 0) - slat) < 0.001
                
                if len(source_nodes) == 0 or len(sink_nodes) == 0:
                    print(f"  Skipping component {comp_idx + 1}: No valid sources/sinks")
                    continue
                
                # Build flow network for this component
                comp_analyzer.build_flow_network(source_func, sink_func)
                
                try:
                    max_flow, flow_dict = comp_analyzer.calculate_max_flow()
                    min_cut_val, cut_edges = comp_analyzer.calculate_min_cut()
                    
                    total_max_flow += max_flow
                    all_cut_edges.extend(cut_edges)
                    
                    # Extract bottleneck locations
                    bottleneck_locations = comp_analyzer.extract_bottleneck_locations(cut_edges)
                    
                    # Add to predictor for future predictions
                    bottleneck_predictor.add_year_data(year, season_name, bottleneck_locations, comp_subgraph)
                    
                    # Collect bottlenecks for summary (aggregate across components)
                    if bottleneck_locations['node_bottlenecks']:
                        for bn in bottleneck_locations['node_bottlenecks']:
                            bn['component'] = comp_idx + 1
                            all_bottlenecks_mincut.append(bn)
                    
                    # Save bottlenecks to CSV (detailed folder)
                    if bottleneck_locations['node_bottlenecks']:
                        node_bn_df = pd.DataFrame(bottleneck_locations['node_bottlenecks'])
                        node_bn_df.to_csv(f"{output_base_detailed}_component{comp_idx+1}_node_bottlenecks.csv", index=False)
                        print(f"  Saved {len(node_bn_df)} node bottlenecks to CSV")
                    
                    if bottleneck_locations['edge_bottlenecks']:
                        edge_bn_df = pd.DataFrame(bottleneck_locations['edge_bottlenecks'])
                        edge_bn_df.to_csv(f"{output_base_detailed}_component{comp_idx+1}_edge_bottlenecks.csv", index=False)
                        print(f"  Saved {len(edge_bn_df)} edge bottlenecks to CSV")
                    
                    # Community Detection
                    print(f"  Detecting communities in component {comp_idx + 1}...")
                    try:
                        comm_results = community_detector.detect_communities(comp_subgraph, method='louvain')
                        print(f"  Found {comm_results['num_communities']} communities (modularity: {comm_results['modularity']:.3f})")
                        
                        # Compare communities with bottlenecks
                        comparison = community_detector.compare_with_bottlenecks(
                            comp_subgraph, 
                            comm_results['communities'], 
                            bottleneck_locations
                        )
                        
                        # Collect community summary data
                        comm_locations = community_detector.get_community_locations(
                            comp_subgraph, 
                            comm_results['communities']
                        )
                        
                        # Add to summary list
                        for _, row in comm_locations.iterrows():
                            comm_id = row['community_id']
                            all_communities_summary.append({
                                'community_id': comm_id,
                                'component': comp_idx + 1,
                                'num_nodes': comparison['nodes_per_community'][comm_id],
                                'num_bottlenecks': comparison['bottlenecks_per_community'].get(comm_id, 0),
                                'bottleneck_density': comparison['bottleneck_density'][comm_id],
                                'lat': row.get('lat'),
                                'lon': row.get('lon')
                            })
                        
                        # Save community results (detailed folder)
                        comm_df = pd.DataFrame([
                            {'node_id': node, 'community_id': comm_id}
                            for node, comm_id in comm_results['communities'].items()
                        ])
                        comm_df.to_csv(f"{output_base_detailed}_component{comp_idx+1}_communities.csv", index=False)
                        
                        comm_locations.to_csv(f"{output_base_detailed}_component{comp_idx+1}_community_locations.csv", index=False)
                        
                        comparison_df = pd.DataFrame([
                            {
                                'community_id': comm_id,
                                'num_nodes': comparison['nodes_per_community'][comm_id],
                                'num_bottlenecks': comparison['bottlenecks_per_community'].get(comm_id, 0),
                                'bottleneck_density': comparison['bottleneck_density'][comm_id]
                            }
                            for comm_id in set(comm_results['communities'].values())
                        ])
                        comparison_df.to_csv(f"{output_base_detailed}_component{comp_idx+1}_community_bottleneck_comparison.csv", index=False)
                        
                        print(f"  Community-bottleneck comparison saved")
                        
                    except Exception as e:
                        print(f"  Community detection failed: {e}")
                    
                    print(f"  Component {comp_idx + 1}: Max Flow = {max_flow:.2f}, Min Cut = {min_cut_val:.2f}")
                    
                    # Track statistics for this component
                    summary_statistics.append({
                        'year': year,
                        'season': season_name,
                        'node_type': node_type,
                        'component': comp_idx + 1,
                        'max_flow': max_flow,
                        'min_cut': min_cut_val,
                        'nodes': comp_subgraph.number_of_nodes(),
                        'edges': comp_subgraph.number_of_edges(),
                        'num_bottlenecks': len(bottleneck_locations['node_bottlenecks']) + len(bottleneck_locations['edge_bottlenecks'])
                    })
                    
                except Exception as e:
                    print(f"  Flow analysis failed for component {comp_idx + 1}: {e}")
                    continue
            
            # Track overall network statistics (aggregated across components)
            # Always save overall stats, even if flow is 0 (to track that flow analysis ran)
            summary_statistics.append({
                'year': year,
                'season': season_name,
                'node_type': node_type,
                'component': 'all',
                'max_flow': total_max_flow,
                'min_cut': total_max_flow,  # Min cut equals max flow
                'nodes': network.number_of_nodes(),
                'edges': network.number_of_edges(),
                'num_bottlenecks': len(all_cut_edges)
            })

            # 6. Create Summary Table for this season/year
            print(f"\nCreating summary table for {season_name} {year}...")
            summary_table = create_summary_table(
                all_bottlenecks_mincut, 
                all_communities_summary,
                year, 
                season_name,
                total_max_flow
            )
            summary_file = os.path.join(MAIN_DIR, f"summary_{year}_{season_name}_{node_type}.csv")
            summary_table.to_csv(summary_file, index=False)
            print(f"Saved summary table to {summary_file}")
            
            # 7. Visualize (using full network, but with combined results) - Save to main folder
            if total_max_flow > 0:
                Visualizer.create_interactive_map(
                    network, 
                    f"{output_base_main}.html", 
                    title=f"Sandhill Crane Migration ({season_name} {year}) - Total Max Flow: {total_max_flow:.0f}"
                )
                
                Visualizer.plot_min_cut(
                    network, 
                    all_cut_edges, 
                    f"{output_base_main}_mincut.png", 
                    title=f"Min-Cut Bottlenecks ({season_name} {year}) - Total Flow: {total_max_flow:.0f}"
                )
            else:
                print(f"Warning: No valid flow found for {year} {season_name}. Skipping visualization.")
    
    # Generate summary visualizations
    print("\n" + "=" * 60)
    print("Generating Summary Visualizations")
    print("=" * 60)
    
    if summary_statistics:
        summary_df = pd.DataFrame(summary_statistics)
        summary_df.to_csv(os.path.join(DETAILED_DIR, "network_summary_statistics.csv"), index=False)
        print("Saved summary statistics to CSV")
        
        # Aggregate by year/season (sum across components)
        aggregated = summary_df.groupby(['year', 'season', 'node_type']).agg({
            'max_flow': 'sum',
            'nodes': 'first',  # Use first since all components have same total nodes
            'edges': 'first',
            'num_bottlenecks': 'sum'
        }).reset_index()
        
        # Create flow intensity plots (main folder) - creates 3 separate files
        Visualizer.plot_flow_intensity(
            aggregated.to_dict('records'),
            MAIN_DIR
        )
        
        # Create network density plots (main folder) - creates 4 separate files
        Visualizer.plot_network_density(
            aggregated.to_dict('records'),
            MAIN_DIR
        )
    
    # Generate predictions for future year (if we have enough historical data)
    print("\n" + "=" * 60)
    print("Generating Bottleneck Predictions")
    print("=" * 60)
    
    # For demonstration, predict for next year using last year's network
    if len(YEARS) > 0:
        last_year = max(YEARS)
        print(f"Historical data collected for bottleneck prediction.")
        print(f"To predict bottlenecks for year {last_year + 1}, provide that year's network data.")

    print("\nPipeline execution complete!")

if __name__ == "__main__":
    main()

