"""
Generate flow intensity and network density plots from existing GraphML files.
This script can be run independently without re-running the full pipeline.
"""

import os
import re
import pandas as pd
import numpy as np
import networkx as nx
from glob import glob
from src.visualization import Visualizer

def extract_info_from_filename(filename):
    """
    Extract year, season, and node_type from GraphML filename.
    Format: migration_network_YYYY_SeasonName_nodetype.graphml
    """
    basename = os.path.basename(filename)
    match = re.match(r'migration_network_(\d{4})_(PreBreeding|PostBreeding)_(county|cluster)\.graphml', basename)
    if match:
        return {
            'year': int(match.group(1)),
            'season': match.group(2),
            'node_type': match.group(3)
        }
    return None

def load_network_statistics(results_dir='results'):
    """
    Load network statistics from GraphML files.
    """
    print("Scanning for GraphML files...")
    graphml_files = glob(os.path.join(results_dir, 'migration_network_*.graphml'))
    
    if not graphml_files:
        print(f"No GraphML files found in {results_dir}")
        return []
    
    print(f"Found {len(graphml_files)} GraphML files")
    
    statistics = []
    
    for graphml_file in graphml_files:
        info = extract_info_from_filename(graphml_file)
        if not info:
            print(f"Could not parse filename: {graphml_file}")
            continue
        
        try:
            print(f"Loading {os.path.basename(graphml_file)}...")
            network = nx.read_graphml(graphml_file)
            
            # Extract network statistics
            num_nodes = network.number_of_nodes()
            num_edges = network.number_of_edges()
            
            # Calculate density (for directed graph: edges / (nodes * (nodes - 1)))
            if num_nodes > 1:
                density = num_edges / (num_nodes * (num_nodes - 1))
            else:
                density = 0
            
            # Calculate average degree (for directed: edges / nodes)
            avg_degree = (2 * num_edges) / num_nodes if num_nodes > 0 else 0
            
            statistics.append({
                'year': info['year'],
                'season': info['season'],
                'node_type': info['node_type'],
                'nodes': num_nodes,
                'edges': num_edges,
                'density': density,
                'avg_degree': avg_degree,
                'max_flow': None,  # Will be filled from CSV if available
                'num_bottlenecks': None
            })
            
        except Exception as e:
            print(f"Error loading {graphml_file}: {e}")
            continue
    
    return statistics

def load_flow_data_from_csvs(results_dir='results'):
    """
    Try to extract flow data from bottleneck CSV files or summary CSV.
    """
    flow_data = {}
    
    # Check for summary statistics CSV first
    summary_csv = os.path.join(results_dir, 'network_summary_statistics.csv')
    if os.path.exists(summary_csv):
        print(f"Loading flow data from {summary_csv}...")
        try:
            df = pd.read_csv(summary_csv)
            # Aggregate by year/season/node_type
            aggregated = df.groupby(['year', 'season', 'node_type']).agg({
                'max_flow': 'sum',
                'num_bottlenecks': 'sum'
            }).reset_index()
            
            for _, row in aggregated.iterrows():
                key = (row['year'], row['season'], row['node_type'])
                flow_data[key] = {
                    'max_flow': row['max_flow'],
                    'num_bottlenecks': row['num_bottlenecks']
                }
            print(f"Loaded flow data for {len(flow_data)} year/season/node_type combinations")
        except Exception as e:
            print(f"Error reading summary CSV: {e}")
    
    # If no summary CSV, try to infer from bottleneck CSV filenames
    # (This is less reliable, but better than nothing)
    if not flow_data:
        print("No summary CSV found. Checking bottleneck CSV files...")
        bottleneck_csvs = glob(os.path.join(results_dir, '*_node_bottlenecks.csv'))
        for csv_file in bottleneck_csvs:
            # Try to extract info from filename
            basename = os.path.basename(csv_file)
            # Format might be: migration_network_YYYY_SeasonName_nodetype_componentN_node_bottlenecks.csv
            match = re.match(r'migration_network_(\d{4})_(PreBreeding|PostBreeding)_(county|cluster)_component\d+_node_bottlenecks\.csv', basename)
            if match:
                year = int(match.group(1))
                season = match.group(2)
                node_type = match.group(3)
                key = (year, season, node_type)
                
                if key not in flow_data:
                    flow_data[key] = {'max_flow': 0, 'num_bottlenecks': 0}
                
                # Count bottlenecks
                try:
                    bn_df = pd.read_csv(csv_file)
                    flow_data[key]['num_bottlenecks'] += len(bn_df)
                except:
                    pass
    
    return flow_data

def main():
    print("=" * 60)
    print("Generating Summary Visualizations from Existing Data")
    print("=" * 60)
    
    results_dir = 'results'
    if not os.path.exists(results_dir):
        print(f"Results directory '{results_dir}' not found!")
        return
    
    # Load network statistics from GraphML files
    statistics = load_network_statistics(results_dir)
    
    if not statistics:
        print("No statistics found. Cannot generate plots.")
        return
    
    # Load flow data from CSV files if available
    flow_data = load_flow_data_from_csvs(results_dir)
    
    # Merge flow data into statistics
    for stat in statistics:
        key = (stat['year'], stat['season'], stat['node_type'])
        if key in flow_data:
            stat['max_flow'] = flow_data[key]['max_flow']
            stat['num_bottlenecks'] = flow_data[key]['num_bottlenecks']
    
    # Convert to DataFrame
    df = pd.DataFrame(statistics)
    
    # Note: Cluster nodes have flow data, county nodes may not
    # Keep both for comparison - visualizations will prioritize cluster data
    if 'node_type' in df.columns:
        print(f"\nNode types in data: {df['node_type'].unique()}")
        print(f"Cluster entries: {len(df[df['node_type'] == 'cluster'])}")
        print(f"County entries: {len(df[df['node_type'] == 'county'])}")
        if len(df[df['node_type'] == 'cluster']) > 0:
            cluster_flow = df[df['node_type'] == 'cluster']['max_flow'].sum()
            print(f"Cluster total flow: {cluster_flow}")
        if len(df[df['node_type'] == 'county']) > 0:
            county_flow = df[df['node_type'] == 'county']['max_flow'].sum()
            print(f"County total flow: {county_flow}")
    
    # Filter out entries with no flow data (if flow is required)
    # For now, we'll include all entries but note missing flow data
    print(f"\nLoaded statistics for {len(df)} networks")
    print(f"Networks with flow data: {df['max_flow'].notna().sum()}")
    
    # Save raw statistics
    output_csv = os.path.join(results_dir, 'network_summary_statistics.csv')
    df.to_csv(output_csv, index=False)
    print(f"Saved statistics to {output_csv}")
    
    # Generate visualizations
    # Filter to entries with flow data for flow intensity plot
    # Check if max_flow has any non-zero values
    flow_df = df[df['max_flow'].notna()].copy()
    has_flow_data = len(flow_df) > 0 and flow_df['max_flow'].sum() > 0
    
    if has_flow_data:
        print("\nGenerating flow intensity plots...")
        Visualizer.plot_flow_intensity(
            flow_df.to_dict('records'),
            results_dir
        )
    else:
        print("\n" + "=" * 60)
        print("WARNING: No flow data available!")
        print("=" * 60)
        print("All max_flow values are 0.0. This means:")
        print("  1. Flow analysis may not have been run")
        print("  2. Flow analysis may have failed (no path from source to sink)")
        print("  3. Flow values were not properly saved")
        print("\nTo generate flow intensity plots, you need to:")
        print("  - Re-run the pipeline: python run_pipeline.py")
        print("  - Ensure flow analysis completes successfully")
        print("  - Check that max_flow values are > 0 in the summary CSV")
        print("\nSkipping flow intensity plot.")
    
    # Network density plot doesn't require flow data
    print("\nGenerating network density plots...")
    Visualizer.plot_network_density(
        df.to_dict('records'),
        results_dir
    )
    
    print("\n" + "=" * 60)
    print("Summary visualizations generated successfully!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - {os.path.join(results_dir, 'network_summary_statistics.csv')}")
    print(f"  - {os.path.join(results_dir, 'flow_intensity_over_time.png')}")
    print(f"  - {os.path.join(results_dir, 'network_density_over_time.png')}")

if __name__ == "__main__":
    main()

