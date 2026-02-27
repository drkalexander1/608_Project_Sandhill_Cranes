"""
Simple Folium PNG Generator for Sandhill Crane Migration Networks
================================================================

This script generates PNG maps with US map background using folium
and selenium for screenshot capture.
"""

import pandas as pd
import numpy as np
import networkx as nx
import folium
from folium import plugins
import os
import warnings
from src.network_transformation import EfficientCraneNetworkBuilderV2

warnings.filterwarnings('ignore')

class SimpleFoliumPNGGenerator:
    """
    Generates PNG maps with US map background using folium.
    """
    
    def __init__(self):
        """
        Initialize the PNG generator.
        """
        self.networks = {}
        self.nodes_data = {}
        self.years = [2018, 2019, 2020, 2021, 2022, 2023]
        
    def generate_all_networks(self):
        """
        Generate migration networks for all years.
        """
        print("Generating migration networks for all years...")
        print("=" * 60)
        
        for year in self.years:
            print(f"\nProcessing year {year}...")
            
            # Check if parquet file exists
            parquet_file = f'Antigone canadensis_all_checklist_{year}.parquet'
            if not os.path.exists(parquet_file):
                print(f"Warning: {parquet_file} not found, skipping year {year}")
                continue
            
            # Initialize network builder
            builder = EfficientCraneNetworkBuilderV2(
                max_distance_km=500,
                max_time_days=14
            )
            
            # Build network
            network = builder.build_network(parquet_file)
            
            # Save network
            output_file = f'efficient_sandhill_crane_migration_network_{year}.graphml'
            builder.save_network(output_file)
            
            # Store network and extract node data
            self.networks[year] = network
            
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
            
            self.nodes_data[year] = pd.DataFrame(nodes_data)
            
            print(f"Year {year}: {network.number_of_nodes()} nodes, {network.number_of_edges()} edges")
        
        print("=" * 60)
        print("All networks generated successfully!")
    
    def create_folium_png(self, year, nodes_df, network, output_file):
        """
        Create a PNG map using folium with US map background.
        """
        print(f"Creating PNG map for year {year}...")
        
        # Filter to significant counties (50+ cranes for better visibility)
        significant_counties = nodes_df[nodes_df['total_cranes'] >= 50].copy()
        
        # Create base map
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
            
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=size,
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
                        opacity=0.4
                    ).add_to(m)
        
        # Add legend
        legend_html = f'''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Sandhill Crane Migration Network - {year}</b></p>
        <p><i class="fa fa-circle" style="color:red"></i> >1000 cranes</p>
        <p><i class="fa fa-circle" style="color:orange"></i> 500-1000 cranes</p>
        <p><i class="fa fa-circle" style="color:yellow"></i> 200-500 cranes</p>
        <p><i class="fa fa-circle" style="color:green"></i> 50-200 cranes</p>
        <p><i class="fa fa-minus" style="color:blue"></i> Migration routes</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add statistics text
        stats_html = f'''
        <div style="position: fixed; 
                    top: 50px; right: 50px; width: 200px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:12px; padding: 10px">
        <p><b>Network Statistics</b></p>
        <p>Nodes: {network.number_of_nodes()}</p>
        <p>Edges: {network.number_of_edges()}</p>
        <p>Significant Counties: {len(significant_counties)}</p>
        <p>Total Cranes: {nodes_df['total_cranes'].sum():.0f}</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(stats_html))
        
        # Save as HTML first
        html_file = output_file.replace('.png', '.html')
        m.save(html_file)
        
        # Try to convert to PNG using selenium
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            
            # Take screenshot
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--window-size=1920,1080')
            
            driver = webdriver.Chrome(options=chrome_options)
            driver.get(f'file://{os.path.abspath(html_file)}')
            driver.save_screenshot(output_file)
            driver.quit()
            
            print(f"PNG map saved as '{output_file}'")
            
        except ImportError:
            print("Selenium not available. HTML file saved instead.")
            print(f"Open '{html_file}' in your browser to view the map.")
        except Exception as e:
            print(f"Selenium screenshot failed: {e}")
            print("HTML file saved instead.")
            print(f"Open '{html_file}' in your browser to view the map.")
    
    def create_all_png_maps(self):
        """
        Create PNG maps for all years.
        """
        print("\nCreating PNG maps for all years...")
        print("=" * 60)
        
        for year in self.years:
            if year in self.networks and year in self.nodes_data:
                output_file = f'sandhill_crane_migration_network_{year}.png'
                self.create_folium_png(year, self.nodes_data[year], self.networks[year], output_file)
        
        print("=" * 60)
        print("All PNG maps created successfully!")
    
    def run_complete_analysis(self):
        """
        Run the complete analysis.
        """
        print("Starting Simple Folium PNG Generation")
        print("=" * 80)
        
        # Generate all networks
        self.generate_all_networks()
        
        # Create PNG maps for all years
        self.create_all_png_maps()
        
        print("\n" + "=" * 80)
        print("Analysis Complete!")
        print("=" * 80)
        print("\nGenerated files:")
        print("- Migration networks: efficient_sandhill_crane_migration_network_YYYY.graphml")
        print("- PNG maps: sandhill_crane_migration_network_YYYY.png")
        print("- HTML maps: sandhill_crane_migration_network_YYYY.html")

def main():
    """
    Main function to run the simple PNG generation.
    """
    generator = SimpleFoliumPNGGenerator()
    generator.run_complete_analysis()

if __name__ == "__main__":
    main()
