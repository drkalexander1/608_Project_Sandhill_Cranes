"""
Multi-Year Sandhill Crane Migration Network Analysis
==================================================

This script generates migration networks for all years (2018-2023),
creates PNG visualizations for each year, and generates year-over-year
comparison maps.
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from geopy.distance import geodesic
from datetime import datetime, timedelta
import warnings
import os
from src.network_transformation import EfficientCraneNetworkBuilderV2
import contextily as ctx
from matplotlib.patches import Rectangle
import folium
from folium import plugins
import io
from PIL import Image

warnings.filterwarnings('ignore')

class MultiYearMigrationAnalyzer:
    """
    Analyzes sandhill crane migration patterns across multiple years.
    """
    
    def __init__(self, max_distance_km=500, max_time_days=14):
        """
        Initialize the analyzer.
        """
        self.max_distance_km = max_distance_km
        self.max_time_days = max_time_days
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
                max_distance_km=self.max_distance_km,
                max_time_days=self.max_time_days
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
    
    def create_png_map(self, year, nodes_df, network, output_file):
        """
        Create a PNG map visualization for a specific year with US map background.
        """
        print(f"Creating PNG map for year {year}...")
        
        # Filter to significant counties (50+ cranes for better visibility)
        significant_counties = nodes_df[nodes_df['total_cranes'] >= 50].copy()
        
        # Create base map using folium first to get background
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
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Sandhill Crane Migration Network - ''' + str(year) + '''</b></p>
        <p><i class="fa fa-circle" style="color:red"></i> >1000 cranes</p>
        <p><i class="fa fa-circle" style="color:orange"></i> 500-1000 cranes</p>
        <p><i class="fa fa-circle" style="color:yellow"></i> 200-500 cranes</p>
        <p><i class="fa fa-circle" style="color:green"></i> 50-200 cranes</p>
        <p><i class="fa fa-minus" style="color:blue"></i> Migration routes</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add statistics text
        stats_text = f"""Network Statistics:
Nodes: {network.number_of_nodes()}
Edges: {network.number_of_edges()}
Significant Counties: {len(significant_counties)}
Total Cranes: {nodes_df['total_cranes'].sum():.0f}"""
        
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
        
        # Save as PNG using folium's screenshot capability
        try:
            # Try to use selenium for screenshot
            import selenium
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            
            # Save temporary HTML file
            temp_html = f'temp_map_{year}.html'
            m.save(temp_html)
            
            # Take screenshot
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--window-size=1920,1080')
            
            driver = webdriver.Chrome(options=chrome_options)
            driver.get(f'file://{os.path.abspath(temp_html)}')
            driver.save_screenshot(output_file)
            driver.quit()
            
            # Clean up temp file
            os.remove(temp_html)
            
        except ImportError:
            print("Selenium not available, falling back to matplotlib with contextily...")
            self.create_png_map_matplotlib(year, nodes_df, network, output_file)
            return
        except Exception as e:
            print(f"Selenium screenshot failed: {e}")
            print("Falling back to matplotlib with contextily...")
            self.create_png_map_matplotlib(year, nodes_df, network, output_file)
            return
        
        print(f"PNG map saved as '{output_file}'")
    
    def create_png_map_matplotlib(self, year, nodes_df, network, output_file):
        """
        Create a PNG map visualization using matplotlib with contextily background.
        """
        print(f"Creating PNG map for year {year} using matplotlib...")
        
        # Set up the plot
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Filter to significant counties (50+ cranes for better visibility)
        significant_counties = nodes_df[nodes_df['total_cranes'] >= 50].copy()
        
        # Set map bounds (continental US)
        ax.set_xlim(-125, -66)
        ax.set_ylim(24, 50)
        ax.set_aspect('equal')
        
        # Add background map using contextily
        try:
            ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.OpenStreetMap.Mapnik)
        except Exception as e:
            print(f"Contextily failed: {e}")
            # Fallback to simple background
            ax.set_facecolor('lightblue')
            ax.add_patch(Rectangle((-125, 24), 59, 26, facecolor='lightgreen', alpha=0.3))
        
        # Add grid
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Sandhill Crane Migration Network - {year}', fontsize=16, fontweight='bold')
        
        # Create subgraph for significant counties
        subgraph = network.subgraph(significant_counties['county_code'])
        
        # Draw migration routes first (so they appear behind nodes)
        edges_with_weights = [(source, target, data['weight']) for source, target, data in subgraph.edges(data=True)]
        sorted_edges = sorted(edges_with_weights, key=lambda x: x[2], reverse=True)
        
        for source, target, weight in sorted_edges:
            source_data = significant_counties[significant_counties['county_code'] == source]
            target_data = significant_counties[significant_counties['county_code'] == target]
            
            if not source_data.empty and not target_data.empty:
                if weight > 0.01:  # Lower threshold to show more connections
                    ax.plot([source_data['lon'].iloc[0], target_data['lon'].iloc[0]],
                           [source_data['lat'].iloc[0], target_data['lat'].iloc[0]],
                           color='blue', alpha=0.3, linewidth=weight*3)
        
        # Draw nodes
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
            size = max(20, min(100, row['total_cranes'] / 10))
            
            ax.scatter(row['lon'], row['lat'], 
                      s=size, c=color, alpha=0.7, 
                      edgecolors='black', linewidth=0.5)
        
        # Add legend
        legend_elements = [
            plt.scatter([], [], s=100, c='red', alpha=0.7, edgecolors='black', label='>1000 cranes'),
            plt.scatter([], [], s=80, c='orange', alpha=0.7, edgecolors='black', label='500-1000 cranes'),
            plt.scatter([], [], s=60, c='yellow', alpha=0.7, edgecolors='black', label='200-500 cranes'),
            plt.scatter([], [], s=40, c='green', alpha=0.7, edgecolors='black', label='50-200 cranes'),
            plt.Line2D([0], [0], color='blue', alpha=0.3, linewidth=2, label='Migration routes')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # Add statistics text
        stats_text = f"""Network Statistics:
Nodes: {network.number_of_nodes()}
Edges: {network.number_of_edges()}
Significant Counties: {len(significant_counties)}
Total Cranes: {nodes_df['total_cranes'].sum():.0f}"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"PNG map saved as '{output_file}'")
    
    def create_all_png_maps(self):
        """
        Create PNG maps for all years.
        """
        print("\nCreating PNG maps for all years...")
        print("=" * 60)
        
        for year in self.years:
            if year in self.networks and year in self.nodes_data:
                output_file = f'sandhill_crane_migration_network_{year}.png'
                self.create_png_map(year, self.nodes_data[year], self.networks[year], output_file)
        
        print("=" * 60)
        print("All PNG maps created successfully!")
    
    def create_year_comparison_maps(self):
        """
        Create year-over-year comparison maps.
        """
        print("\nCreating year-over-year comparison maps...")
        print("=" * 60)
        
        # Create comparison for consecutive years
        for i in range(len(self.years) - 1):
            year1 = self.years[i]
            year2 = self.years[i + 1]
            
            if year1 in self.nodes_data and year2 in self.nodes_data:
                self.create_year_difference_map(year1, year2)
        
        # Create comparison for first and last year
        if 2018 in self.nodes_data and 2023 in self.nodes_data:
            self.create_year_difference_map(2018, 2023, "2018_vs_2023")
        
        print("=" * 60)
        print("All comparison maps created successfully!")
    
    def create_year_difference_map(self, year1, year2, suffix=None):
        """
        Create a difference map between two years.
        """
        if suffix is None:
            suffix = f"{year1}_vs_{year2}"
        
        print(f"Creating difference map: {year1} vs {year2}")
        
        # Get nodes data for both years
        nodes1 = self.nodes_data[year1].copy()
        nodes2 = self.nodes_data[year2].copy()
        
        # Merge on county_code to find common counties
        merged = pd.merge(nodes1, nodes2, on='county_code', suffixes=('_1', '_2'))
        
        # Calculate differences
        merged['crane_difference'] = merged['total_cranes_2'] - merged['total_cranes_1']
        merged['crane_change_pct'] = ((merged['total_cranes_2'] - merged['total_cranes_1']) / 
                                      merged['total_cranes_1'].replace(0, np.nan) * 100)
        
        # Use coordinates from year1 (or year2 if year1 doesn't have them)
        merged['lat'] = merged['lat_1'].fillna(merged['lat_2'])
        merged['lon'] = merged['lon_1'].fillna(merged['lon_2'])
        
        # Set up the plot
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Set map bounds
        ax.set_xlim(-125, -66)
        ax.set_ylim(24, 50)
        ax.set_aspect('equal')
        
        # Add background map using contextily
        try:
            ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.OpenStreetMap.Mapnik)
        except Exception as e:
            print(f"Contextily failed: {e}")
            # Fallback to simple background
            ax.set_facecolor('lightblue')
            ax.add_patch(Rectangle((-125, 24), 59, 26, facecolor='lightgreen', alpha=0.3))
        
        # Add grid
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Sandhill Crane Population Change: {year1} → {year2}', fontsize=16, fontweight='bold')
        
        # Filter to significant changes
        significant_changes = merged[abs(merged['crane_difference']) >= 10].copy()
        
        # Draw nodes with color based on change
        for _, row in significant_changes.iterrows():
            if row['crane_difference'] > 0:
                color = 'green'  # Increase
                alpha = min(0.8, abs(row['crane_difference']) / 500)
            else:
                color = 'red'    # Decrease
                alpha = min(0.8, abs(row['crane_difference']) / 500)
            
            # Size based on absolute change
            size = max(20, min(100, abs(row['crane_difference']) / 5))
            
            ax.scatter(row['lon'], row['lat'], 
                      s=size, c=color, alpha=alpha, 
                      edgecolors='black', linewidth=0.5)
        
        # Add legend
        legend_elements = [
            plt.scatter([], [], s=100, c='green', alpha=0.7, edgecolors='black', label='Population Increase'),
            plt.scatter([], [], s=100, c='red', alpha=0.7, edgecolors='black', label='Population Decrease')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # Add statistics text
        total_change = merged['crane_difference'].sum()
        avg_change = merged['crane_difference'].mean()
        counties_with_change = len(significant_changes)
        
        stats_text = f"""Change Statistics:
Total Change: {total_change:+.0f} cranes
Average Change: {avg_change:+.1f} cranes
Counties with Significant Change: {counties_with_change}
Total Counties: {len(merged)}"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        output_file = f'sandhill_crane_change_{suffix}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Difference map saved as '{output_file}'")
    
    def generate_summary_report(self):
        """
        Generate a summary report of all years.
        """
        print("\nGenerating summary report...")
        print("=" * 60)
        
        summary_data = []
        
        for year in self.years:
            if year in self.networks and year in self.nodes_data:
                network = self.networks[year]
                nodes_df = self.nodes_data[year]
                
                summary_data.append({
                    'Year': year,
                    'Nodes': network.number_of_nodes(),
                    'Edges': network.number_of_edges(),
                    'Total_Cranes': nodes_df['total_cranes'].sum(),
                    'Counties_50_Plus': len(nodes_df[nodes_df['total_cranes'] >= 50]),
                    'Max_Cranes_County': nodes_df['total_cranes'].max(),
                    'Avg_Cranes_County': nodes_df['total_cranes'].mean()
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary
        summary_df.to_csv('migration_network_summary.csv', index=False)
        
        # Print summary
        print("\nMigration Network Summary (2018-2023):")
        print(summary_df.to_string(index=False))
        
        # Create summary visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total cranes over time
        axes[0, 0].plot(summary_df['Year'], summary_df['Total_Cranes'], marker='o', linewidth=2)
        axes[0, 0].set_title('Total Cranes Over Time')
        axes[0, 0].set_xlabel('Year')
        axes[0, 0].set_ylabel('Total Cranes')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Number of nodes over time
        axes[0, 1].plot(summary_df['Year'], summary_df['Nodes'], marker='s', color='orange', linewidth=2)
        axes[0, 1].set_title('Number of Counties Over Time')
        axes[0, 1].set_xlabel('Year')
        axes[0, 1].set_ylabel('Number of Counties')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Number of edges over time
        axes[1, 0].plot(summary_df['Year'], summary_df['Edges'], marker='^', color='green', linewidth=2)
        axes[1, 0].set_title('Number of Migration Routes Over Time')
        axes[1, 0].set_xlabel('Year')
        axes[1, 0].set_ylabel('Number of Routes')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Counties with 50+ cranes over time
        axes[1, 1].plot(summary_df['Year'], summary_df['Counties_50_Plus'], marker='d', color='red', linewidth=2)
        axes[1, 1].set_title('Significant Counties (50+ Cranes) Over Time')
        axes[1, 1].set_xlabel('Year')
        axes[1, 1].set_ylabel('Number of Counties')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('migration_network_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Summary report saved as 'migration_network_summary.csv'")
        print("Summary visualization saved as 'migration_network_summary.png'")
    
    def run_complete_analysis(self):
        """
        Run the complete multi-year analysis.
        """
        print("Starting Multi-Year Sandhill Crane Migration Analysis")
        print("=" * 80)
        
        # Generate all networks
        self.generate_all_networks()
        
        # Create PNG maps for all years
        self.create_all_png_maps()
        
        # Create year-over-year comparison maps
        self.create_year_comparison_maps()
        
        # Generate summary report
        self.generate_summary_report()
        
        print("\n" + "=" * 80)
        print("Multi-Year Analysis Complete!")
        print("=" * 80)
        print("\nGenerated files:")
        print("- Migration networks: efficient_sandhill_crane_migration_network_YYYY.graphml")
        print("- PNG maps: sandhill_crane_migration_network_YYYY.png")
        print("- Comparison maps: sandhill_crane_change_YYYY_vs_YYYY.png")
        print("- Summary report: migration_network_summary.csv")
        print("- Summary visualization: migration_network_summary.png")

def main():
    """
    Main function to run the multi-year analysis.
    """
    analyzer = MultiYearMigrationAnalyzer(
        max_distance_km=500,
        max_time_days=14
    )
    
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
