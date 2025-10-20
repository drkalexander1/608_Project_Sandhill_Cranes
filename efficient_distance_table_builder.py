"""
Efficient Distance Table Builder for Sandhill Crane Migration Networks
====================================================================

This script precomputes distances between all possible county pairs
to avoid recalculating distances for each year's network construction.
"""

import pandas as pd
import numpy as np
from geopy.distance import geodesic
import warnings
import os
from itertools import combinations
import pickle

warnings.filterwarnings('ignore')

class DistanceTableBuilder:
    """
    Builds a precomputed distance table for all county pairs.
    """
    
    def __init__(self):
        """
        Initialize the distance table builder.
        """
        self.distance_table = {}
        self.county_coordinates = {}
        
    def extract_all_counties(self):
        """
        Extract all unique counties from all parquet files.
        """
        print("Extracting all unique counties from all years...")
        
        all_counties = set()
        years = [2018, 2019, 2020, 2021, 2022, 2023]
        
        for year in years:
            parquet_file = f'Antigone canadensis_all_checklist_{year}.parquet'
            if os.path.exists(parquet_file):
                print(f"Processing {parquet_file}...")
                df = pd.read_parquet(parquet_file)
                
                # Filter for sandhill crane observations only
                if 'scientific_name' in df.columns:
                    df = df[df['scientific_name'] == 'Antigone canadensis']
                
                # Filter out zero-crane observations
                df = df[df['observation_count'] > 0]
                
                # Extract unique counties with coordinates
                county_data = df.groupby('county_code').agg({
                    'latitude': 'mean',
                    'longitude': 'mean',
                    'observation_count': 'sum'
                }).reset_index()
                
                # Store county coordinates
                for _, row in county_data.iterrows():
                    county_code = row['county_code']
                    if pd.notna(county_code) and pd.notna(row['latitude']) and pd.notna(row['longitude']):
                        all_counties.add(county_code)
                        self.county_coordinates[county_code] = {
                            'lat': row['latitude'],
                            'lon': row['longitude'],
                            'total_cranes': row['observation_count']
                        }
        
        print(f"Found {len(all_counties)} unique counties across all years")
        return list(all_counties)
    
    def build_distance_table(self, max_distance_km=500):
        """
        Build the distance table for all county pairs within max_distance_km.
        """
        print("Building distance table...")
        
        # Get all counties
        counties = self.extract_all_counties()
        
        # Convert to list for consistent ordering
        county_list = sorted(counties)
        n_counties = len(county_list)
        
        print(f"Calculating distances for {n_counties} counties...")
        print(f"Total possible pairs: {n_counties * (n_counties - 1) // 2}")
        
        # Initialize distance table
        self.distance_table = {}
        
        # Calculate distances for all pairs
        calculated = 0
        for i, county1 in enumerate(county_list):
            if county1 not in self.county_coordinates:
                continue
                
            coords1 = (self.county_coordinates[county1]['lat'], 
                      self.county_coordinates[county1]['lon'])
            
            for j, county2 in enumerate(county_list):
                if i >= j or county2 not in self.county_coordinates:
                    continue
                
                coords2 = (self.county_coordinates[county2]['lat'], 
                          self.county_coordinates[county2]['lon'])
                
                # Calculate distance
                distance = geodesic(coords1, coords2).kilometers
                
                # Only store if within max distance
                if distance <= max_distance_km:
                    pair_key = tuple(sorted([county1, county2]))
                    self.distance_table[pair_key] = distance
                
                calculated += 1
                if calculated % 1000 == 0:
                    print(f"Calculated {calculated} pairs...")
        
        print(f"Distance table built with {len(self.distance_table)} pairs within {max_distance_km}km")
        
        # Add self-distances (0) for each county
        for county in county_list:
            self.distance_table[(county, county)] = 0.0
    
    def get_distance(self, county1, county2):
        """
        Get distance between two counties from the precomputed table.
        """
        pair_key = tuple(sorted([county1, county2]))
        return self.distance_table.get(pair_key, float('inf'))
    
    def save_distance_table(self, filename='county_distance_table.pkl'):
        """
        Save the distance table to a file.
        """
        print(f"Saving distance table to {filename}...")
        
        data_to_save = {
            'distance_table': self.distance_table,
            'county_coordinates': self.county_coordinates
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data_to_save, f)
        
        print(f"Distance table saved successfully!")
        
        # Also save as CSV for inspection
        csv_data = []
        for (county1, county2), distance in self.distance_table.items():
            if county1 != county2:  # Skip self-distances
                csv_data.append({
                    'county1': county1,
                    'county2': county2,
                    'distance_km': distance
                })
        
        csv_df = pd.DataFrame(csv_data)
        csv_filename = filename.replace('.pkl', '.csv')
        csv_df.to_csv(csv_filename, index=False)
        print(f"Distance table also saved as CSV: {csv_filename}")
    
    def load_distance_table(self, filename='county_distance_table.pkl'):
        """
        Load the distance table from a file.
        """
        print(f"Loading distance table from {filename}...")
        
        if not os.path.exists(filename):
            print(f"Distance table file {filename} not found!")
            return False
        
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        self.distance_table = data['distance_table']
        self.county_coordinates = data['county_coordinates']
        
        print(f"Distance table loaded successfully!")
        print(f"Loaded {len(self.distance_table)} distance pairs")
        print(f"Loaded coordinates for {len(self.county_coordinates)} counties")
        
        return True
    
    def get_county_coordinates(self, county_code):
        """
        Get coordinates for a specific county.
        """
        return self.county_coordinates.get(county_code, None)
    
    def get_all_counties(self):
        """
        Get list of all counties in the distance table.
        """
        return list(self.county_coordinates.keys())

def main():
    """
    Main function to build the distance table.
    """
    print("Building Efficient Distance Table for Sandhill Crane Migration Networks")
    print("=" * 80)
    
    builder = DistanceTableBuilder()
    
    # Check if distance table already exists
    if os.path.exists('county_distance_table.pkl'):
        print("Distance table already exists. Loading...")
        builder.load_distance_table()
    else:
        print("Building new distance table...")
        builder.build_distance_table(max_distance_km=500)
        builder.save_distance_table()
    
    print("\nDistance table ready for use!")
    print("=" * 80)

if __name__ == "__main__":
    main()
