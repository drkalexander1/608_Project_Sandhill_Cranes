import pandas as pd
import numpy as np
import warnings

class DataLoader:
    """
    Handles loading and processing of Sandhill Crane observation data.
    """

    @staticmethod
    def load_and_filter_data(parquet_file):
        """
        Load data and filter out zero-crane observations immediately.
        
        Args:
            parquet_file (str): Path to the parquet file.
            
        Returns:
            pd.DataFrame: Filtered dataframe.
        """
        print(f"Loading data from {parquet_file}...")
        try:
            df = pd.read_parquet(parquet_file)
        except Exception as e:
            print(f"Error loading file: {e}")
            return pd.DataFrame()
        
        # Filter for sandhill crane observations only
        if 'scientific_name' in df.columns:
            df = df[df['scientific_name'] == 'Antigone canadensis']
        
        print(f"Total observations before filtering: {len(df):,}")
        
        # CRITICAL: Filter out zero-crane observations immediately
        if 'observation_count' in df.columns:
            df = df[df['observation_count'] > 0]
            print(f"Observations with cranes: {len(df):,}")
        else:
            print("Warning: 'observation_count' column not found.")
        
        # Convert date columns if needed
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
        return df
    
    @staticmethod
    def aggregate_by_county(df):
        """
        Aggregate observations by FIPS county code.
        
        Args:
            df (pd.DataFrame): Raw observation dataframe.
            
        Returns:
            pd.DataFrame: Aggregated county data.
        """
        print("Aggregating observations by county...")
        
        if df.empty:
            print("Warning: Empty dataframe provided for aggregation.")
            return pd.DataFrame()

        # Group by county and aggregate
        county_data = df.groupby('county_code').agg({
            'observation_count': 'sum',           # Total cranes per county
            'latitude': 'mean',                   # County center coordinates
            'longitude': 'mean',
            'observation_date': ['min', 'max'],   # First and last observation dates
            'observer_id': 'nunique',             # Number of unique observers
            'checklist_id': 'nunique'             # Number of checklists
        }).reset_index()
        
        # Flatten column names
        county_data.columns = [
            'county_code', 'total_cranes', 'lat', 'lon', 
            'first_observation', 'last_observation', 
            'num_observers', 'num_checklists'
        ]
        
        # Calculate observation duration
        county_data['observation_duration'] = (
            county_data['last_observation'] - county_data['first_observation']
        ).dt.days
        
        print(f"Aggregated to {len(county_data)} counties with cranes")
        return county_data

