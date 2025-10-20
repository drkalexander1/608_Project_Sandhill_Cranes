import pandas as pd
import numpy as np

def analyze_parquet_file(filename):
    """Analyze the structure of a parquet file"""
    print(f"\n=== Analyzing {filename} ===")
    
    try:
        # Read the parquet file
        df = pd.read_parquet(filename)
        
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types:\n{df.dtypes}")
        
        # Show first few rows
        print(f"\nFirst 5 rows:")
        print(df.head())
        
        # Show basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"\nNumeric columns statistics:")
            print(df[numeric_cols].describe())
        
        # Show unique values for categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
            unique_count = df[col].nunique()
            print(f"\nColumn '{col}': {unique_count} unique values")
            if unique_count <= 20:
                print(f"Values: {df[col].unique()}")
            else:
                print(f"Sample values: {df[col].unique()[:10]}")
        
        return df
        
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

if __name__ == "__main__":
    # Analyze the 2018 data file
    df_2018 = analyze_parquet_file("Antigone canadensis_all_checklist_2018.parquet")
    
    if df_2018 is not None:
        print(f"\n=== Summary for Network Analysis ===")
        print(f"Total records: {len(df_2018)}")
        print(f"Available columns for potential network nodes/edges:")
        for col in df_2018.columns:
            print(f"  - {col}: {df_2018[col].dtype}")
