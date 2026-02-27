"""
Run Efficient Multi-Year Sandhill Crane Migration Analysis
========================================================

This script first builds the distance table, then runs the multi-year analysis.
"""

import os
import sys
from src.distance_table import DistanceTableBuilder
from multi_year_migration_analysis import MultiYearMigrationAnalyzer

def main():
    """
    Main function to run the complete efficient analysis.
    """
    print("Starting Efficient Multi-Year Sandhill Crane Migration Analysis")
    print("=" * 80)
    
    # Step 1: Build distance table if it doesn't exist
    print("\nStep 1: Building/Checking Distance Table")
    print("-" * 40)
    
    distance_builder = DistanceTableBuilder()
    
    if os.path.exists('county_distance_table.pkl'):
        print("Distance table already exists. Loading...")
        distance_builder.load_distance_table()
    else:
        print("Building new distance table...")
        distance_builder.build_distance_table(max_distance_km=500)
        distance_builder.save_distance_table()
    
    print("Distance table ready!")
    
    # Step 2: Run multi-year analysis
    print("\nStep 2: Running Multi-Year Analysis")
    print("-" * 40)
    
    analyzer = MultiYearMigrationAnalyzer(
        max_distance_km=500,
        max_time_days=14
    )
    
    analyzer.run_complete_analysis()
    
    print("\n" + "=" * 80)
    print("Efficient Analysis Complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
