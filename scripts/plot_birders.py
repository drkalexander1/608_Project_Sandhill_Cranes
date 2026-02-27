"""
Generate graph showing number of birders (observers) over time.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.data_processing import DataLoader
from src.season_detection import SeasonSplitter

def plot_birders_over_time(years=[2018, 2019, 2020, 2021, 2022, 2023], 
                          split_seasons=True, output_dir="results/main"):
    """
    Plot number of unique birders (observers) over time.
    
    Args:
        years: List of years to analyze
        split_seasons: If True, split by PreBreeding/PostBreeding
        output_dir: Output directory for plots
    """
    print("=" * 60)
    print("Analyzing Birder Participation Over Time")
    print("=" * 60)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    birder_stats = []
    
    for year in years:
        file_path = f'Antigone canadensis_all_checklist_{year}.parquet'
        if not os.path.exists(file_path):
            print(f"File {file_path} not found. Skipping.")
            continue
        
        df = DataLoader.load_and_filter_data(file_path)
        if df.empty:
            continue
        
        if 'observer_id' not in df.columns:
            print(f"Warning: observer_id column not found in {year} data")
            continue
        
        if split_seasons:
            splitter = SeasonSplitter()
            seasons = splitter.split_seasons(df, date_col='observation_date')
            
            for season_name, season_df in seasons.items():
                if season_df.empty:
                    continue
                
                unique_observers = season_df['observer_id'].nunique()
                total_observations = len(season_df)
                unique_checklists = season_df['checklist_id'].nunique() if 'checklist_id' in season_df.columns else 0
                
                birder_stats.append({
                    'year': year,
                    'season': season_name,
                    'unique_observers': unique_observers,
                    'total_observations': total_observations,
                    'unique_checklists': unique_checklists
                })
                
                print(f"{year} {season_name}: {unique_observers:,} unique observers, {total_observations:,} observations")
        else:
            unique_observers = df['observer_id'].nunique()
            total_observations = len(df)
            unique_checklists = df['checklist_id'].nunique() if 'checklist_id' in df.columns else 0
            
            birder_stats.append({
                'year': year,
                'season': 'FullYear',
                'unique_observers': unique_observers,
                'total_observations': total_observations,
                'unique_checklists': unique_checklists
            })
            
            print(f"{year}: {unique_observers:,} unique observers, {total_observations:,} observations")
    
    if not birder_stats:
        print("No birder data found.")
        return
    
    df_stats = pd.DataFrame(birder_stats)
    
    if split_seasons and 'season' in df_stats.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        years_sorted = sorted(df_stats['year'].unique())
        x = np.arange(len(years_sorted))
        width = 0.35
        
        pre_observers = []
        post_observers = []
        for y in years_sorted:
            pre_data = df_stats[(df_stats['year'] == y) & (df_stats['season'] == 'PreBreeding')]
            post_data = df_stats[(df_stats['year'] == y) & (df_stats['season'] == 'PostBreeding')]
            pre_observers.append(pre_data['unique_observers'].sum() if len(pre_data) > 0 else 0)
            post_observers.append(post_data['unique_observers'].sum() if len(post_data) > 0 else 0)
        
        bars1 = ax.bar(x - width/2, pre_observers, width, label='PreBreeding', alpha=0.8, color='steelblue')
        bars2 = ax.bar(x + width/2, post_observers, width, label='PostBreeding', alpha=0.8, color='coral')
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height):,}',
                            ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Number of Unique Observers', fontsize=12)
        ax.set_title('Number of Birders (Observers) Over Time', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(years_sorted)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        output_file = os.path.join(output_dir, "birders_over_time.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nSaved birder plot to {output_file}")
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twinx()
        
        pre_obs = []
        post_obs = []
        pre_counts = []
        post_counts = []
        for y in years_sorted:
            pre_data = df_stats[(df_stats['year'] == y) & (df_stats['season'] == 'PreBreeding')]
            post_data = df_stats[(df_stats['year'] == y) & (df_stats['season'] == 'PostBreeding')]
            pre_obs.append(pre_data['unique_observers'].sum() if len(pre_data) > 0 else 0)
            post_obs.append(post_data['unique_observers'].sum() if len(post_data) > 0 else 0)
            pre_counts.append(pre_data['total_observations'].sum() if len(pre_data) > 0 else 0)
            post_counts.append(post_data['total_observations'].sum() if len(post_data) > 0 else 0)
        
        bars1 = ax1.bar(x - width/2, pre_obs, width, label='PreBreeding Observers', alpha=0.6, color='steelblue')
        bars2 = ax1.bar(x + width/2, post_obs, width, label='PostBreeding Observers', alpha=0.6, color='coral')
        
        ax2.plot(x, pre_counts, marker='o', linewidth=2, label='PreBreeding Observations', color='darkblue')
        ax2.plot(x, post_counts, marker='s', linewidth=2, label='PostBreeding Observations', color='darkred')
        
        ax1.set_xlabel('Year', fontsize=12)
        ax1.set_ylabel('Number of Unique Observers', fontsize=12, color='black')
        ax2.set_ylabel('Total Observations', fontsize=12, color='black')
        ax1.set_title('Birder Participation vs Observation Activity', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(years_sorted)
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax1.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        output_file = os.path.join(output_dir, "birders_vs_observations.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved birder vs observations plot to {output_file}")
        
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        years_sorted = sorted(df_stats['year'].unique())
        observers = [df_stats[df_stats['year'] == y]['unique_observers'].sum() for y in years_sorted]
        
        bars = ax.bar(years_sorted, observers, alpha=0.8, color='steelblue')
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}',
                        ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Number of Unique Observers', fontsize=12)
        ax.set_title('Number of Birders (Observers) Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        output_file = os.path.join(output_dir, "birders_over_time.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nSaved birder plot to {output_file}")
    
    stats_file = os.path.join(output_dir, "birder_statistics.csv")
    df_stats.to_csv(stats_file, index=False)
    print(f"Saved birder statistics to {stats_file}")

if __name__ == "__main__":
    plot_birders_over_time()
