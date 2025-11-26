import pandas as pd
from datetime import datetime

class SeasonSplitter:
    """
    Handles separating data into migration seasons (Pre-breeding/North vs Post-breeding/South).
    """
    
    def __init__(self, pre_breeding_start="01-01", pre_breeding_end="07-31", 
                 post_breeding_start="08-01", post_breeding_end="12-31"):
        """
        Initialize with date ranges. Format: "MM-DD"
        Default: Before August (pre-breeding) and After August (post-breeding)
        """
        self.pre_breeding_start = pre_breeding_start
        self.pre_breeding_end = pre_breeding_end
        self.post_breeding_start = post_breeding_start
        self.post_breeding_end = post_breeding_end

    def split_seasons(self, df, date_col='observation_date'):
        """
        Split the dataframe into Pre-breeding (before August) and Post-breeding (after August) datasets.
        
        Args:
            df (pd.DataFrame): Input dataframe with a datetime column.
            date_col (str): Name of the date column.
            
        Returns:
            dict: {'PreBreeding': pd.DataFrame, 'PostBreeding': pd.DataFrame}
        """
        print(f"Splitting data into seasons based on {date_col}...")
        
        if df.empty:
            return {'PreBreeding': pd.DataFrame(), 'PostBreeding': pd.DataFrame()}
            
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])
            
        # Parse cutoff dates
        pre_s_m, pre_s_d = map(int, self.pre_breeding_start.split('-'))
        pre_e_m, pre_e_d = map(int, self.pre_breeding_end.split('-'))
        post_s_m, post_s_d = map(int, self.post_breeding_start.split('-'))
        post_e_m, post_e_d = map(int, self.post_breeding_end.split('-'))
        
        # Create masks
        def is_in_range(date_series, start_m, start_d, end_m, end_d):
            # Convert to month-day integer for comparison (MMDD)
            md = date_series.dt.month * 100 + date_series.dt.day
            start_md = start_m * 100 + start_d
            end_md = end_m * 100 + end_d
            
            if start_md <= end_md:
                return (md >= start_md) & (md <= end_md)
            else: # Range wraps around year end (unlikely for this specific use case but good for robustness)
                return (md >= start_md) | (md <= end_md)

        pre_breeding_mask = is_in_range(df[date_col], pre_s_m, pre_s_d, pre_e_m, pre_e_d)
        post_breeding_mask = is_in_range(df[date_col], post_s_m, post_s_d, post_e_m, post_e_d)
        
        pre_breeding_df = df[pre_breeding_mask].copy()
        post_breeding_df = df[post_breeding_mask].copy()
        
        print(f"Pre-breeding (before August) observations: {len(pre_breeding_df):,}")
        print(f"Post-breeding (after August) observations: {len(post_breeding_df):,}")
        
        return {'PreBreeding': pre_breeding_df, 'PostBreeding': post_breeding_df}

