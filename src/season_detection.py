import pandas as pd
from datetime import datetime

class SeasonSplitter:
    """
    Handles separating data into migration seasons (Spring/North vs Fall/South).
    """
    
    def __init__(self, spring_start_date="01-01", spring_end_date="07-15", 
                 fall_start_date="07-16", fall_end_date="12-31"):
        """
        Initialize with date ranges. Format: "MM-DD"
        """
        self.spring_start = spring_start_date
        self.spring_end = spring_end_date
        self.fall_start = fall_start_date
        self.fall_end = fall_end_date

    def split_seasons(self, df, date_col='observation_date'):
        """
        Split the dataframe into Spring and Fall datasets based on the date column.
        
        Args:
            df (pd.DataFrame): Input dataframe with a datetime column.
            date_col (str): Name of the date column.
            
        Returns:
            dict: {'Spring': pd.DataFrame, 'Fall': pd.DataFrame}
        """
        print(f"Splitting data into seasons based on {date_col}...")
        
        if df.empty:
            return {'Spring': pd.DataFrame(), 'Fall': pd.DataFrame()}
            
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])
            
        # We need to handle multiple years if present, but typically we process one year at a time.
        # However, the mask should work on month-day regardless of year.
        
        # Parse cutoff dates
        spring_s_m, spring_s_d = map(int, self.spring_start.split('-'))
        spring_e_m, spring_e_d = map(int, self.spring_end.split('-'))
        fall_s_m, fall_s_d = map(int, self.fall_start.split('-'))
        fall_e_m, fall_e_d = map(int, self.fall_end.split('-'))
        
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

        spring_mask = is_in_range(df[date_col], spring_s_m, spring_s_d, spring_e_m, spring_e_d)
        fall_mask = is_in_range(df[date_col], fall_s_m, fall_s_d, fall_e_m, fall_e_d)
        
        spring_df = df[spring_mask].copy()
        fall_df = df[fall_mask].copy()
        
        print(f"Spring observations: {len(spring_df):,}")
        print(f"Fall observations: {len(fall_df):,}")
        
        return {'Spring': spring_df, 'Fall': fall_df}

