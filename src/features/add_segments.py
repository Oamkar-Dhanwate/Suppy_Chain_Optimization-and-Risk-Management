import pandas as pd

def add_customer_segments(features_df_path, segments_df_path, output_filepath):
    """Merges customer segment labels into the main feature dataset."""
    print("Adding customer segments...")
    
    # Load the main features and the customer segments
    features_df = pd.read_csv(features_df_path)
    segments_df = pd.read_csv(segments_df_path)
    
    # Merge the segments into the main dataframe
    # We use a left merge to ensure we keep all the original orders
    final_df = pd.merge(features_df, segments_df, on='customer_id', how='left')
    
    # Fill any potential missing cluster values (for safety)
    final_df['cluster'].fillna(-1, inplace=True) # Use -1 for "Uncategorized"
    
    # Save the final, enriched dataset
    final_df.to_csv(output_filepath, index=False)
    print(f"Final data with segments saved to {output_filepath}")
    return final_df