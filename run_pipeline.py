from src.data.make_dataset import clean_data
from src.features.build_features import create_features
from src.features.add_segments import add_customer_segments # ✅ NEW IMPORT
from src.models.train_model import train_and_save_pipeline

# Define file paths
RAW_DATA_PATH = 'data/raw/DataCoSupplyChainDataset.csv'
CLEANED_DATA_PATH = 'data/processed/cleaned_orders.csv'
FEATURES_DATA_PATH = 'data/processed/features.csv' # Renamed for clarity
SEGMENTS_PATH = 'data/processed/customer_segments.csv' # ✅ NEW PATH
FINAL_DATA_PATH = 'data/processed/final_data_with_segments.csv' # ✅ NEW FINAL PATH
MODEL_PATH = 'models/delivery_risk_pipeline.joblib'

if __name__ == '__main__':
    # Step 1: Clean the raw data
    cleaned_df = clean_data(RAW_DATA_PATH, CLEANED_DATA_PATH)
    
    # Step 2: Create features from cleaned data
    features_df = create_features(cleaned_df, FEATURES_DATA_PATH)
    
    # Step 3: ✅ NEW - Add customer segments to the feature dataset
    # Note: This step assumes you have already run the 5.0 notebook once to create customer_segments.csv
    final_df = add_customer_segments(FEATURES_DATA_PATH, SEGMENTS_PATH, FINAL_DATA_PATH)
    
    # Step 4: Train the model on the final, enriched data
    train_and_save_pipeline(final_df, MODEL_PATH)
    
    print("\n--- Pipeline execution complete! ---")