import pandas as pd

def create_features(df, output_filepath):
    """Creates new features from the cleaned data and saves the result."""
    print("Creating features...")
    # 1. Shipping Delay
    df['shipping_delay'] = df['days_for_shipping_real'] - df['days_for_shipment_scheduled']

    # 2. Profit Margin Ratio
    df['profit_margin_ratio'] = df['benefit_per_order'] / df['sales_per_customer'].replace(0, 1)

    # 3. Extract Time-Based Features
    df['order_year'] = df['order_date_dateorders'].dt.year
    df['order_month'] = df['order_date_dateorders'].dt.month
    df['order_weekday'] = df['order_date_dateorders'].dt.dayofweek
    
    # 4. ✅ NEW: Add Perfect Order Flag
    df['is_perfect_order'] = ((df['late_delivery_risk'] == 0) & (df['benefit_per_order'] > 0)).astype(int)

    # Save the feature-engineered data
    df.to_csv(output_filepath, index=False)
    print(f"Feature-engineered data saved to {output_filepath}")
    return df