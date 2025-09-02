import pandas as pd

def clean_data(input_filepath, output_filepath):
    """Loads raw data, standardizes columns, cleans it, and saves the result."""
    print("Cleaning data...")
    # Load the dataset
    df = pd.read_csv(input_filepath, encoding='latin1')

    # Standardize column names
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

    # Drop PII and unnecessary columns
    columns_to_drop = [
        'customer_email', 'customer_fname', 'customer_lname',
        'customer_password', 'customer_street', 'product_description'
    ]
    df.drop(columns=columns_to_drop, errors='ignore', inplace=True)

    # Clean data types and handle missing values
    df['order_date_dateorders'] = pd.to_datetime(df['order_date_dateorders'])
    df['customer_zipcode'] = df['customer_zipcode'].fillna(0)

    # Save the cleaned data
    df.to_csv(output_filepath, index=False)
    print(f"Cleaned data saved to {output_filepath}")
    return df