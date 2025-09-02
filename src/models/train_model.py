import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

def train_and_save_pipeline(df, model_output_filepath):
    """Trains a delivery risk prediction pipeline and saves it."""
    print("Training model pipeline...")
    # Define features (X) and target (y)
    target = 'late_delivery_risk'
    features = [
        'days_for_shipment_scheduled', 'benefit_per_order', 'sales_per_customer',
        'category_name', 'customer_segment', 'market', 'order_region',
        'shipping_mode', 'order_month', 'order_weekday'
    ]
    X = df[features]
    y = df[target]

    # Split data
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Define preprocessing steps
    categorical_features = ['category_name', 'customer_segment', 'market', 'order_region', 'shipping_mode']
    numerical_features = ['days_for_shipment_scheduled', 'benefit_per_order', 'sales_per_customer', 'order_month', 'order_weekday']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Define the model
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)

    # Create the full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # Train the pipeline
    pipeline.fit(X_train, y_train)

    # Save the pipeline
    joblib.dump(pipeline, model_output_filepath)
    print(f"Model pipeline saved to {model_output_filepath}")