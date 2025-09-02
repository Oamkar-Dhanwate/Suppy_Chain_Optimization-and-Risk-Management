Supply Chain Optimization & Risk Management Dashboard
A comprehensive, data-driven dashboard built with Python and Streamlit to analyze supply chain performance, predict late delivery risks, and provide actionable insights for logistics management.

🚀 Live Demo
AddyourStreamlitCommunityClouddeploymentlinkhereonceit 
′
 slive
✨ Key Features
Executive KPIs: At-a-glance metrics including On-Time-In-Full (OTIF) Rate and Perfect Order Rate for a quick overview of supply chain health.

Interactive Analytics: Dynamic charts and a geographical heatmap to visualize performance, identify high-risk regions, and analyze profitability.

ML-Powered Risk Prediction: A real-time tool, powered by a Random Forest model, to predict the late delivery risk for new orders.

Optimal Shipping Recommendation: An intelligent feature that recommends the best shipping mode based on a balance of delivery risk and profitability.

Dynamic Filtering & Data Export: Users can filter data by region, shipping mode, and date, and export the results to a CSV file.

📸 Dashboard Preview
Addascreenshotofyourrunningdashboardhere.Thisisagreatwaytoshowcaseyourwork!
📂 Project Structure
supply_chain_optimization/
├── 📂 dashboard/
│   └── 📄 app.py              # Main Streamlit application
├── 📂 data/
│   └── 📄 raw/                 # Raw dataset
├── 📂 notebooks/                # Jupyter notebooks for analysis & modeling
├── 📂 src/                      # Source code for the data pipeline
│   ├── 📦 data/
│   ├── 📦 features/
│   └── 📦 models/
├── 📄 run_pipeline.py           # Master script to run the backend
└── 📄 requirements.txt           # Project dependencies



🛠 Tech Stack
Backend: Python, Pandas, Scikit-learn

Machine Learning: Random Forest (for classification), K-Means (for segmentation), Prophet (for forecasting)

Dashboard: Streamlit

Plotting: Plotly Express

⚙ How to Run Locally
Clone the repository:

git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name



Set up the environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt



Run the data pipeline:
This script processes the raw data and trains the ML models. This step is crucial as it generates the files the dashboard depends on.

python run_pipeline.py



Launch the dashboard:

streamlit run dashboard/app.py



📊 Data Source
This project uses the "DataCo Global Supply Chain" dataset, which is publicly available on Kaggle.

📄 License
This project is licensed under the MIT License.
