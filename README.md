# 📦 Supply Chain Optimization & Risk Management Dashboard

A comprehensive **data-driven dashboard** built with **Python** and **Streamlit** to analyze supply chain performance, predict late delivery risks, and provide actionable insights for logistics management.

---

## 🚀 Live Demo  
[Add your Streamlit Community Cloud deployment link here]

---

## ✨ Key Features
- **Executive KPIs**: Quick insights with OTIF (On-Time-In-Full) Rate & Perfect Order Rate.  
- **Interactive Analytics**: Dynamic charts & geographical heatmaps to spot high-risk regions.  
- **ML-Powered Risk Prediction**: Real-time late delivery risk detection with Random Forest.  
- **Optimal Shipping Recommendation**: Suggests best shipping mode balancing risk & profit.  
- **Dynamic Filtering & Export**: Filter by region, mode, date & export results as CSV.  

---

## 📸 Dashboard Preview  
(Add a screenshot of your running dashboard here)

---

## 📂 Project Structure
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

---

## 🛠 Tech Stack
- **Backend**: Python, Pandas, Scikit-learn  
- **Machine Learning**: Random Forest (classification), K-Means (segmentation), Prophet (forecasting)  
- **Dashboard**: Streamlit  
- **Visualization**: Plotly Express  

---

## ⚙️ How to Run Locally
```bash
# Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# Create & activate virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run data pipeline (process data + train ML models)
python run_pipeline.py

# Launch dashboard
streamlit run dashboard/app.py
