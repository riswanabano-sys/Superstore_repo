# Superstore_repo
End-to-end analysis of Tableau Superstore dataset with business insights and profit prediction. Includes EDA, visualizations, regression models, and a PDF report.
**Superstore Sales & Profit Analysis**

This project performs Exploratory Data Analysis (EDA) and Predictive Modeling on the Tableau Superstore dataset to derive business insights and build models to predict profitability.
**PROJECT STRUCTURE**
superstore_repo/
├── data/
│   └── Sample - Superstore.csv           # Dataset
├── reports/
│   └── Superstore_Analysis_Report.pdf    # Final PDF report with charts & insights
├── src/
│   └── analysis.py                       # Main Python script (EDA + ML)
├── requirements.txt                      # Python dependencies
├── README.md                             # Project overview (this file)
├── .gitignore
└── LICENSE
**HOW TO RUN**
# 1. Clone this repository
git clone https://github.com/<your-username>/superstore-analysis.git
cd superstore-analysis

# 2. (Optional) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # On Linux/Mac
.venv\Scripts\activate      # On Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run analysis
python src/analysis.py

**KEY FEATURES**
EDA
- Sales & profit trends (monthly, category, region, segment, ship mode)
- Discount impact on profitability
- Top profit-making and loss-making products

Predictive Modeling
- Linear Regression & Random Forest to predict Profit
- Evaluation with R², RMSE, MAE
- Feature importance ranking

Automated PDF Report
- Executive summary, charts, insights, and recommendations

**Business Insight**

Avoid heavy discounts (>30%) — they destroy profit margins.

Focus on Office Supplies (West region) and Technology (Corporate segment) for growth.

Furniture Tables often lead to losses → optimize pricing/discounting strategy.

Standard shipping is most profitable; upsell faster modes only for high-margin orders.

📌 Dataset

Source: Tableau Superstore dataset (educational use only)
Rows: 9,994
Columns: 21

