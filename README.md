# 📊 Telecom Customer Churn & LTV Analysis: Business-Focused Intelligence System

An end-to-end customer intelligence platform that translates predictive machine learning models into actionable, budget-optimized business decisions. This project uses the Telco Customer Churn dataset to predict customer churn, estimate Customer Lifetime Value (CLV), segment users, and run campaign simulations under budget constraints to maximize Return on Investment (ROI).

---

## 🚀 Key Business Innovations

* **Revenue at Risk vs. Simple Churn Risk**: Targeting customers based on churn risk alone is inefficient. A customer with a $95\%$ churn risk who spends $\$10/\text{month}$ is not worth a $\$20$ intervention. Conversely, a customer with a $\$1,000$ CLV and a $20\%$ risk represents a substantial financial threat ($\$200$ Revenue at Risk) and should be prioritized. This platform solves the "churn blind spot" by combining risk and lifetime value:
  $$\text{Revenue at Risk} = P(\text{Churn}) \times \text{CLV}$$
* **Expected Saved Value**: The system accounts for real-world campaign failure rates by scaling Revenue at Risk by the intervention's success rate:
  $$\text{Expected Saved Value} = \text{Revenue at Risk} \times \text{Campaign Success Rate}$$
* **ROI-Driven Budget Allocation**: We rank customers by expected campaign ROI descending and allocate the retention budget dynamically, ensuring 100% budget efficiency and eliminating subsidy waste.
  $$\text{ROI} = \frac{\text{Expected Saved Value} - \text{Intervention Cost}}{\text{Intervention Cost}}$$

---

## 📁 Repository Structure

```directory
.
├── 01_data_ingestion_audit.ipynb      # Initial exploratory analysis and data quality auditing
├── 02_data_cleaning.ipynb             # Data preprocessing, handling missing values & type conversion
├── 03_feature_engineering.ipynb        # Engineering behavioral, RFM-inspired, and tenure indicators
├── 04_clv_analysis.ipynb              # CLV estimation modeling and scoring logic
├── WA_Fn-UseC_-Telco-Customer-Churn.csv # Raw Telco dataset
├── app.py                             # Interactive Streamlit dashboard code
├── business_impact.py                 # Core business logic, campaign simulator, & ROI calculator
├── final_customer_action_list.csv     # Model outputs merged with segment classifications and decisions
└── requirements.txt                   # Dependency list
```

---

## 🛠️ Data & ML Pipeline

The pipeline is split into six sequential phases across the Jupyter notebooks:

1. **Ingestion & Audit (`01_data_ingestion_audit.ipynb`)**
   * Inspects the raw dataset and audits data types, distributions, and cardinality.
2. **Data Cleaning (`02_data_cleaning.ipynb`)**
   * Coerces numerical values (e.g., converting `TotalCharges` from string to numeric), imputes missing records, and standardizes variables.
3. **Feature Engineering (`03_feature_engineering.ipynb`)**
   * Creates behavioral features including tenure bins, contract ordinal values, trial-period flags, and internet service indicators.
   * Performs scaling on numerical variables and encodes categorical predictors.
4. **Churn Prediction Modeling**
   * Explores baseline Logistic Regression and XGBoost classifiers.
   * Generates churn probabilities ($P(\text{Churn})$) for every active customer account.
5. **Customer Lifetime Value Layer (`04_clv_analysis.ipynb`)**
   * Establishes a CLV estimation layer based on contract terms, monthly fees, tenure, and services utilized to score customer value on a scale of $0.0 - 1.0$.
6. **Decision Engine & Segment Classification**
   * Combines Churn Risk and CLV to categorize customer profiles into strategic segments:
     * **Retention Target** (High Risk, High CLV) $\rightarrow$ *Offer Discount*
     * **Loyalty Target** (Low Risk, High Tenure) $\rightarrow$ *Upsell / Reward*
     * **Non-Target** (High Risk, Low CLV) $\rightarrow$ *No Action*
     * **General Monitoring** (All others) $\rightarrow$ *Standard Nurture*

---

## 🖥️ Streamlit Interactive Dashboard

The project includes a production-ready **Streamlit dashboard** (`app.py`) designed for marketing operators, finance leaders, and product teams. 

### Features:
1. **Interactive Data Ingestion**: Upload new customer CSVs to run them automatically through the ML inference pipeline.
2. **Business Impact Customizer**: Dynamically configure the CLV dollar scaling factor, total campaign budget, and individual intervention parameters (costs and success rates for discounts, rewards, nurture campaigns).
3. **Executive Dashboard & KPIs**: Visualizes total revenue at risk, campaign target group size, expected value saved, and overall projected campaign ROI.
4. **CRM Export Suite**: Filter and segment customers based on budget inclusion, minimum ROI thresholds, or strategic segments, then export targeted customer lists directly for upload to CRMs (like Salesforce or HubSpot).
5. **Explainability & Individual Drill-Down**: Analyze global model feature coefficients alongside individual customer cards showing risk drivers (e.g. month-to-month contracts, fiber optic upgrades) and corresponding business action cards.

---

## ⚙️ Installation & Usage

### 1. Set Up Environment
Create a virtual environment and install dependencies:
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Run the Jupyter Notebooks
Open the notebooks to review the step-by-step model building, feature engineering, and evaluation phases:
```bash
jupyter notebook
```

### 3. Launch the Dashboard
Run the Streamlit application to explore the dashboard interface:
```bash
streamlit run app.py
```

---

## 📊 Business Logic Configuration Reference
The default business parameters used in the ROI calculations are:
* **Retention Discount**: Cost: $\$150.00$, Success Rate: $15\%$
* **Loyalty Reward**: Cost: $\$100.00$, Success Rate: $10\%$
* **Standard Nurture**: Cost: $\$10.00$, Success Rate: $5\%$
* **No Action**: Cost: $\$0.00$, Success Rate: $0\%$

---

## 🤝 Contributing & Feedback

Contributions, feature requests, and feedback are highly welcome! 

1. **Fork the Repository**
2. **Create a Feature Branch**: `git checkout -b feature/AmazingFeature`
3. **Commit your Changes**: `git commit -m 'Add some AmazingFeature'`
4. **Push to the Branch**: `git push origin feature/AmazingFeature`
5. **Open a Pull Request**

For major changes, please open an issue first to discuss what you would like to change.

