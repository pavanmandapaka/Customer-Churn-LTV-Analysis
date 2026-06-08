import streamlit as st
import pandas as pd
import plotly.express as px
import time

# -----------------------------------------------------------------------------
# 1. Page Config & Title
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn & LTV Analysis Dashboard", 
    page_icon="📊", 
    layout="wide"
)

st.title("Customer Churn & LTV Analysis Dashboard")

# -----------------------------------------------------------------------------
# Modularized Pipeline Function (Phases 1-6)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_end_to_end_pipeline(raw_df):
    """
    Executes the entire data science pipeline on a newly uploaded raw dataset.
    """
    # Standardize column names (strip whitespace and handle casing)
    df_processed = raw_df.copy()
    df_processed.columns = [str(c).strip() for c in df_processed.columns]
    
    # Map common variations of required columns case-insensitively
    column_mapping = {}
    for col in df_processed.columns:
        col_lower = col.lower()
        if col_lower in ['customerid', 'customer_id', 'customer id']:
            column_mapping[col] = 'customerID'
        elif col_lower in ['tenure', 'tenure_months', 'tenure months']:
            column_mapping[col] = 'tenure'
        elif col_lower in ['monthlycharges', 'monthly charges', 'monthly_charges']:
            column_mapping[col] = 'MonthlyCharges'
        elif col_lower in ['totalcharges', 'total charges', 'total_charges']:
            column_mapping[col] = 'TotalCharges'
            
    df_processed = df_processed.rename(columns=column_mapping)
    
    # 2 & 5. Validate dataset
    required_cols = ['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges']
    missing_cols = [c for c in required_cols if c not in df_processed.columns]
    
    if missing_cols:
        st.error(f"❌ Invalid CSV. The following required columns are missing: {missing_cols}")
        return pd.DataFrame()

    with st.status("Running Data Science Pipeline...", expanded=True) as status:
        # Step 1: Preprocessing & Cleaning (Phase 2)
        st.write("⚙️ 1. Cleaning and preprocessing data...")
        df_clean = df_processed.copy()
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce').fillna(0)
        time.sleep(0.5) # Simulated process time
        
        # Step 2: Feature Engineering (Phase 3)
        st.write("🛠️ 2. Engineering features...")
        # Note: Insert actual feature engineering (one-hot encoding, scaling) here
        time.sleep(0.5)
        
        # Step 3: Model Inferences (Phases 4 & 5)
        st.write("🤖 3. Predicting Churn Risk & CLV...")
        # Note: You need to save your models in the notebook using `joblib.dump(model, 'model.pkl')`
        # and load them here: `model = joblib.load('churn_model.pkl')`
        
        # FOR NOW: Simulating model outputs to allow dashboard testing
        import numpy as np
        np.random.seed(42)
        df_clean['churn_probability'] = np.random.uniform(0, 1, size=len(df_clean))
        df_clean['clv_score'] = np.random.uniform(0, 1, size=len(df_clean))
        
        # We need engagement_level for the decision engine
        df_clean['engagement_level'] = (df_clean['tenure'] / df_clean['tenure'].max()) 
        time.sleep(0.5)

        # Step 4: Decision Engine (Phase 6)
        st.write("🧠 4. Applying Decision Engine rules...")
        
        # Calculate Segment thresholds (Tertiles for CLV)
        df_clean['clv_segment'] = pd.qcut(df_clean['clv_score'], q=[0, 0.33, 0.66, 1.0], labels=['Low CLV', 'Medium CLV', 'High CLV'])
        
        def assign_logic(row):
            if row['churn_probability'] > 0.50 and row['clv_segment'] == 'High CLV':
                return pd.Series(['Retention Target', 'Offer Discount'])
            elif row['churn_probability'] > 0.50 and row['clv_segment'] == 'Low CLV':
                return pd.Series(['Non-Target', 'No Action'])
            elif row['churn_probability'] < 0.30 and row['engagement_level'] > 0.50:
                return pd.Series(['Loyalty/Upsell Target', 'Upsell / Reward'])
            else:
                return pd.Series(['General Monitoring', 'Standard Nurture'])
                
        df_clean[['customer_segment', 'recommended_action']] = df_clean.apply(assign_logic, axis=1)
        df_clean['priority_score'] = df_clean['churn_probability'] * df_clean['clv_score']
        
        status.update(label="✅ Pipeline execution complete!", state="complete", expanded=False)
        
    return df_clean

# -----------------------------------------------------------------------------
# Default Data Loader
# -----------------------------------------------------------------------------
@st.cache_data
def load_default_data():
    try:
        return pd.read_csv('final_customer_action_list.csv')
    except FileNotFoundError:
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# SIDEBAR: File Uploader Worklflow
# -----------------------------------------------------------------------------
st.sidebar.header("Data Source")

# 1. Add Streamlit file uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload New Customer Data (CSV)", 
    type=["csv"],
    help="Upload raw telco data. It will automatically run through the ML pipeline."
)

st.sidebar.divider()
st.sidebar.header("Dashboard Navigation")
st.sidebar.info("Use this dashboard to monitor churn risk, Customer Lifetime Value (CLV), and prioritized business actions.")

# 3 & 4. Execute Workflow based on upload
if uploaded_file is not None:
    try:
        # Read the uploaded file
        raw_csv = pd.read_csv(uploaded_file)
        st.sidebar.success("File uploaded successfully!")
        
        # Run the modularized pipeline on the new data
        df = run_end_to_end_pipeline(raw_csv)
    except Exception as e:
        st.sidebar.error(f"❌ Error parsing CSV file: {str(e)}")
        df = pd.DataFrame()
else:
    # Fallback to the default saved Phase 6 data
    df = load_default_data()

# =============================================================================
# DASHBOARD UI COMPONENTS (Overview, KPIs, Visualizations, Insights)
# =============================================================================

st.header("Overview")
st.write("""
This dashboard provides a unified view of customer health. It combines predictive churn 
probabilities with Customer Lifetime Value (CLV) scores to recommend prioritized, 
ROI-positive interventions for the marketing and retention teams.
""")

st.divider()

if not df.empty:
    # -------------------------------------------------------------------------
    # KPIs Section
    # -------------------------------------------------------------------------
    st.header("KPIs")
    total_customers = len(df)
    avg_churn_prob = df['churn_probability'].mean()
    avg_clv_score = df['clv_score'].mean()
    retention_targets = len(df[df['customer_segment'] == 'Retention Target'])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Customers", value=f"{total_customers:,}")
    with col2:
        st.metric(label="Avg Churn Risk", value=f"{avg_churn_prob:.1%}")
    with col3:
        st.metric(label="Avg CLV Score", value=f"{avg_clv_score:.2f}")
    with col4:
        st.metric(label="High-Priority Targets", value=f"{retention_targets:,}")

    st.divider()

    # -------------------------------------------------------------------------
    # Visualizations Section
    # -------------------------------------------------------------------------
    st.header("Visualizations")
    st.write("Explore customer distributions and the relationship between churn risk and lifetime value.")
    
    col_hist1, col_hist2 = st.columns(2)
    with col_hist1:
        fig_churn = px.histogram(df, x='churn_probability', nbins=30, title='Distribution of Churn Risk', color_discrete_sequence=['#EF553B'])
        fig_churn.update_layout(bargap=0.1)
        st.plotly_chart(fig_churn, use_container_width=True)
        st.caption("**How to read this:** A spike on the right indicates a large group of flight-risk customers.")

    with col_hist2:
        fig_clv = px.histogram(df, x='clv_score', nbins=30, title='Distribution of CLV Scores', color_discrete_sequence=['#00CC96'])
        fig_clv.update_layout(bargap=0.1)
        st.plotly_chart(fig_clv, use_container_width=True)
        st.caption("**How to read this:** Higher scores mean we have highly profitable, engaged customers.")
        
    st.write("<br>", unsafe_allow_html=True)
    col_bar, col_scatter = st.columns(2)
    
    with col_bar:
        segment_counts = df['customer_segment'].value_counts().reset_index()
        segment_counts.columns = ['Segment', 'Count']
        fig_segments = px.bar(segment_counts, x='Segment', y='Count', title='Customer Strategy Breakdown', color='Segment', text_auto=True)
        st.plotly_chart(fig_segments, use_container_width=True)
        st.caption("**How to read this:** Summarizes how many customers fall into our distinct intervention buckets.")

    with col_scatter:
        fig_scatter = px.scatter(df, x='churn_probability', y='clv_score', color='customer_segment', title='Risk vs. Reward (Intervention Matrix)', opacity=0.7, hover_data=['recommended_action'])
        fig_scatter.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
        fig_scatter.add_vline(x=0.5, line_dash="dash", line_color="gray", opacity=0.5)
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption("**How to read this:** Top Right = Save them. Bottom Right = Let them go organically.")

    st.divider()

    # -------------------------------------------------------------------------
    # Retention Prioritization View
    # -------------------------------------------------------------------------
    st.header("Retention Prioritization View")
    st.write("Filter and download prioritized customer lists for targeted marketing and retention campaigns.")

    st.subheader("1. Filter Customers")
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        all_segments = df['customer_segment'].unique().tolist()
        selected_segments = st.multiselect("Select Customer Segments", options=all_segments, default=all_segments)
    with filter_col2:
        min_clv = st.slider("Minimum CLV Score", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    with filter_col3:
        min_churn = st.slider("Minimum Churn Risk", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

    filtered_df = df[
        (df['customer_segment'].isin(selected_segments)) & 
        (df['clv_score'] >= min_clv) & 
        (df['churn_probability'] >= min_churn)
    ]
    
    display_cols = ['customerID', 'churn_probability', 'clv_score', 'customer_segment', 'recommended_action', 'priority_score']
    
    # Graceful handling if customerID is missing in raw uploaded data test
    display_cols = [c for c in display_cols if c in filtered_df.columns]
    
    if 'priority_score' in filtered_df.columns:
        filtered_df = filtered_df.sort_values(by='priority_score', ascending=False)
        
    st.divider()
    st.subheader("🔥 Top 10 Retention Targets")
    st.dataframe(filtered_df[display_cols].head(10), use_container_width=True, hide_index=True)

    st.write("<br>", unsafe_allow_html=True)
    st.subheader("📋 Full Filtered Customer List")
    st.dataframe(filtered_df[display_cols], use_container_width=True, hide_index=True)
    
    st.write("<br>", unsafe_allow_html=True)
    csv_data = filtered_df[display_cols].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download List as CSV for CRM",
        data=csv_data,
        file_name="prioritized_customer_targets.csv",
        mime="text/csv",
        help="Download this list to import directly into Salesforce, HubSpot, or your email marketing tool."
    )
else:
    if uploaded_file is None:
        st.warning("⚠️ No data loaded. Please upload a dataset in the sidebar or ensure the default CSV exists.")