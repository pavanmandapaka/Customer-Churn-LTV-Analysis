import streamlit as st
import pandas as pd
import plotly.express as px 

# -----------------------------------------------------------------------------
# 1 & 2. Page Config & Title
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn & LTV Analysis Dashboard", 
    page_icon="📊", 
    layout="wide"
)

st.title("Customer Churn & LTV Analysis Dashboard")

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
st.sidebar.header("Dashboard Navigation")
st.sidebar.info("Use this dashboard to monitor churn risk, Customer Lifetime Value (CLV), and prioritized business actions.")

# -----------------------------------------------------------------------------
# 4. Load Dataset
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Loads the final prioritized list of customers."""
    try:
        # Load the CSV generated in Phase 6 - Part 3
        df = pd.read_csv('final_customer_action_list.csv')
        return df
    except FileNotFoundError:
        st.error("⚠️ 'final_customer_action_list.csv' not found. Please run the final pipeline step first.")
        return pd.DataFrame()

df = load_data()

# -----------------------------------------------------------------------------
# 3. Overview Section
# -----------------------------------------------------------------------------
st.header("Overview")
st.write("""
This dashboard provides a unified view of customer health. It combines predictive churn 
probabilities with Customer Lifetime Value (CLV) scores to recommend prioritized, 
ROI-positive interventions for the marketing and retention teams.
""")

st.divider()

# -----------------------------------------------------------------------------
# 5 & 6. KPIs Section (Using Columns & Metrics)
# -----------------------------------------------------------------------------
st.header("KPIs")

if not df.empty:
    # Calculate metrics
    total_customers = len(df)
    avg_churn_prob = df['churn_probability'].mean()
    avg_clv_score = df['clv_score'].mean()
    
    # Count how many are flagged as specific retention targets
    retention_targets = len(df[df['customer_segment'] == 'Retention Target'])

    # Render metrics in 4 columns
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

# -----------------------------------------------------------------------------
# Visualizations Section
# -----------------------------------------------------------------------------
st.header("Visualizations")

if not df.empty:
    st.write("Explore customer distributions and the relationship between churn risk and lifetime value.")
    
    # ── ROW 1: Histograms ──
    col_hist1, col_hist2 = st.columns(2)
    
    with col_hist1:
        # Churn Probability Distribution
        fig_churn = px.histogram(
            df, x='churn_probability', nbins=30,
            title='Distribution of Churn Risk',
            labels={'churn_probability': 'Churn Probability (0 = Safe, 1 = High Risk)'},
            color_discrete_sequence=['#EF553B']
        )
        fig_churn.update_layout(bargap=0.1)
        st.plotly_chart(fig_churn, use_container_width=True)
        st.caption("**How to read this:** Shows the volume of customers at each risk level. A spike on the right indicates a large group of flight-risk customers.")

    with col_hist2:
        # CLV Score Distribution
        fig_clv = px.histogram(
            df, x='clv_score', nbins=30,
            title='Distribution of CLV Scores',
            labels={'clv_score': 'CLV Score (0 = Low Value, 1 = High Value)'},
            color_discrete_sequence=['#00CC96']
        )
        fig_clv.update_layout(bargap=0.1)
        st.plotly_chart(fig_clv, use_container_width=True)
        st.caption("**How to read this:** Shows the spread of lifetime value. Higher scores mean we have highly profitable, engaged customers.")
        
    st.write("<br>", unsafe_allow_html=True) # Spacer

    # ── ROW 2: Bar Chart & Scatter Plot ──
    col_bar, col_scatter = st.columns(2)
    
    with col_bar:
        # Customer Segment Breakdown
        segment_counts = df['customer_segment'].value_counts().reset_index()
        segment_counts.columns = ['Segment', 'Count']
        
        fig_segments = px.bar(
            segment_counts, x='Segment', y='Count',
            title='Customer Strategy Breakdown',
            color='Segment',
            text_auto=True
        )
        st.plotly_chart(fig_segments, use_container_width=True)
        st.caption("**How to read this:** Summarizes how many customers fall into our distinct intervention buckets (e.g., how many we need to actively try and save).")

    with col_scatter:
        # Risk vs Reward Matrix
        fig_scatter = px.scatter(
            df, x='churn_probability', y='clv_score',
            color='customer_segment',
            title='Risk vs. Reward (Intervention Matrix)',
            labels={
                'churn_probability': 'Churn Risk', 
                'clv_score': 'Value (CLV Score)'
            },
            opacity=0.7,
            hover_data=['recommended_action']
        )
        # Add visual quadrants (Optional enhancements for readability)
        fig_scatter.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
        fig_scatter.add_vline(x=0.5, line_dash="dash", line_color="gray", opacity=0.5)
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption("**How to read this:** Top Right = High Value + High Risk (Save them!). Bottom Right = Low Value + High Risk (Let them go organically).")

st.divider()

# -----------------------------------------------------------------------------
# Customer Insights & Retention Prioritization Section
# -----------------------------------------------------------------------------
st.header("Retention Prioritization View")
st.write("Filter and download prioritized customer lists for targeted marketing and retention campaigns.")

if not df.empty:
    st.subheader("1. Filter Customers")
    
    # Create layout columns for the filters
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        # Segment filter
        all_segments = df['customer_segment'].unique().tolist()
        selected_segments = st.multiselect(
            "Select Customer Segments", 
            options=all_segments, 
            default=all_segments
        )
        
    with filter_col2:
        # Minimum CLV filter
        min_clv = st.slider(
            "Minimum CLV Score", 
            min_value=0.0, max_value=1.0, value=0.0, step=0.05
        )
        
    with filter_col3:
        # Minimum Churn filter
        min_churn = st.slider(
            "Minimum Churn Risk", 
            min_value=0.0, max_value=1.0, value=0.0, step=0.05
        )

    # Apply the filters to the dataframe
    filtered_df = df[
        (df['customer_segment'].isin(selected_segments)) & 
        (df['clv_score'] >= min_clv) & 
        (df['churn_probability'] >= min_churn)
    ]
    
    # Define the core columns to display
    display_cols = [
        'customerID', 'churn_probability', 'clv_score', 
        'customer_segment', 'recommended_action', 'priority_score'
    ]
    
    # Ensure it's sorted by priority (Highest risk + value first)
    if 'priority_score' in filtered_df.columns:
        filtered_df = filtered_df.sort_values(by='priority_score', ascending=False)
        
    st.divider()
    
    # Display Top 10 Actionable Targets
    st.subheader("🔥 Top 10 Retention Targets")
    st.caption("The 10 single most important customers to contact right now based on your filters.")
    st.dataframe(
        filtered_df[display_cols].head(10),
        use_container_width=True,
        hide_index=True
    )

    st.write("<br>", unsafe_allow_html=True)

    # Display the full filtered list
    st.subheader("📋 Full Filtered Customer List")
    st.write(f"Showing **{len(filtered_df):,}** customers matching your criteria.")
    st.dataframe(
        filtered_df[display_cols],
        use_container_width=True,
        hide_index=True
    )
    
    st.write("<br>", unsafe_allow_html=True)
    
    # -----------------------------------------------------------------------------
    # CSV Download Button
    # -----------------------------------------------------------------------------
    # Convert filtered DataFrame to CSV format for download
    csv_data = filtered_df[display_cols].to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download List as CSV for CRM",
        data=csv_data,
        file_name="prioritized_customer_targets.csv",
        mime="text/csv",
        help="Download this list to import directly into Salesforce, HubSpot, or your email marketing tool."
    )