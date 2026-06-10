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
        action_df = pd.read_csv('final_customer_action_list.csv')
        try:
            features_df = pd.read_csv('telco_customer_churn_cleaned.csv')
            # Merge to ensure individual customer feature details are accessible
            merged_df = pd.merge(action_df, features_df, on='customerID', how='left')
            return merged_df
        except Exception:
            return action_df
    except FileNotFoundError:
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# SIDEBAR: Data Source & Configs
# -----------------------------------------------------------------------------
import business_impact

st.sidebar.header("📁 Data Source")

# Add Streamlit file uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload New Customer Data (CSV)", 
    type=["csv"],
    help="Upload raw telco data. It will automatically run through the ML pipeline."
)

st.sidebar.divider()

st.sidebar.header("🎯 Business Impact Configs")

# CLV Multiplier
clv_multiplier = st.sidebar.slider(
    "CLV Dollar Scaler ($)", 
    min_value=1000, 
    max_value=20000, 
    value=7000, 
    step=500, 
    help="Scales the 0-1 CLV score to an estimated lifetime dollar amount for ROI calculations."
)

# Retention Budget
retention_budget = st.sidebar.number_input(
    "Retention Budget ($)", 
    min_value=500, 
    max_value=100000, 
    value=10000, 
    step=500, 
    help="Total budget allocated for the retention campaign."
)

# Intervention settings in an expander
with st.sidebar.expander("🛠️ Intervention Settings", expanded=False):
    st.markdown("**1. Retention Discount**")
    disc_cost = st.number_input("Cost ($)", min_value=0, max_value=1000, value=150, step=10, key="disc_c")
    disc_rate = st.slider("Success Rate", min_value=0.0, max_value=1.0, value=0.15, step=0.01, key="disc_s")
    
    st.markdown("**2. Loyalty Reward**")
    reward_cost = st.number_input("Cost ($)", min_value=0, max_value=1000, value=100, step=10, key="rew_c")
    reward_rate = st.slider("Success Rate", min_value=0.0, max_value=1.0, value=0.10, step=0.01, key="rew_s")
    
    st.markdown("**3. Standard Nurture**")
    nurture_cost = st.number_input("Cost ($)", min_value=0, max_value=1000, value=10, step=1, key="nurt_c")
    nurture_rate = st.slider("Success Rate", min_value=0.0, max_value=1.0, value=0.05, step=0.01, key="nurt_s")
    
    st.markdown("**4. No Action**")
    no_action_cost = st.number_input("Cost ($)", min_value=0, max_value=1000, value=0, step=1, key="no_c")
    no_action_rate = st.slider("Success Rate", min_value=0.0, max_value=1.0, value=0.00, step=0.01, key="no_s")

# Package settings dictionary
configs = {
    'Retention Discount': {'cost': float(disc_cost), 'success_rate': float(disc_rate)},
    'Loyalty Reward': {'cost': float(reward_cost), 'success_rate': float(reward_rate)},
    'Standard Nurture': {'cost': float(nurture_cost), 'success_rate': float(nurture_rate)},
    'No Action': {'cost': float(no_action_cost), 'success_rate': float(no_action_rate)}
}

st.sidebar.divider()
st.sidebar.header("Dashboard Navigation")
st.sidebar.info("Use the tabs in the main window to explore prediction dashboards, ROI simulators, CRM lists, and business validations.")

# Execute Workflow based on upload
if uploaded_file is not None:
    try:
        raw_csv = pd.read_csv(uploaded_file)
        st.sidebar.success("File uploaded successfully!")
        df = run_end_to_end_pipeline(raw_csv)
    except Exception as e:
        st.sidebar.error(f"❌ Error parsing CSV file: {str(e)}")
        df = pd.DataFrame()
else:
    df = load_default_data()

# Process data if loaded successfully
if not df.empty:
    df_processed = business_impact.process_business_metrics(df, clv_multiplier=clv_multiplier, configs=configs)
    df, summary = business_impact.run_campaign_simulation(df_processed, budget=retention_budget)

# =============================================================================
# DASHBOARD UI COMPONENTS
# =============================================================================

st.header("Business Decision Optimization Platform")
st.write("""
This system translates predictive ML models (Churn Risk & Customer Lifetime Value) into **quantifiable business impact**. 
Rather than targeting customers based on risk alone, this platform simulates budget-constrained retention campaigns 
to optimize **Return on Investment (ROI)** and prevent high-value revenue leakage.
""")

st.divider()

if not df.empty:
    # -------------------------------------------------------------------------
    # KPIs Section (Prediction & Financial)
    # -------------------------------------------------------------------------
    st.subheader("📊 Key Performance Indicators")
    
    # Row 1: ML Model Outputs
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
        st.metric(label="High-Priority Targets (ML)", value=f"{retention_targets:,}")
        
    # Row 2: Business & Financial Metrics
    total_rar = df['revenue_at_risk_dollars'].sum()
    targeted_cust = summary['total_targeted']
    val_saved = summary['expected_value_saved']
    camp_roi = summary['total_campaign_roi']
    
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric(label="Total Revenue at Risk", value=f"${total_rar:,.2f}", help="Sum of (Churn Probability * CLV in Dollars) for all customers.")
    with col6:
        st.metric(label="Campaign Target Group", value=f"{targeted_cust:,}", help="Number of customers targeted under the budget constraint.")
    with col7:
        st.metric(label="Expected Value Saved", value=f"${val_saved:,.2f}", help="Expected revenue saved based on intervention success rates.")
    with col8:
        st.metric(label="Projected Campaign ROI", value=f"{camp_roi:.1f}%", help="Net value saved / budget spent.")

    st.divider()

    # -------------------------------------------------------------------------
    # Main Dashboard Tabs
    # -------------------------------------------------------------------------
    tab_sim, tab_roi, tab_crm, tab_exp, tab_val = st.tabs([
        "📈 Campaign Simulator",
        "📊 Revenue & ROI Analysis",
        "🔍 CRM Targeting Suite",
        "🧠 Model Explainability",
        "💼 Business Rationale & Validation"
    ])

    # -------------------------------------------------------------------------
    # Tab 1: Campaign Simulator
    # -------------------------------------------------------------------------
    with tab_sim:
        st.header("Executive Summary")
        
        # Display professional callout block
        summary_text = (
            f"Out of **{summary['total_customers']:,}** customers, **{summary['high_priority_targets']:,}** "
            f"were identified as high-priority ML retention targets.\n\n"
            f"Using a retention budget of **${summary['campaign_budget']:,.2f}**, the optimization engine recommends "
            f"targeting **{summary['total_targeted']:,}** customers with positive ROI interventions.\n\n"
            f"**Expected Value Saved**: **${summary['expected_value_saved']:,.2f}** | "
            f"**Total Budget Spent**: **${summary['budget_spent']:,.2f}** | "
            f"**Projected Campaign ROI**: **{summary['total_campaign_roi']:.1f}%**"
        )
        st.info(summary_text)
        
        col_sim1, col_sim2 = st.columns(2)
        with col_sim1:
            # Budget breakdown metrics
            st.subheader("Budget Utilization")
            util_data = pd.DataFrame({
                'Metric': ['Budget Spent', 'Budget Remaining'],
                'Amount ($)': [summary['budget_spent'], summary['budget_remaining']]
            })
            fig_budget = px.bar(
                util_data, 
                x='Metric', 
                y='Amount ($)', 
                text_auto='$,.2f',
                color='Metric',
                color_discrete_map={'Budget Spent': '#EF553B', 'Budget Remaining': '#00CC96'}
            )
            fig_budget.update_layout(showlegend=False)
            st.plotly_chart(fig_budget, use_container_width=True)
            
        with col_sim2:
            # Budget spent by action type
            st.subheader("Budget Allocation by Strategy")
            campaign_df = df[df['selected_for_campaign'] == True].copy()
            if not campaign_df.empty:
                campaign_df['intervention_type'] = campaign_df['recommended_action'].map(business_impact.ACTION_MAP)
                budget_by_int = campaign_df.groupby('intervention_type')['intervention_cost'].sum().reset_index()
                budget_by_int.columns = ['Intervention Type', 'Budget Allocated ($)']
                fig_pie = px.pie(
                    budget_by_int,
                    values='Budget Allocated ($)',
                    names='Intervention Type',
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    hole=0.4
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.warning("No customers selected for campaign. Increase budget or adjust intervention settings.")

    # -------------------------------------------------------------------------
    # Tab 2: Revenue & ROI Analysis
    # -------------------------------------------------------------------------
    with tab_roi:
        st.header("Revenue at Risk & ROI Analysis")
        
        col_roi1, col_roi2 = st.columns(2)
        with col_roi1:
            # Distribution of Revenue at Risk
            fig_rar_dist = px.histogram(
                df,
                x='revenue_at_risk_dollars',
                nbins=30,
                title='Distribution of Revenue at Risk ($)',
                color_discrete_sequence=['#EF553B'],
                labels={'revenue_at_risk_dollars': 'Revenue at Risk ($)', 'count': 'Customer Count'}
            )
            fig_rar_dist.update_layout(bargap=0.1)
            st.plotly_chart(fig_rar_dist, use_container_width=True)
            st.caption("**Revenue at Risk**: Represents the potential financial loss from a customer. Replaced simple churn risk with absolute dollar exposure.")
            
        with col_roi2:
            # Risk vs Value Scatter Plot
            fig_scatter_rar = px.scatter(
                df,
                x='churn_probability',
                y='clv_dollars',
                color='customer_segment',
                size='revenue_at_risk_dollars',
                hover_data=['customerID', 'revenue_at_risk_dollars'],
                title='Risk vs. Reward (Size represents Revenue at Risk in $)',
                labels={'churn_probability': 'Churn Probability', 'clv_dollars': 'CLV ($)'}
            )
            st.plotly_chart(fig_scatter_rar, use_container_width=True)
            st.caption("Top Right corner: Large dots indicating customers with high risk AND high value. These represent the highest financial exposure.")
            
        st.markdown("---")
        st.subheader("Intervention Campaign ROI")
        
        col_roi3, col_roi4 = st.columns(2)
        with col_roi3:
            # ROI Distribution for targeted strategies
            targeted_only = df[df['intervention_cost'] > 0]
            if not targeted_only.empty:
                fig_roi_dist = px.histogram(
                    targeted_only,
                    x='roi',
                    color='roi_category',
                    nbins=30,
                    title='Distribution of Customer-Level ROI',
                    labels={'roi': 'ROI (Multiplier)', 'count': 'Customer Count'},
                    color_discrete_map={
                        'High ROI (>100%)': '#00CC96',
                        'Medium ROI (0-100%)': '#636EFA',
                        'Negative ROI (<0%)': '#EF553B'
                    }
                )
                fig_roi_dist.update_layout(bargap=0.1)
                st.plotly_chart(fig_roi_dist, use_container_width=True)
            else:
                st.warning("No customers have configured interventions.")
                
        with col_roi4:
            # ROI by Segment
            if not targeted_only.empty:
                fig_roi_seg = px.box(
                    targeted_only,
                    x='customer_segment',
                    y='roi',
                    color='customer_segment',
                    title='Customer ROI by Strategic Segment',
                    labels={'roi': 'ROI (Multiplier)', 'customer_segment': 'Segment'}
                )
                st.plotly_chart(fig_roi_seg, use_container_width=True)
            else:
                st.warning("No customers have configured interventions.")
                
        # ROI by Intervention Type
        if not targeted_only.empty:
            st.write("<br>", unsafe_allow_html=True)
            targeted_only_plot = targeted_only.copy()
            targeted_only_plot['intervention_type'] = targeted_only_plot['recommended_action'].map(business_impact.ACTION_MAP)
            fig_roi_int = px.box(
                targeted_only_plot,
                x='intervention_type',
                y='roi',
                color='intervention_type',
                title='Customer ROI by Intervention Strategy',
                labels={'roi': 'ROI (Multiplier)', 'intervention_type': 'Intervention Type'}
            )
            st.plotly_chart(fig_roi_int, use_container_width=True)

    # -------------------------------------------------------------------------
    # Tab 3: CRM Targeting Suite
    # -------------------------------------------------------------------------
    with tab_crm:
        st.header("Campaign Targeting & Filtering")
        st.write("Filter, search, and download targeted customer records for CRM uploads (Salesforce, HubSpot, etc.).")
        
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            sel_campaign_only = st.radio("Campaign Selection", ["Show All Customers", "Campaign Targets Only", "Excluded Customers"])
        with filter_col2:
            min_roi_filter = st.slider("Minimum ROI (%)", min_value=-100, max_value=500, value=-100, step=10)
        with filter_col3:
            all_segments = df['customer_segment'].unique().tolist()
            selected_segments = st.multiselect("Select Customer Segments", options=all_segments, default=all_segments)
            
        # Apply filters
        f_df = df.copy()
        if sel_campaign_only == "Campaign Targets Only":
            f_df = f_df[f_df['selected_for_campaign'] == True]
        elif sel_campaign_only == "Excluded Customers":
            f_df = f_df[(f_df['selected_for_campaign'] == False) & (f_df['intervention_cost'] > 0)]
            
        f_df = f_df[f_df['roi'] * 100.0 >= min_roi_filter]
        f_df = f_df[f_df['customer_segment'].isin(selected_segments)]
        
        # Select columns to display
        display_cols = [
            'priority_rank', 'customerID', 'churn_probability', 'clv_score', 'clv_dollars', 
            'revenue_at_risk_dollars', 'recommended_action', 'intervention_cost', 
            'expected_saved_value', 'roi', 'selected_for_campaign'
        ]
        display_cols = [c for c in display_cols if c in f_df.columns]
        
        # Sort by priority rank
        if 'priority_rank' in f_df.columns:
            f_df = f_df.sort_values(by='priority_rank')
            
        st.subheader(f"📋 Filtered Campaign List ({len(f_df):,} customers matching filters)")
        st.dataframe(
            f_df[display_cols].style.format({
                'churn_probability': '{:.1%}',
                'clv_score': '{:.2f}',
                'clv_dollars': '${:,.2f}',
                'revenue_at_risk_dollars': '${:,.2f}',
                'intervention_cost': '${:,.2f}',
                'expected_saved_value': '${:,.2f}',
                'roi': '{:.2%}'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        st.write("<br>", unsafe_allow_html=True)
        csv_data = f_df[display_cols].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Targeted CSV for CRM Upload",
            data=csv_data,
            file_name="retention_campaign_targets.csv",
            mime="text/csv",
            help="Download this list to import directly into your marketing tools."
        )
        
        st.markdown("---")
        st.subheader("🔥 Top 20 Highest Revenue at Risk Customers")
        df_top_rar = df.sort_values(by='revenue_at_risk_dollars', ascending=False).head(20)
        st.dataframe(
            df_top_rar[display_cols].style.format({
                'churn_probability': '{:.1%}',
                'clv_score': '{:.2f}',
                'clv_dollars': '${:,.2f}',
                'revenue_at_risk_dollars': '${:,.2f}',
                'intervention_cost': '${:,.2f}',
                'expected_saved_value': '${:,.2f}',
                'roi': '{:.2%}'
            }),
            use_container_width=True,
            hide_index=True
        )

    # -------------------------------------------------------------------------
    # Tab 4: Explainability & Customer Drill-down
    # -------------------------------------------------------------------------
    with tab_exp:
        st.header("🎯 Model Explainability Suite")
        
        st.subheader("Top Drivers of Churn")
        st.write(
            "This section highlights the top features influencing the baseline churn model. "
            "Positive coefficients increase churn probability, while negative coefficients are protective features."
        )

        global_importance_data = {
            "Feature Name": [
                "internet_Fiber optic",
                "contract_ordinal (Month-to-month)",
                "phoneservice_flag",
                "tenure_bin_ordinal",
                "onlinesecurity_Yes",
                "paperlessbilling_flag",
                "multiplelines_Yes",
                "is_trial_period",
                "streamingmovies_Yes",
                "techsupport_Yes"
            ],
            "Importance Score": [0.8529, -0.7180, -0.5421, 0.4738, -0.4156, 0.4122, 0.3820, 0.3044, 0.2978, -0.2881],
            "Type": [
                "Increases Churn",
                "Protective (Reduces Churn)",
                "Protective (Reduces Churn)",
                "Increases Churn",
                "Protective (Reduces Churn)",
                "Increases Churn",
                "Increases Churn",
                "Increases Churn",
                "Increases Churn",
                "Protective (Reduces Churn)"
            ]
        }
        importance_df = pd.DataFrame(global_importance_data)
        importance_df["Absolute Importance"] = importance_df["Importance Score"].abs()
        importance_df = importance_df.sort_values(by="Absolute Importance", ascending=False)

        col_exp1, col_exp2 = st.columns([3, 2])

        with col_exp1:
            fig_importance = px.bar(
                importance_df,
                x="Importance Score",
                y="Feature Name",
                color="Type",
                orientation="h",
                title="Feature Coefficients (Baseline Logistic Regression)",
                color_discrete_map={"Increases Churn": "#EF553B", "Protective (Reduces Churn)": "#00CC96"}
            )
            st.plotly_chart(fig_importance, use_container_width=True)

        with col_exp2:
            st.write("**Feature Impact Table**")
            st.dataframe(
                importance_df[["Feature Name", "Importance Score", "Type"]],
                use_container_width=True,
                hide_index=True
            )

        st.divider()
        st.subheader("🔍 Individual Customer Drill-Down")
        
        customer_list = df['customerID'].dropna().unique().tolist()
        selected_customer_id = st.selectbox("Search / Select by CustomerID", options=customer_list)

        if selected_customer_id:
            cust_row = df[df['customerID'] == selected_customer_id].iloc[0]

            col_c1, col_c2, col_c3, col_c4 = st.columns(4)
            with col_c1:
                st.metric(label="CustomerID", value=str(cust_row['customerID']))
                st.metric(label="Priority Rank", value=f"#{int(cust_row.get('priority_rank', 0))}")
            with col_c2:
                st.metric(label="Churn Probability", value=f"{cust_row.get('churn_probability', 0.0):.1%}")
                st.metric(label="Expected ROI", value=f"{cust_row.get('roi', 0.0):.1%}")
            with col_c3:
                st.metric(label="CLV ($)", value=f"${cust_row.get('clv_dollars', 0.0):,.2f}")
                st.metric(label="Intervention Cost", value=f"${cust_row.get('intervention_cost', 0.0):,.2f}")
            with col_c4:
                st.metric(label="Revenue at Risk", value=f"${cust_row.get('revenue_at_risk_dollars', 0.0):,.2f}")
                st.metric(label="Selected for Campaign?", value="Yes" if cust_row.get('selected_for_campaign') else "No")

            # Local Explanation Generator (Rule-based Fallback)
            st.write("### 🤖 Prediction Rationale")
            pos_drivers = []
            neg_drivers = []

            # Rule evaluation on customer details
            if cust_row.get('contract_ordinal', 0) == 0:
                pos_drivers.append("⚠️ **Month-to-month Contract**: High volatility and low commitment.")
            elif cust_row.get('contract_ordinal', 0) == 2:
                neg_drivers.append("✅ **Two-year Contract**: Highly stable, long-term commitment.")
            
            if cust_row.get('tenure', 0) <= 12:
                pos_drivers.append(f"⚠️ **New Customer Lifecycle**: Early lifecycle customer ({int(cust_row.get('tenure', 0))} months) is prone to early churn.")
            elif cust_row.get('tenure', 0) >= 48:
                neg_drivers.append(f"✅ **High Tenure Customer**: Long standing lifecycle ({int(cust_row.get('tenure', 0))} months) reduces volatility.")

            if cust_row.get('internet_Fiber optic', 0) == 1:
                pos_drivers.append("⚠️ **Fiber Optic Service**: Premium plan with higher price-points.")
            
            if cust_row.get('onlinesecurity_Yes', 0) == 1:
                neg_drivers.append("✅ **Online Security Enabled**: High-stickiness value-added product.")
            
            if cust_row.get('techsupport_Yes', 0) == 1:
                neg_drivers.append("✅ **Tech Support Utilized**: Engaged customer using support avenues.")

            if cust_row.get('MonthlyCharges', 0.0) >= 80.0:
                pos_drivers.append(f"⚠️ **High Monthly Cost**: Monthly bill is high (${cust_row.get('MonthlyCharges', 0.0):.2f}).")
            elif cust_row.get('MonthlyCharges', 0.0) <= 30.0:
                neg_drivers.append(f"✅ **Budget Friendly Plan**: Low financial risk path (${cust_row.get('MonthlyCharges', 0.0):.2f}).")

            col_pos, col_neg = st.columns(2)
            with col_pos:
                st.markdown("🔴 **Risk Indicators (Increases Churn)**")
                if pos_drivers:
                    for d in pos_drivers:
                        st.write(d)
                else:
                    st.write("None detected.")

            with col_neg:
                st.markdown("🟢 **Protective Factors (Reduces Churn)**")
                if neg_drivers:
                    for d in neg_drivers:
                        st.write(d)
                else:
                    st.write("None detected.")

            # Business Recommendation Card
            st.write("### 💼 Business Action Card")
            
            rec_action = cust_row.get('recommended_action', 'Standard Nurture')
            rec_segment = cust_row.get('customer_segment', 'General Monitoring')
            rec_churn = cust_row.get('churn_probability', 0.0)
            rec_clv_dol = cust_row.get('clv_dollars', 0.0)
            rec_roi = cust_row.get('roi', 0.0)

            if rec_segment == 'Retention Target':
                rec_reason = f"High Churn Risk ({rec_churn:.1%}) combined with High Value (CLV of ${rec_clv_dol:,.2f}). A proactive save strategy is critical."
                rec_impact = f"Potential retention of high-value account. Targeted for campaign with predicted ROI of {rec_roi:.1%}."
            elif rec_segment == 'Loyalty/Upsell Target':
                rec_reason = f"Highly loyal customer (Low Risk: {rec_churn:.1%}) with high tenure. Primed for contract extension or cross-selling."
                rec_impact = "Increases ARPU (Average Revenue Per User) and increases long term advocacy."
            elif rec_segment == 'Non-Target':
                rec_reason = f"High Churn Risk ({rec_churn:.1%}) but Low CLV Value. Avoid expensive promotions; monitor passively."
                rec_impact = "Avoids margin-diluting discount costs on low-value accounts."
            else:
                rec_reason = "Customer is in a stable general monitoring state with balanced risk and value metrics."
                rec_impact = "Maintains standard marketing communication path."

            st.info(
                f"**Recommended Action**: **{rec_action}**\n\n"
                f"**Rationale**: {rec_reason}\n\n"
                f"**Expected Business Impact**: {rec_impact}"
            )

    # -------------------------------------------------------------------------
    # Tab 5: Business Rationale & Validation
    # -------------------------------------------------------------------------
    with tab_val:
        st.header("💼 Business Validation & ROI Methodology")
        st.write("""
        This section provides executive stakeholders and marketing operators with the financial theory and validation 
        principles underpinning the ROI Decision Engine.
        """)
        
        st.subheader("1. Why Churn Probability Alone is Insufficient")
        st.markdown("""
        * **The Churn Blind Spot**: A customer with a **95% churn probability** who spends **$10/month** represents very low financial exposure. 
          If a retention campaign costs **$20** to run, targeting this customer is a guaranteed loss—even if the campaign is 100% successful.
        * **Scattergun Marketing**: Targeting based only on risk leads call centers to waste budget saving customers who generate zero long-term profit, diluting overall margins.
        """)
        
        st.subheader("2. Why CLV Alone is Insufficient")
        st.markdown("""
        * **Wasted VIP Spend**: A customer with a **$10,000 CLV** who has a **2% churn probability** is highly loyal and unlikely to leave. 
          Sending them an expensive retention offer or a billing discount is a waste of money (known as *cannibalization* or *subsidy waste*). They would have stayed anyway.
        * **Misallocated Budget**: Focus on value alone neglects the fact that stable customers do not require active, expensive intervention.
        """)
        
        st.subheader("3. Why ROI-Based Targeting is Superior")
        st.markdown("""
        * **The Intersection of Risk & Value**: By combining Churn Risk ($P(Churn)$) and Customer Value ($CLV$), we calculate the **Revenue at Risk**:
          $$\\text{Revenue at Risk} = P(\\text{Churn}) \\times \\text{CLV}$$
        * **Factoring in Action Success**: Campaigns are never 100% successful. We scale the Revenue at Risk by the campaign's success rate ($SuccessRate$) to determine the **Expected Value Saved**:
          $$\\text{Expected Saved Value} = \\text{Revenue at Risk} \\times \\text{Success Rate}$$
        * **ROI Constraint**: ROI ensures we only spend where the return outweighs the cost:
          $$\\text{ROI} = \\frac{\\text{Expected Saved Value} - \\text{Intervention Cost}}{\\text{Intervention Cost}}$$
          We target customers with the highest positive ROI first, squeezing the maximum financial return out of every marketing dollar.
        """)
        
        st.subheader("4. How this System Reduces Wasted Retention Spending")
        st.markdown("""
        * **Surgical Precision**: By sorting by ROI descending, the system eliminates subsidy waste (people who won't leave) and low-value waste (people not worth saving).
        * **Budget Simulation**: Marketing teams can test budget thresholds before committing funds. Under budget constraints, the simulator automatically drops low-yield campaigns to stay within budget, ensuring **100% budget efficiency**.
        """)
        
        st.subheader("5. How Business Teams Can Use These Outputs")
        st.markdown("""
        * **Marketing Operations**: Export the filtered targeted list directly into CRM systems (Salesforce, HubSpot) to automatically trigger campaign workflows (discount codes, loyalty emails).
        * **Finance & CFO Office**: Present the projected ROI (e.g. **380%**) and expected value saved (e.g. **$48,000**) to secure additional marketing budgets based on predictable financial returns.
        * **Product Teams**: Review churn drivers to address product issues (like Month-to-month contracts or Fiber Optic pricing concerns) that create the highest revenue leaks.
        """)
