import pandas as pd
import numpy as np

# Mapping of dataset recommended actions to intervention strategy categories
ACTION_MAP = {
    'Offer Discount': 'Retention Discount',
    'Upsell / Reward': 'Loyalty Reward',
    'Standard Nurture': 'Standard Nurture',
    'No Action': 'No Action'
}

def get_default_configs():
    """
    Returns default business configurations for costs and retention success rates.
    These can be overridden dynamically by dashboard user inputs.
    """
    return {
        'Retention Discount': {'cost': 150.0, 'success_rate': 0.15},
        'Loyalty Reward': {'cost': 100.0, 'success_rate': 0.10},
        'Standard Nurture': {'cost': 10.0, 'success_rate': 0.05},
        'No Action': {'cost': 0.0, 'success_rate': 0.00}
    }

def process_business_metrics(df, clv_multiplier=7000.0, configs=None):
    """
    Calculates customer-level Revenue at Risk, Intervention Cost, Expected Saved Value, and ROI.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input customer dataframe with 'churn_probability' and 'clv_score'.
    clv_multiplier : float
        Dollar value multiplier to scale normalized CLV scores (0-1) to estimated dollars.
    configs : dict, optional
        Dictionary containing custom cost and success rate configurations for interventions.
        If None, DEFAULT_CONFIGS is used.
        
    Returns:
    --------
    pd.DataFrame
        Dataframe containing computed business metrics.
    """
    if configs is None:
        configs = get_default_configs()
        
    res = df.copy()
    
    # 1. Define CLV in dollars and Revenue at Risk
    # clv_dollars = clv_score * clv_multiplier
    # revenue_at_risk = churn_probability * clv_score (Index, normalized)
    # revenue_at_risk_dollars = churn_probability * clv_dollars (Monetary value)
    res['clv_dollars'] = res['clv_score'] * clv_multiplier
    res['revenue_at_risk'] = res['churn_probability'] * res['clv_score']
    res['revenue_at_risk_dollars'] = res['churn_probability'] * res['clv_dollars']
    
    # 2. Map Recommended Action to Intervention and assign configurable costs/success rates
    def map_intervention_cost(action):
        mapped_action = ACTION_MAP.get(action, 'No Action')
        return configs.get(mapped_action, configs['No Action'])['cost']
        
    def map_intervention_success(action):
        mapped_action = ACTION_MAP.get(action, 'No Action')
        return configs.get(mapped_action, configs['No Action'])['success_rate']
        
    recommended_action_col = 'recommended_action' if 'recommended_action' in res.columns else 'recommended_action'
    if recommended_action_col in res.columns:
        res['intervention_cost'] = res[recommended_action_col].apply(map_intervention_cost)
        res['retention_success_rate'] = res[recommended_action_col].apply(map_intervention_success)
    else:
        # Fallback if action column is not present
        res['intervention_cost'] = 0.0
        res['retention_success_rate'] = 0.0
        
    # 3. Calculate Expected Value Saved
    # expected_saved_value = revenue_at_risk_dollars * retention_success_rate
    res['expected_saved_value'] = res['revenue_at_risk_dollars'] * res['retention_success_rate']
    
    # 4. Calculate Customer-level ROI
    # ROI = (expected_saved_value - intervention_cost) / intervention_cost
    # If intervention_cost is 0, ROI is 0
    res['roi'] = np.where(
        res['intervention_cost'] > 0,
        (res['expected_saved_value'] - res['intervention_cost']) / res['intervention_cost'],
        0.0
    )
    
    # 5. Categorize customers into ROI Categories
    def categorize_roi(roi_val, cost):
        if cost == 0:
            return 'No Action / Neutral'
        elif roi_val > 1.0:
            return 'High ROI (>100%)'
        elif roi_val >= 0.0:
            return 'Medium ROI (0-100%)'
        else:
            return 'Negative ROI (<0%)'
            
    res['roi_category'] = res.apply(lambda r: categorize_roi(r['roi'], r['intervention_cost']), axis=1)
    
    # Rank customers by revenue_at_risk descending
    res = res.sort_values(by='revenue_at_risk', ascending=False)
    
    return res

def run_campaign_simulation(df, budget=10000.0):
    """
    Simulates a retention campaign under budget constraints.
    Sorts customers by customer-level ROI descending, then by revenue_at_risk_dollars descending,
    and selects customers for the campaign until cumulative intervention cost reaches the budget limit.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with computed business metrics (output of process_business_metrics).
    budget : float
        Total marketing budget for retention.
        
    Returns:
    --------
    pd.DataFrame, dict
        Updated DataFrame with 'selected_for_campaign' and 'priority_rank' columns,
        and a summary dictionary of campaign outcomes.
    """
    res = df.copy()
    
    # We only want to target customers with recommended actions that have costs and positive ROI.
    # Sorting by ROI descending, and then by revenue_at_risk descending for ties
    res = res.sort_values(by=['roi', 'revenue_at_risk'], ascending=[False, False]).reset_index(drop=True)
    
    # Assign priority rank (1-indexed based on sorting)
    res['priority_rank'] = res.index + 1
    
    selected = []
    cumulative_cost = 0.0
    
    for idx, row in res.iterrows():
        cost = row['intervention_cost']
        roi = row['roi']
        
        # Select customer if:
        # 1. Cost is positive (they need an intervention)
        # 2. ROI is positive (otherwise we lose money by saving them!)
        # 3. We have budget left to cover the cost
        if cost > 0 and roi > 0 and (cumulative_cost + cost) <= budget:
            selected.append(True)
            cumulative_cost += cost
        else:
            selected.append(False)
            
    res['selected_for_campaign'] = selected
    
    # Calculate summary metrics
    campaign_df = res[res['selected_for_campaign'] == True]
    total_targeted = len(campaign_df)
    budget_spent = campaign_df['intervention_cost'].sum()
    budget_remaining = budget - budget_spent
    expected_value_saved = campaign_df['expected_saved_value'].sum()
    net_value_saved = expected_value_saved - budget_spent
    total_roi = (net_value_saved / budget_spent) * 100.0 if budget_spent > 0 else 0.0
    
    summary = {
        'total_customers': len(df),
        'high_priority_targets': len(df[df['customer_segment'] == 'Retention Target']),
        'campaign_budget': budget,
        'total_targeted': total_targeted,
        'budget_spent': budget_spent,
        'budget_remaining': budget_remaining,
        'expected_value_saved': expected_value_saved,
        'net_value_saved': net_value_saved,
        'total_campaign_roi': total_roi
    }
    
    return res, summary
