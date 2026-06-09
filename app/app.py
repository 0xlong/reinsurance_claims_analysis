import os
import pickle
import duckdb
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go

def get_clean_feature_label(prep_feature_name, prep_value, raw_row):
    """Generate a clean user-facing label like '35 = Driver Age' or '1 = Area Code (C)'."""
    # Capitalize and clean up name
    def clean_name(n):
        mapping = {
            'area_code': 'Area Code',
            'vehicle_power': 'Vehicle Power',
            'vehicle_age': 'Vehicle Age',
            'driver_age': 'Driver Age',
            'bonus_malus': 'Bonus/Malus',
            'vehicle_brand': 'Vehicle Brand',
            'fuel_type': 'Fuel Type',
            'population_density': 'Population Density',
            'region_code': 'Region Code',
            'region_population': 'Region Population'
        }
        return mapping.get(n, n.replace('_', ' ').title())

    if prep_feature_name.startswith("num__"):
        orig_name = prep_feature_name.replace("num__", "")
        if orig_name in raw_row.columns:
            val = raw_row.loc[0, orig_name]
            if isinstance(val, (int, np.integer)):
                val_str = f"{val}"
            elif isinstance(val, (float, np.floating)):
                val_str = f"{val:,.2f}".rstrip('0').rstrip('.')
            else:
                val_str = str(val)
        else:
            val_str = f"{prep_value:.2f}"
        return f"{val_str} = {clean_name(orig_name)}"
        
    elif prep_feature_name.startswith("cat__"):
        # e.g. cat__area_code_B
        for cat_feat in ['area_code', 'vehicle_brand', 'fuel_type', 'region_code']:
            prefix = f"cat__{cat_feat}_"
            if prep_feature_name.startswith(prefix):
                category = prep_feature_name[len(prefix):]
                val_str = str(int(prep_value))
                return f"{val_str} = {clean_name(cat_feat)} ({category})"
                
    # Fallback
    val_str = f"{prep_value:.2f}" if isinstance(prep_value, (float, np.floating)) else str(prep_value)
    return f"{val_str} = {prep_feature_name}"


def draw_plotly_bar(shap_values, feature_names, preprocessed_row, raw_row, max_display=10):
    """Draw a directional horizontal SHAP bar plot similar to shap.plots.bar()."""
    # Create DataFrame of SHAP values and feature values
    df_shap = pd.DataFrame({
        'feature': feature_names,
        'shap_value': shap_values,
    })
    
    # Calculate absolute SHAP value for sorting
    df_shap['abs_shap'] = df_shap['shap_value'].abs()
    
    # Sort all features by impact
    df_shap = df_shap.sort_values(by='abs_shap', ascending=False).reset_index(drop=True)
    
    # Group remaining features if we have more than max_display
    if len(df_shap) > max_display:
        top_df = df_shap.head(max_display - 1).copy()
        other_df = df_shap.iloc[max_display - 1:]
        sum_other_shap = other_df['shap_value'].sum()
        
        # Create the summary row
        other_row = pd.DataFrame([{
            'feature': f'Sum of {len(other_df)} other features',
            'shap_value': sum_other_shap,
            'abs_shap': abs(sum_other_shap)
        }])
        
        # Combine
        df_plot = pd.concat([top_df, other_row], ignore_index=True)
    else:
        df_plot = df_shap.copy()
        
    # Reverse the order so the largest impact is at the top of the horizontal bar chart
    df_plot = df_plot.iloc[::-1].reset_index(drop=True)
    
    # Create labels and annotations
    y_labels = []
    text_labels = []
    text_colors = []
    bar_colors = []
    
    # Define color palette matching default SHAP
    pos_color = '#ff0051' # SHAP pink/red
    neg_color = '#008bfb' # SHAP blue
    
    for _, row in df_plot.iterrows():
        feat_name = row['feature']
        s_val = row['shap_value']
        
        # Color mapping based on positive or negative SHAP value
        if s_val >= 0:
            bar_colors.append(pos_color)
            text_colors.append(pos_color)
            text_labels.append(f"+{s_val:.2f}")
        else:
            bar_colors.append(neg_color)
            text_colors.append(neg_color)
            text_labels.append(f"{s_val:.2f}")
            
        # Label formatting: check if it's the sum or a regular feature
        if feat_name.startswith("Sum of "):
            y_labels.append(feat_name)
        else:
            # Get original display label using the helper function
            y_labels.append(get_clean_feature_label(
                feat_name, 
                preprocessed_row[feat_name].values[0] if feat_name in preprocessed_row.columns else 0.0, 
                raw_row
            ))
            
    fig = go.Figure(go.Bar(
        x=df_plot['shap_value'],
        y=y_labels,
        orientation='h',
        marker=dict(
            color=bar_colors,
            line=dict(width=0)
        ),
        text=text_labels,
        textposition='outside',
        textfont=dict(
            family="Inter, sans-serif",
            size=12,
            color='white'
        ),
        cliponaxis=False # Ensure text labels aren't clipped off
    ))
    
    # Calculate padding for x-axis range to prevent label clipping
    min_x = min(df_plot['shap_value'])
    max_x = max(df_plot['shap_value'])
    span = max_x - min_x if max_x != min_x else 1.0
    padding = span * 0.15
    
    # Styling layout
    fig.update_layout(
        margin=dict(l=10, r=40, t=10, b=10),
        height=30 + 38 * len(df_plot), # Dynamic height based on number of features
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title="SHAP value",
            titlefont=dict(family="Inter, sans-serif", size=13, color='white'),
            tickfont=dict(family="Inter, sans-serif", size=11, color='white'),
            showgrid=False,
            zeroline=True,
            zerolinecolor='rgba(128, 128, 128, 0.8)',
            zerolinewidth=1,
            range=[min_x - padding, max_x + padding]
        ),
        yaxis=dict(
            tickfont=dict(family="Inter, sans-serif", size=12, color='white'),
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.15)',
            griddash='dot',
            type='category',
            automargin=True
        ),
        showlegend=False
    )
    
    return fig


# Set page config for a premium wide layout
st.set_page_config(
    page_title="risktec Underwriting Copilot",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS for theme-adaptive modern design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 600;
        color: var(--text-color);
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem;
        color: var(--text-color);
        opacity: 0.7;
        margin-bottom: 2rem;
    }
    
    /* Make the metrics stand out a bit more */
    div[data-testid="stMetricValue"] {
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Database path (relative to project root)
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'insurance.db')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'models')

# Region Metadata Map (Aligned with regions.csv / stg_regions)
REGION_MAP = {
    "R11": {"name": "Ile-de-France", "population": 11938714},
    "R21": {"name": "Champagne-Ardenne", "population": 1373935},
    "R22": {"name": "Picardie", "population": 1962150},
    "R23": {"name": "Haute-Normandie", "population": 1879146},
    "R24": {"name": "Centre", "population": 2619613},
    "R25": {"name": "Basse-Normandie", "population": 1518103},
    "R26": {"name": "Bourgogne", "population": 1693742},
    "R31": {"name": "Nord-Pas-de-Calais", "population": 4107148},
    "R41": {"name": "Lorraine", "population": 2406524},
    "R42": {"name": "Alsace", "population": 1880860},
    "R43": {"name": "Franche-Comte", "population": 1208268},
    "R52": {"name": "Pays-de-la-Loire", "population": 3676582},
    "R53": {"name": "Brittany-Bretagne", "population": 3301802},
    "R54": {"name": "Poitou-Charentes", "population": 1824367},
    "R72": {"name": "Aquitaine", "population": 3321058},
    "R73": {"name": "Midi-Pyrenees", "population": 2964308},
    "R74": {"name": "Limousin", "population": 764935},
    "R82": {"name": "Rhone-Alpes", "population": 6384816},
    "R83": {"name": "Auvergne", "population": 1388779},
    "R91": {"name": "Languedoc-Roussillon", "population": 2693275},
    "R93": {"name": "Provence-Alpes-Cote-d-Azur (PACA)", "population": 4984058},
    "R94": {"name": "Corsica-Corse", "population": 314867}
}

@st.cache_resource
def load_models():
    """Load serialized frequency and severity pipelines."""
    freq_path = os.path.join(MODELS_DIR, 'frequency_pipeline.pkl')
    sev_path = os.path.join(MODELS_DIR, 'severity_pipeline.pkl')
    
    with open(freq_path, 'rb') as f:
        freq_pipeline = pickle.load(f)
    with open(sev_path, 'rb') as f:
        sev_pipeline = pickle.load(f)
        
    return freq_pipeline, sev_pipeline

@st.cache_data
def fetch_portfolio_data(_freq_pipeline, _sev_pipeline):
    """Query data and pre-calculate predictions for historical portfolio analysis."""
    conn = duckdb.connect(DB_PATH)
    df = conn.execute("SELECT * FROM mart_pricing").fetchdf()
    conn.close()
    
    # Feature list matching the model's training columns
    feature_cols = [
        'area_code', 'vehicle_power', 'vehicle_age', 'driver_age', 
        'bonus_malus', 'vehicle_brand', 'fuel_type', 'population_density', 
        'region_code', 'region_population'
    ]
    
    # Run predictions on the historical dataset
    df['pred_frequency'] = _freq_pipeline.predict(df[feature_cols])
    df['pred_severity'] = _sev_pipeline.predict(df[feature_cols])
    df['pred_pure_premium'] = df['pred_frequency'] * df['pred_severity']
    
    # Calculate policy-level expected quantities
    df['expected_claims'] = df['pred_frequency'] * df['exposure']
    df['expected_losses'] = df['pred_pure_premium'] * df['exposure']
    
    # 1. Age Analysis Grouping
    df['age_bucket'] = pd.cut(
        df['driver_age'], 
        bins=[0, 24, 39, 59, 120], 
        labels=['18-24', '25-39', '40-59', '60+']
    )
    
    age_agg = df.groupby('age_bucket', observed=False).agg(
        total_exposure=('exposure', 'sum'),
        actual_claims=('claim_count', 'sum'),
        actual_losses=('total_claim_amount', 'sum'),
        expected_claims=('expected_claims', 'sum'),
        expected_losses=('expected_losses', 'sum')
    ).reset_index()
    
    age_agg['observed_pure_premium'] = age_agg['actual_losses'] / age_agg['total_exposure']
    age_agg['predicted_pure_premium'] = age_agg['expected_losses'] / age_agg['total_exposure']
    
    # 2. Region Analysis Grouping
    region_agg = df.groupby(['region_code', 'region_name']).agg(
        total_exposure=('exposure', 'sum'),
        actual_claims=('claim_count', 'sum'),
        actual_losses=('total_claim_amount', 'sum'),
        expected_claims=('expected_claims', 'sum'),
        expected_losses=('expected_losses', 'sum')
    ).reset_index()
    
    region_agg['observed_pure_premium'] = region_agg['actual_losses'] / region_agg['total_exposure']
    region_agg['predicted_pure_premium'] = region_agg['expected_losses'] / region_agg['total_exposure']
    
    # Sort regions by exposure for concentration analysis
    region_agg = region_agg.sort_values(by='total_exposure', ascending=False).reset_index(drop=True)
    
    # 3. Bonus/Malus Analysis Grouping
    df['bm_bucket'] = pd.cut(
        df['bonus_malus'],
        bins=[0, 59, 99, 149, 350],
        labels=['50-59 (Best)', '60-99 (Bonus)', '100-149 (Neutral/Malus)', '150+ (High Malus)']
    )
    
    bm_agg = df.groupby('bm_bucket', observed=False).agg(
        total_exposure=('exposure', 'sum'),
        actual_claims=('claim_count', 'sum'),
        actual_losses=('total_claim_amount', 'sum'),
        expected_claims=('expected_claims', 'sum'),
        expected_losses=('expected_losses', 'sum')
    ).reset_index()
    
    bm_agg['observed_pure_premium'] = bm_agg['actual_losses'] / bm_agg['total_exposure']
    bm_agg['predicted_pure_premium'] = bm_agg['expected_losses'] / bm_agg['total_exposure']
    
    # 4. Vehicle Power Analysis Grouping
    df['power_bucket'] = pd.cut(
        df['vehicle_power'],
        bins=[0, 5, 7, 10, 15],
        labels=['4-5 (Low)', '6-7 (Mid)', '8-10 (High)', '11+ (Very High)']
    )
    
    power_agg = df.groupby('power_bucket', observed=False).agg(
        total_exposure=('exposure', 'sum'),
        actual_claims=('claim_count', 'sum'),
        actual_losses=('total_claim_amount', 'sum'),
        expected_claims=('expected_claims', 'sum'),
        expected_losses=('expected_losses', 'sum')
    ).reset_index()
    
    power_agg['observed_pure_premium'] = power_agg['actual_losses'] / power_agg['total_exposure']
    power_agg['predicted_pure_premium'] = power_agg['expected_losses'] / power_agg['total_exposure']
    
    # 5. Fuel Type Analysis Grouping
    fuel_agg = df.groupby('fuel_type').agg(
        total_exposure=('exposure', 'sum'),
        actual_claims=('claim_count', 'sum'),
        actual_losses=('total_claim_amount', 'sum'),
        expected_claims=('expected_claims', 'sum'),
        expected_losses=('expected_losses', 'sum')
    ).reset_index()
    
    fuel_agg['observed_pure_premium'] = fuel_agg['actual_losses'] / fuel_agg['total_exposure']
    fuel_agg['predicted_pure_premium'] = fuel_agg['expected_losses'] / fuel_agg['total_exposure']
    
    return (age_agg, region_agg, bm_agg, power_agg, fuel_agg,
            df['exposure'].sum(), len(df), df['claim_count'].sum(), df['total_claim_amount'].sum())


def make_minimal_bar_chart(x, y, text_vals, colors, x_title, y_title, orientation='h', height=300):
    """Generate a clean, modern, gridless bar chart mimicking SHAP styling with white labels."""
    fig = go.Figure(go.Bar(
        x=x,
        y=y,
        orientation=orientation,
        marker=dict(
            color=colors,
            line=dict(width=0)
        ),
        text=text_vals,
        textposition='outside',
        textfont=dict(
            family="Inter, sans-serif",
            size=10,
            color='white'
        ),
        cliponaxis=False
    ))
    
    fig.update_layout(
        margin=dict(l=10, r=40, t=10, b=10),
        height=height,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title=x_title,
            titlefont=dict(family="Inter, sans-serif", size=12, color='white'),
            tickfont=dict(family="Inter, sans-serif", size=10, color='white'),
            showgrid=False,
            zeroline=True,
            zerolinecolor='rgba(128, 128, 128, 0.3)',
            showticklabels=True
        ),
        yaxis=dict(
            title=y_title,
            titlefont=dict(family="Inter, sans-serif", size=12, color='white'),
            tickfont=dict(family="Inter, sans-serif", size=11, color='white'),
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.1)',
            griddash='dot',
            automargin=True
        ),
        showlegend=False
    )
    return fig


def make_grouped_bar_chart(categories, val1, val2, name1="Actual", name2="Expected", colors=['#008bfb', '#ff0051'], y_title="", height=300):
    """Generate a clean, side-by-side bar chart for comparing actual vs expected with white labels."""
    fig = go.Figure(data=[
        go.Bar(
            name=name1, 
            x=categories, 
            y=val1, 
            marker=dict(color=colors[0], line=dict(width=0)),
            text=[f"€{v:,.0f}" if "PP" in name1 else f"{v:,.0f}" for v in val1],
            textposition='outside',
            textfont=dict(family="Inter, sans-serif", size=10, color='white')
        ),
        go.Bar(
            name=name2, 
            x=categories, 
            y=val2, 
            marker=dict(color=colors[1], line=dict(width=0)),
            text=[f"€{v:,.0f}" if "PP" in name2 else f"{v:,.0f}" for v in val2],
            textposition='outside',
            textfont=dict(family="Inter, sans-serif", size=10, color='white')
        )
    ])
    
    fig.update_layout(
        barmode='group',
        margin=dict(l=10, r=10, t=30, b=10),
        height=height,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            tickfont=dict(family="Inter, sans-serif", size=11, color='white'),
            showgrid=False
        ),
        yaxis=dict(
            title=y_title,
            titlefont=dict(family="Inter, sans-serif", size=12, color='white'),
            tickfont=dict(family="Inter, sans-serif", size=10, color='white'),
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.1)',
            griddash='dot',
            automargin=True
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(family="Inter, sans-serif", size=11, color='white')
        )
    )
    return fig


# Initialize app state & load pipelines
try:
    freq_pipeline, sev_pipeline = load_models()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading model pipelines. Please ensure they are trained and saved in {MODELS_DIR}.")
    st.error(str(e))
    models_loaded = False

# Main app layout (Single Page)
st.markdown('<div class="main-title">Underwriting Pricing Copilot</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Evaluate claim frequency, cost severity, and pure premium based on policy risk features.</div>', unsafe_allow_html=True)

if models_loaded:
    if 'calculated' not in st.session_state:
        st.session_state.calculated = False

    # Render Portfolio Overview (Historical Analysis) expander
    with st.spinner("Analyzing historical portfolio data..."):
        age_agg, region_agg, bm_agg, power_agg, fuel_agg, total_exposure, total_policies, total_claims, total_losses = fetch_portfolio_data(freq_pipeline, sev_pipeline)
    
    with st.expander("📊 Portfolio Overview (Historical Analysis)", expanded=True):
        #st.markdown("#### 📈 Key Portfolio Metrics")
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        with kpi_col1:
            with st.container(border=True):
                st.metric("Total Policies", f"{total_policies:,}", help="Number of policies in the database")
        with kpi_col2:
            with st.container(border=True):
                st.metric("Total Exposure (Years)", f"{total_exposure:,.2f}", help="Sum of risk exposure time in years")
        with kpi_col3:
            with st.container(border=True):
                st.metric("Total Claims", f"{total_claims:,}", help="Total number of claims filed historically")
        with kpi_col4:
            with st.container(border=True):
                observed_pp = total_losses / total_exposure
                st.metric("Observed Pure Premium", f"€{observed_pp:,.2f}", help="Historical claim cost per exposure year")
                
        st.write("")
        
        # Row 2: Driver Age Analysis
        #st.markdown("#### 👥 Driver Age Demographics & Pricing Performance")
        row2_col1, row2_col2 = st.columns(2)
        
        with row2_col1:
            with st.container(border=True):
                st.markdown("**Observed Pure Premium by Driver Age Bucket**")
                #st.write("Average historical claims cost per unit of exposure.")
                fig_age_pp = make_minimal_bar_chart(
                    x=age_agg['age_bucket'],
                    y=age_agg['observed_pure_premium'],
                    text_vals=[f"€{v:,.2f}" for v in age_agg['observed_pure_premium']],
                    colors='#ff0051',  # SHAP pink
                    x_title="Driver Age Bucket",
                    y_title="Pure Premium (€)",
                    orientation='v',
                    height=300
                )
                st.plotly_chart(fig_age_pp, use_container_width=True)
                
        with row2_col2:
            with st.container(border=True):
                st.markdown("**Actual vs. Expected Claims by Driver Age Bucket**")
                #st.write("How the XGBoost frequency model predictions compare to historical reality.")
                fig_age_claims = make_grouped_bar_chart(
                    categories=age_agg['age_bucket'],
                    val1=age_agg['actual_claims'],
                    val2=age_agg['expected_claims'],
                    name1="Actual Claims",
                    name2="Expected Claims",
                    colors=['#008bfb', '#ff0051'],  # SHAP blue, SHAP pink
                    y_title="Number of Claims",
                    height=300
                )
                st.plotly_chart(fig_age_claims, use_container_width=True)
                
        st.write("")
        
        # Row 3: Regional Risk Profiling
        #st.markdown("#### 🗺️ Regional Risk Profiling")
        row3_col1, row3_col2 = st.columns(2)
        
        # Take Top 8 regions for cleaner visuals
        top_regions = region_agg.head(8).copy()
        
        with row3_col1:
            with st.container(border=True):
                st.markdown("**Exposure Concentration by Region (Top 8)**")
                #st.write("Where our risk exposure is geographically concentrated.")
                fig_reg_exp = make_minimal_bar_chart(
                    x=top_regions['total_exposure'],
                    y=top_regions['region_name'],
                    text_vals=[f"{v:,.0f} yr" for v in top_regions['total_exposure']],
                    colors='#008bfb',  # SHAP blue
                    x_title="Total Exposure (Years)",
                    y_title="Region Name",
                    orientation='h',
                    height=320
                )
                st.plotly_chart(fig_reg_exp, use_container_width=True)
                
        with row3_col2:
            with st.container(border=True):
                st.markdown("**Actual vs. Expected Pure Premium by Region (Top 8)**")
                #st.write("Observed loss cost per exposure vs. the model's average pure premium prediction.")
                fig_reg_pp = make_grouped_bar_chart(
                    categories=top_regions['region_name'],
                    val1=top_regions['observed_pure_premium'],
                    val2=top_regions['predicted_pure_premium'],
                    name1="Observed PP",
                    name2="Expected PP",
                    colors=['#008bfb', '#ff0051'],  # SHAP blue, SHAP pink
                    y_title="Pure Premium (€)",
                    height=320
                )
                st.plotly_chart(fig_reg_pp, use_container_width=True)
        
        st.write("")
        
        # Row 4: Bonus/Malus & Vehicle Power Analysis
        row4_col1, row4_col2 = st.columns(2)
        
        with row4_col1:
            with st.container(border=True):
                st.markdown("**Actual vs. Expected Pure Premium by Bonus/Malus**")
                fig_bm_pp = make_grouped_bar_chart(
                    categories=bm_agg['bm_bucket'],
                    val1=bm_agg['observed_pure_premium'],
                    val2=bm_agg['predicted_pure_premium'],
                    name1="Observed PP",
                    name2="Expected PP",
                    colors=['#008bfb', '#ff0051'],
                    y_title="Pure Premium (€)",
                    height=320
                )
                st.plotly_chart(fig_bm_pp, use_container_width=True)
        
        with row4_col2:
            with st.container(border=True):
                st.markdown("**Actual vs. Expected Pure Premium by Vehicle Power**")
                fig_power_pp = make_grouped_bar_chart(
                    categories=power_agg['power_bucket'],
                    val1=power_agg['observed_pure_premium'],
                    val2=power_agg['predicted_pure_premium'],
                    name1="Observed PP",
                    name2="Expected PP",
                    colors=['#008bfb', '#ff0051'],
                    y_title="Pure Premium (€)",
                    height=320
                )
                st.plotly_chart(fig_power_pp, use_container_width=True)
        
        st.write("")
        
        # Row 5: Fuel Type & Actual vs Expected Claims by Bonus/Malus
        row5_col1, row5_col2 = st.columns(2)
        
        with row5_col1:
            with st.container(border=True):
                st.markdown("**Actual vs. Expected Pure Premium by Fuel Type**")
                fig_fuel_pp = make_grouped_bar_chart(
                    categories=fuel_agg['fuel_type'],
                    val1=fuel_agg['observed_pure_premium'],
                    val2=fuel_agg['predicted_pure_premium'],
                    name1="Observed PP",
                    name2="Expected PP",
                    colors=['#008bfb', '#ff0051'],
                    y_title="Pure Premium (€)",
                    height=320
                )
                st.plotly_chart(fig_fuel_pp, use_container_width=True)
        
        with row5_col2:
            with st.container(border=True):
                st.markdown("**Actual vs. Expected Claims by Bonus/Malus**")
                fig_bm_claims = make_grouped_bar_chart(
                    categories=bm_agg['bm_bucket'],
                    val1=bm_agg['actual_claims'],
                    val2=bm_agg['expected_claims'],
                    name1="Actual Claims",
                    name2="Expected Claims",
                    colors=['#008bfb', '#ff0051'],
                    y_title="Number of Claims",
                    height=320
                )
                st.plotly_chart(fig_bm_claims, use_container_width=True)

    with st.expander("⚙️ Underwriting Settings", expanded=False):
        with st.form("underwriting_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                with st.container(border=True):
                    driver_age = st.slider("Driver Age", min_value=18, max_value=95, value=35, step=1)
                with st.container(border=True):
                    vehicle_power = st.number_input("Power Group", min_value=4, max_value=15, value=6, step=1)
                with st.container(border=True):
                    fuel_type = st.selectbox("Fuel Type", options=["Regular", "Diesel"], index=0)
            
            with col2:
                with st.container(border=True):
                    bonus_malus = st.slider("Bonus/Malus", min_value=50, max_value=350, value=100, step=5)
                with st.container(border=True):
                    population_density = st.number_input("Pop Density", min_value=1, max_value=100000, value=250, step=50)
                with st.container(border=True):
                    area_code = st.selectbox("Area Code", options=["A", "B", "C", "D", "E", "F"], index=2)
            
            with col3:
                with st.container(border=True):
                    vehicle_age = st.slider("Vehicle Age", min_value=0, max_value=50, value=5, step=1)
                with st.container(border=True):
                    vehicle_brand = st.selectbox("Car Brand", options=[f"B{i}" for i in [1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14]], index=0)
                with st.container(border=True):
                    region_code = st.selectbox("Region", options=list(REGION_MAP.keys()), index=0, format_func=lambda x: f"{x} - {REGION_MAP[x]['name']}")
            
            st.write("")
            submitted = st.form_submit_button("Run Calculations", type="primary", use_container_width=True)
            if submitted:
                st.session_state.calculated = True
                st.rerun()
    
    if st.session_state.calculated:
        # Map selected region to metadata
        region_name = REGION_MAP[region_code]["name"]
        region_population = REGION_MAP[region_code]["population"]
        
        # Create a single-row DataFrame matching the model's exact expectations
        input_data = pd.DataFrame([{
            'area_code': area_code,
            'vehicle_power': vehicle_power,
            'vehicle_age': vehicle_age,
            'driver_age': driver_age,
            'bonus_malus': bonus_malus,
            'vehicle_brand': vehicle_brand,
            'fuel_type': fuel_type,
            'population_density': population_density,
            'region_code': region_code,
            'region_population': region_population
        }])
        
        # Predict dynamically
        predicted_frequency = freq_pipeline.predict(input_data)[0]
        predicted_severity = sev_pipeline.predict(input_data)[0]
        pure_premium = predicted_frequency * predicted_severity
        
        with st.expander("📊 Pricing Analysis & Risk Explanation", expanded=True):
            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                with st.container(border=True):
                    st.metric("Expected Frequency", f"{predicted_frequency:.4f}", help="Expected Claims / Policy Year")
            with res_col2:
                with st.container(border=True):
                    st.metric("Expected Severity", f"€{predicted_severity:,.0f}", help="Expected Cost / Claim Event")
            with res_col3:
                with st.container(border=True):
                    st.metric("Pure Premium", f"€{pure_premium:,.0f}", help="Expected Loss Cost / Year")
            st.write("")
            with st.container(border=True):
                st.markdown("#### 🔍 Risk Explanation (SHAP)")
                #st.write("How different factors impact the pure premium.")
                
                preprocessor_freq = freq_pipeline.named_steps['preprocessor']
                model_freq = freq_pipeline.named_steps['model']
                
                preprocessed_input = preprocessor_freq.transform(input_data)
                feature_names = preprocessor_freq.get_feature_names_out()
                input_df_shap = pd.DataFrame(preprocessed_input, columns=feature_names)
                
                explainer_freq = shap.TreeExplainer(model_freq)
                shap_values_freq = explainer_freq(input_df_shap)
                
                fig_plotly = draw_plotly_bar(
                    shap_values=shap_values_freq.values[0],
                    feature_names=feature_names,
                    preprocessed_row=input_df_shap,
                    raw_row=input_data
                )
                st.plotly_chart(fig_plotly, use_container_width=True)
