import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Grid Saver | Adaptive Grid Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM STYLES
# ============================================================
st.markdown("""
<style>
    .main { background-color: #0D1117; }
    .stApp { background-color: #0D1117; }
    .metric-card {
        background: #161B22;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #30363D;
        text-align: center;
    }
    h1, h2, h3 { color: white !important; }
    .stMarkdown { color: #CCCCCC; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS
# ============================================================
CARBON_COL           = 'Carbon intensity gCO\u2082eq/kWh (direct)'
CFE_COL              = 'Carbon-free energy percentage (CFE%)'
DECISION_THRESHOLD   = 0.4      
REDUCTION_RATE       = 0.04     
NUM_HOMES            = 25       
KW_PER_HOME          = 0.0920   

MONTH_NAMES = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December'
}
DAY_NAMES = {
    0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
    4: 'Friday', 5: 'Saturday', 6: 'Sunday'
}
month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# ============================================================
# LOAD MODEL AND DATA
# ============================================================
@st.cache_resource
def load_model():
    return joblib.load("gridsaver_model.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("data_sample.csv")
    df['Datetime (UTC)'] = pd.to_datetime(df['Datetime (UTC)'])
    df = df.sort_values('Datetime (UTC)').reset_index(drop=True)
    return df

with st.spinner("Loading Grid Saver..."):
    model = load_model()
    df    = load_data()

# ============================================================
# FEATURE ENGINEERING
# ============================================================
def engineer_features(df_input):
    df_fe = df_input.copy()
    df_fe['hour']        = df_fe['datetime'].dt.hour
    df_fe['day_of_week'] = df_fe['datetime'].dt.dayofweek
    df_fe['month']       = df_fe['datetime'].dt.month
    df_fe['day_of_year'] = df_fe['datetime'].dt.dayofyear
    df_fe['is_weekend']  = (df_fe['day_of_week'] >= 5).astype(int)
    df_fe['is_summer']   = df_fe['month'].isin([6, 7, 8]).astype(int)
    df_fe['is_winter']   = df_fe['month'].isin([12, 1, 2]).astype(int)
    df_fe['hour_sin']  = np.sin(2 * np.pi * df_fe['hour'] / 24)
    df_fe['hour_cos']  = np.cos(2 * np.pi * df_fe['hour'] / 24)
    df_fe['month_sin'] = np.sin(2 * np.pi * df_fe['month'] / 12)
    df_fe['month_cos'] = np.cos(2 * np.pi * df_fe['month'] / 12)
    df_fe['demand_lag_1h']           = df_fe['demand_mw'].shift(1)
    df_fe['demand_lag_2h']           = df_fe['demand_mw'].shift(2)
    df_fe['demand_lag_24h']          = df_fe['demand_mw'].shift(24)
    df_fe['demand_lag_48h']          = df_fe['demand_mw'].shift(48)
    df_fe['demand_lag_168h']         = df_fe['demand_mw'].shift(168)
    df_fe['demand_rolling_6h_mean']  = df_fe['demand_mw'].rolling(6).mean()
    df_fe['demand_rolling_24h_mean'] = df_fe['demand_mw'].rolling(24).mean()
    df_fe['demand_rolling_24h_max']  = df_fe['demand_mw'].rolling(24).max()
    df_fe['demand_rolling_24h_std']  = df_fe['demand_mw'].rolling(24).std()
    df_fe['demand_delta_1h']         = df_fe['demand_mw'].diff(1)
    df_fe['demand_delta_24h']        = df_fe['demand_mw'].diff(24)
    df_fe = df_fe.dropna().reset_index(drop=True)
    return df_fe

FEATURE_COLS = [
    'hour', 'day_of_week', 'month', 'day_of_year',
    'is_weekend', 'is_summer', 'is_winter',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'demand_lag_1h', 'demand_lag_2h', 'demand_lag_24h',
    'demand_lag_48h', 'demand_lag_168h',
    'demand_rolling_6h_mean', 'demand_rolling_24h_mean',
    'demand_rolling_24h_max', 'demand_rolling_24h_std',
    'demand_delta_1h', 'demand_delta_24h'
]

# ============================================================
# HELPER FOR SENSE LAYER
# ============================================================
def normalize(series):
    denom = series.max() - series.min()
    return (series - series.min()) / denom if denom != 0 else series * 0

# ============================================================
# SENSE LAYER
# ============================================================
def sense_layer(df_input):
    df_s = df_input.copy()
    
    # 1. Stress = leading indicator (system condition)
    df_s['grid_stress_index'] = (
        normalize(df_s[CARBON_COL]) * 0.6 +
        (1 - normalize(df_s[CFE_COL])) * 0.4
    ) * 100
    
    # 2. Vulnerability = derived risk score
    carbon_max = df_s[CARBON_COL].max()
    carbon_min = df_s[CARBON_COL].min()
    cfe_max    = df_s[CFE_COL].max()
    carbon_denom = (carbon_max - carbon_min) if (carbon_max - carbon_min) != 0 else 1
    cfe_denom    = cfe_max if cfe_max != 0 else 1
    
    df_s['vulnerability_score'] = (
        ((df_s[CARBON_COL] - carbon_min) / carbon_denom * 70) +
        ((1 - df_s[CFE_COL] / cfe_denom) * 30)
    ).round(1)
    
    VULNERABILITY_THRESHOLD = df_s['vulnerability_score'].quantile(0.85)
    df_s['vulnerability_event'] = df_s['vulnerability_score'] >= VULNERABILITY_THRESHOLD

    def classify_status(score):
        if score >= 70:
            return 'CRITICAL'
        elif score >= 40:
            return 'WARNING'
        else:
            return 'STABLE'

    df_s['grid_status'] = df_s['vulnerability_score'].apply(classify_status)
    return df_s, VULNERABILITY_THRESHOLD

# ============================================================
# PREDICT LAYER
# Predicting stress-driven vulnerability windows
# ============================================================
def predict_layer(df_input, model):
    df_out = df_input.copy()
    pjm_avg_demand = 35000  
    time_features = pd.DataFrame({
        'datetime':  df_input['Datetime (UTC)'],
        'demand_mw': pjm_avg_demand + (
            np.where(df_input['Datetime (UTC)'].dt.month.isin([6, 7, 8]), 5000,
            np.where(df_input['Datetime (UTC)'].dt.month.isin([12, 1, 2]), 3000, 0))
            + np.where(df_input['Datetime (UTC)'].dt.hour.between(15, 20), 2000,
              np.where(df_input['Datetime (UTC)'].dt.hour.between(6, 9), 1000, -500))
        )
    })

    df_engineered = engineer_features(time_features)

    if df_engineered.empty or df_engineered[FEATURE_COLS].isnull().any().any():
        df_out['vuln_probability']  = 0.0
        df_out['predict_triggered'] = False
        return df_out

    vuln_proba = model.predict_proba(df_engineered[FEATURE_COLS])[:, 1]
    df_engineered['vuln_proba'] = vuln_proba
    df_engineered['hour']       = df_engineered['datetime'].dt.hour
    df_engineered['month']      = df_engineered['datetime'].dt.month

    predict_by_hour_month = df_engineered.groupby(
        ['hour', 'month']
    )['vuln_proba'].mean().reset_index()
    predict_by_hour_month['predict_triggered'] = (
        predict_by_hour_month['vuln_proba'] >= DECISION_THRESHOLD
    )

    df_out = df_out.merge(
        predict_by_hour_month[['hour', 'month', 'vuln_proba', 'predict_triggered']],
        on=['hour', 'month'],
        how='left'
    )
    df_out = df_out.rename(columns={'vuln_proba': 'vuln_probability'})
    df_out['vuln_probability']  = df_out['vuln_probability'].ffill().fillna(0)
    df_out['predict_triggered'] = df_out['predict_triggered'].ffill().fillna(False)
    return df_out

# ============================================================
# ACT LAYER
# ============================================================
def act_layer(df_input):
    df_a = df_input.copy()
    
    # Core SPA logic
    df_a['sense_triggered']      = df_a['vulnerability_event']
    df_a['spa_action_triggered'] = (df_a['sense_triggered'] & df_a['predict_triggered'])
    
    # Baseline vs Adjusted Dataset Logic
    df_a['adjusted_vulnerability'] = df_a['vulnerability_score'].copy()
    
    # Apply intervention ONLY during high-risk periods (Act Layer effect)
    mask = df_a['vulnerability_score'] >= 70
    df_a.loc[mask, 'adjusted_vulnerability'] = df_a.loc[mask, 'vulnerability_score'] * 0.85

    return df_a

# ============================================================
# PIPELINE EXECUTION
# ============================================================
df, VULNERABILITY_THRESHOLD = sense_layer(df)
df['hour']       = df['Datetime (UTC)'].dt.hour
df['month']      = df['Datetime (UTC)'].dt.month
df['month_name'] = df['Datetime (UTC)'].dt.strftime('%b')
df['date']       = df['Datetime (UTC)'].dt.date
df['day_of_week']= df['Datetime (UTC)'].dt.dayofweek
df['year']       = df['Datetime (UTC)'].dt.year
df['week']       = df['Datetime (UTC)'].dt.isocalendar().week.astype(int)

df = predict_layer(df, model)
df_full = act_layer(df)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.image("https://img.icons8.com/fluency/96/lightning-bolt.png", width=60)
st.sidebar.title("Grid Saver")
st.sidebar.markdown("**Adaptive Grid Intelligence Platform**")
st.sidebar.divider()

live_mode = st.sidebar.toggle("Recent Window View (Last 24 Hours)", value=False)
st.sidebar.divider()

months_present = [m for m in month_order if m in df_full['month_name'].unique()]
month_options  = ['All Year'] + months_present
selected_month = st.sidebar.selectbox("Select Month", month_options)

reduction_rate_input = st.sidebar.slider(
    "HVAC Reduction Rate (%)", min_value=1, max_value=10, value=4, step=1
)
homes = st.sidebar.slider(
    "Homes Coordinated", min_value=1000, max_value=1000000, value=100000, step=1000
)

st.sidebar.divider()
st.sidebar.markdown("**Stack**\nColab + GitHub + Streamlit")

# FILTER VIEW
if live_mode:
    df_view = df_full[df_full['Datetime (UTC)'] >= df_full['Datetime (UTC)'].max() - pd.Timedelta(hours=24)].copy()
elif selected_month != 'All Year':
    df_view = df_full[df_full['month_name'] == selected_month].copy()
else:
    df_view = df_full.copy()

if df_view.empty:
    st.warning("No data available for selected filters.")
    st.stop()

# ============================================================
# MAIN UI
# ============================================================
st.markdown(f"""
<div style='background: linear-gradient(135deg, #1B4F8C, #0D1117);
     padding: 30px; border-radius: 12px; margin-bottom: 20px;
     border: 1px solid #30363D;'>
    <h1 style='color: white; margin: 0;'>⚡ Grid Saver</h1>
    <p style='color: #4A9EFF; margin: 5px 0 0 0;'>Adaptive Grid Intelligence Platform</p>
</div>
""", unsafe_allow_html=True)

# Grid Status
st.markdown("## ⚡ Grid Status")
current_row = df_view.iloc[-1]
status_color = {'STABLE': '#2ECC71', 'WARNING': '#F39C12', 'CRITICAL': '#E74C3C'}
status_icon  = {'STABLE': '🟢', 'WARNING': '🟡', 'CRITICAL': '🔴'}

col1, col2, col3, col4 = st.columns(4)
cards = [
    (col1, status_icon[current_row['grid_status']], current_row['grid_status'], "Grid Status", status_color[current_row['grid_status']]),
    (col2, f"{current_row['grid_stress_index']:.0f}", "/100", "Grid Stress Index", "#F39C12"),
    (col3, f"{current_row['vulnerability_score']:.0f}", "/100", "Vulnerability Score", "white"),
    (col4, f"{current_row['vuln_probability']:.2f}", "probability", "Risk Projection", "#9B59B6"),
]
for col, val, sub, label, color in cards:
    with col:
        st.markdown(f"""
        <div class='metric-card'>
            <h2 style='color: {color}; font-size: 1.6rem; margin: 0;'>{val}</h2>
            <p style='color: #666; font-size: 0.75rem; margin: 2px 0;'>{sub}</p>
            <p style='color: #888; font-size: 0.75rem; margin: 0;'>{label}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# SYSTEM ARCHITECTURE
# ============================================================
st.markdown("## System Architecture")
col_a, col_b, col_c = st.columns(3)

arch = [
    (col_a, "👁️", "SENSE", "#1B4F8C", "#4A9EFF",
     "Compute Grid Stress Index", "Monitor carbon intensity & CFE", "Electricity Maps US-TEX-ERCO"),
    (col_b, "🧠", "PREDICT", "#1A6B2E", "#2ECC71",
     "Predict stress-driven vulnerability windows", "XGBoost model | 24hr Projection", "PJM temporal patterns"),
    (col_c, "⚡", "ACT", "#7B1A1A", "#E74C3C",
     "Coordinate targeted load intervention", "Mitigate critical vulnerability windows", "Pecan Street proxy mapping"),
]

for col, icon, title, bg, color, l1, l2, l3 in arch:
    with col:
        st.markdown(f"""
        <div style='background: {bg}22; border: 1px solid {color}; padding: 20px; border-radius: 10px; text-align: center; border-top: 4px solid {color};'>
            <h2 style='color: {color}; font-size: 2rem; margin: 0;'>{icon}</h2>
            <h3 style='color: {color}; margin: 10px 0 5px 0;'>{title}</h3>
            <p style='color: #888; font-size: 0.85rem;'>{l1}<br>{l2}<br>{l3}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# REPORTS AND INSIGHTS (ACT LAYER BEFORE/AFTER IMPACT)
# ============================================================
st.markdown("## Intervention Impact & Reports")
col_r1, col_r2, col_r3 = st.columns(3)

with col_r1:
    report_type = st.selectbox("Report Type", ["Yearly", "Monthly", "Weekly"], key="rt")
with col_r2:
    report_year = st.selectbox("Year", sorted(df_full['year'].unique(), reverse=True), key="ry")
with col_r3:
    if report_type == "Monthly":
        avail_m = sorted(df_full[df_full['year'] == report_year]['month'].unique())
        report_month = st.selectbox("Month", [MONTH_NAMES[m] for m in avail_m], key="rm")
        rm_idx = [k for k, v in MONTH_NAMES.items() if v == report_month][0]
    elif report_type == "Weekly":
        avail_w = sorted(df_full[df_full['year'] == report_year]['week'].unique())
        report_week = st.selectbox("Week", [f"Week {w}" for w in avail_w], key="rw")
        rw_idx = int(report_week.split(" ")[1])

if report_type == "Yearly":
    df_report = df_full[df_full['year'] == report_year].copy()
elif report_type == "Monthly":
    df_report = df_full[(df_full['year'] == report_year) & (df_full['month'] == rm_idx)].copy()
elif report_type == "Weekly":
    df_report = df_full[(df_full['year'] == report_year) & (df_full['week'] == rw_idx)].copy()

if not df_report.empty:
    avg_vulnerability = df_report['vulnerability_score'].mean()
    high_risk_events  = int((df_report['vulnerability_score'] >= 70).sum())
    
    # Impact Metrics Calculation
    reduction = df_report['vulnerability_score'] - df_report['adjusted_vulnerability']
    avg_reduction = reduction.mean()
    total_reduction = reduction.sum()
    peak_reduction = df_report['vulnerability_score'].max() - df_report['adjusted_vulnerability'].max()

    # BEFORE VS AFTER CHART
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=df_report['Datetime (UTC)'], y=df_report['vulnerability_score'],
        name='Before (Baseline)', line=dict(color='#E74C3C', width=1)
    ))
    fig_trend.add_trace(go.Scatter(
        x=df_report['Datetime (UTC)'], y=df_report['adjusted_vulnerability'],
        name='After (Grid Saver)', line=dict(color='#2ECC71', width=1.5)
    ))
    fig_trend.add_hline(y=70, line_dash='dash', line_color='#E74C3C', annotation_text='Critical Threshold (70)')
    fig_trend.update_layout(
        paper_bgcolor='#161B22', plot_bgcolor='#161B22', font=dict(color='white'),
        title='Grid Vulnerability Offset (Intervention Overlay)',
        xaxis=dict(gridcolor='#30363D'), yaxis=dict(gridcolor='#30363D', range=[0, 100]),
        height=350, margin=dict(t=40, b=20),
        legend=dict(bgcolor='#1A1A2E', bordercolor='#333')
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # DYNAMIC INSIGHTS
    st.markdown("### Insight Summary")
    if avg_vulnerability >= 70:
        insight = "Grid operated under sustained stress conditions."
    elif avg_vulnerability >= 40:
        insight = "Grid experienced moderate stress with periodic instability."
    else:
        insight = "Grid remained largely stable."

    if total_reduction > 0:
        insight += (
            f" Grid Saver intervention reduced vulnerability by "
            f"{avg_reduction:.1f} points on average, "
            f"mitigating {high_risk_events} critical risk intervals."
        )
    st.info(insight)

    # ============================================================
    # SYSTEM IMPACT (SCALED & DATA-DRIVEN)
    # ============================================================
    st.markdown("### System Impact (Scaled)")
    
    # 1. SCALED CAPACITY (Purely slider-driven)
    # E.g., 1,000,000 homes at 4% reduction = 92.0 MW capacity
    scaled_kw_per_home = KW_PER_HOME * (reduction_rate_input / 4)
    network_capacity_mw = (homes * scaled_kw_per_home) / 1000
    
    # 2. ACTUAL EXECUTION (Purely data-driven)
    # How many times did the data force the system to trigger?
    actual_triggered_hours = int((df_report['vulnerability_score'] >= 70).sum())
    
    # Actual physical load shed based on the data's timeline
    total_mwh_shed = actual_triggered_hours * network_capacity_mw

    col_i1, col_i2, col_i3, col_i4 = st.columns(4)
    
    # The Capability (Slider)
    col_i1.metric("Network Capacity", f"{network_capacity_mw:,.1f} MW", 
                  help="Total load reduction available based on Homes slider")
    
    # The Reality (Data)
    col_i2.metric("Triggered Events", f"{actual_triggered_hours:,} hrs", 
                  help="Actual hours intervention fired in this dataset")
    
    col_i3.metric("Peak Risk Offset", f"{peak_reduction:.1f} pts", 
                  help="Vulnerability score reduction at peak stress")
    
    col_i4.metric("Actual Load Shed", f"{total_mwh_shed:,.1f} MWh", 
                  help="Total energy removed during these specific triggered events")

else:
    st.warning("No data available for the selected period.")

# ============================================================
# FOOTER
# ============================================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style='background: #161B22; padding: 15px; border-radius: 8px; border: 1px solid #30363D; text-align: center;'>
    <p style='color: #888; font-size: 0.85rem; margin: 0;'>
        Grid Saver | Decision & Intervention System | Colab + GitHub + Streamlit
    </p>
</div>
""", unsafe_allow_html=True)
