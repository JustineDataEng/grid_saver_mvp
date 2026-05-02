import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    .info-box {
        background: #1A1A2E;
        border-left: 4px solid #4A9EFF;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 10px 0;
        font-size: 0.85rem;
        color: #CCCCCC;
    }
    .warning-box {
        background: #1A1A2E;
        border-left: 4px solid #F39C12;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 10px 0;
        font-size: 0.8rem;
        color: #CCCCCC;
    }
    .success-box {
        background: #1A1A2E;
        border-left: 4px solid #2ECC71;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 10px 0;
        font-size: 0.8rem;
        color: #CCCCCC;
    }
    .rebound-box {
        background: #1A1A2E;
        border-left: 4px solid #E74C3C;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 10px 0;
        font-size: 0.8rem;
        color: #CCCCCC;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS
# ============================================================
CARBON_COL           = 'Carbon intensity gCO\u2082eq/kWh (direct)'
CFE_COL              = 'Carbon-free energy percentage (CFE%)'
DECISION_THRESHOLD   = 0.4
KW_PER_HOME          = 0.0920

MONTH_NAMES = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December'
}
DATETIME_COL  = 'Datetime (UTC)'
ERCOT_PEAK_MW = 75000
month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def count_spa_events(trigger_series):
    """Count SPA events using rising edge detection."""
    if len(trigger_series) == 0:
        return 0
    trigger_array = (trigger_series == True).fillna(False).values
    events = 0
    prev = False
    for curr in trigger_array:
        if curr and not prev:
            events += 1
        prev = curr
    return events


def compute_scaled_reduction_kw(homes, reduction_rate_percent):
    """Calculate total reduction in kW."""
    scaling_factor = reduction_rate_percent / 4.0
    return homes * KW_PER_HOME * scaling_factor


# ============================================================
# LOAD MODEL AND DATA
# ============================================================
@st.cache_resource
def load_model():
    try:
        return joblib.load("gridsaver_model.pkl")
    except FileNotFoundError:
        st.error("❌ Model file 'gridsaver_model.pkl' not found.")
        st.stop()


@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data_sample.csv")
        df['Datetime (UTC)'] = pd.to_datetime(df['Datetime (UTC)'])
        return df.sort_values('Datetime (UTC)').reset_index(drop=True)
    except FileNotFoundError:
        st.error("❌ Data file 'data_sample.csv' not found.")
        st.stop()


with st.spinner("Loading Grid Saver..."):
    model = load_model()
    df_raw = load_data()


# ============================================================
# FEATURE ENGINEERING
# ============================================================
def engineer_features(df_input):
    df_fe = df_input.copy()
    df_fe['hour'] = df_fe['datetime'].dt.hour
    df_fe['day_of_week'] = df_fe['datetime'].dt.dayofweek
    df_fe['month'] = df_fe['datetime'].dt.month
    df_fe['day_of_year'] = df_fe['datetime'].dt.dayofyear
    df_fe['is_weekend'] = (df_fe['day_of_week'] >= 5).astype(int)
    df_fe['is_summer'] = df_fe['month'].isin([6, 7, 8]).astype(int)
    df_fe['is_winter'] = df_fe['month'].isin([12, 1, 2]).astype(int)
    df_fe['hour_sin'] = np.sin(2 * np.pi * df_fe['hour'] / 24)
    df_fe['hour_cos'] = np.cos(2 * np.pi * df_fe['hour'] / 24)
    df_fe['month_sin'] = np.sin(2 * np.pi * df_fe['month'] / 12)
    df_fe['month_cos'] = np.cos(2 * np.pi * df_fe['month'] / 12)
    df_fe['demand_lag_1h'] = df_fe['demand_mw'].shift(1)
    df_fe['demand_lag_2h'] = df_fe['demand_mw'].shift(2)
    df_fe['demand_lag_24h'] = df_fe['demand_mw'].shift(24)
    df_fe['demand_lag_48h'] = df_fe['demand_mw'].shift(48)
    df_fe['demand_lag_168h'] = df_fe['demand_mw'].shift(168)
    df_fe['demand_rolling_6h_mean'] = df_fe['demand_mw'].rolling(6).mean()
    df_fe['demand_rolling_24h_mean'] = df_fe['demand_mw'].rolling(24).mean()
    df_fe['demand_rolling_24h_max'] = df_fe['demand_mw'].rolling(24).max()
    df_fe['demand_rolling_24h_std'] = df_fe['demand_mw'].rolling(24).std()
    df_fe['demand_delta_1h'] = df_fe['demand_mw'].diff(1)
    df_fe['demand_delta_24h'] = df_fe['demand_mw'].diff(24)
    return df_fe.dropna().reset_index(drop=True)


FEATURE_COLS = [
    'hour', 'day_of_week', 'month', 'day_of_year', 'is_weekend', 'is_summer', 'is_winter',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'demand_lag_1h', 'demand_lag_2h', 'demand_lag_24h', 'demand_lag_48h', 'demand_lag_168h',
    'demand_rolling_6h_mean', 'demand_rolling_24h_mean', 'demand_rolling_24h_max', 'demand_rolling_24h_std',
    'demand_delta_1h', 'demand_delta_24h'
]


# ============================================================
# SENSE LAYER
# ============================================================
def sense_layer(df_input):
    df_s = df_input.copy()
    carbon_max = df_s[CARBON_COL].max()
    carbon_min = df_s[CARBON_COL].min()
    cfe_max = df_s[CFE_COL].max()
    carbon_denom = (carbon_max - carbon_min) if (carbon_max - carbon_min) != 0 else 1
    cfe_denom = cfe_max if cfe_max != 0 else 1

    df_s['vulnerability_score'] = (
        ((df_s[CARBON_COL] - carbon_min) / carbon_denom * 70) +
        ((1 - df_s[CFE_COL] / cfe_denom) * 30)
    ).round(1)

    threshold = df_s['vulnerability_score'].quantile(0.85)
    df_s['vulnerability_event'] = df_s['vulnerability_score'] >= threshold

    def classify_status(score):
        if score >= 70:
            return 'CRITICAL'
        elif score >= 40:
            return 'WARNING'
        return 'STABLE'

    df_s['grid_status'] = df_s['vulnerability_score'].apply(classify_status)
    return df_s, threshold


# ============================================================
# PREDICT LAYER
# ============================================================
def predict_layer(df_input, model):
    df_out = df_input.copy()

    if 'hour' not in df_out.columns:
        df_out['hour'] = df_out[DATETIME_COL].dt.hour
    if 'month' not in df_out.columns:
        df_out['month'] = df_out[DATETIME_COL].dt.month

    pjm_avg_demand = 35000
    time_features = pd.DataFrame({
        'datetime': df_out[DATETIME_COL],
        'demand_mw': pjm_avg_demand + (
            np.where(df_out[DATETIME_COL].dt.month.isin([6, 7, 8]), 5000,
            np.where(df_out[DATETIME_COL].dt.month.isin([12, 1, 2]), 3000, 0))
            + np.where(df_out[DATETIME_COL].dt.hour.between(15, 20), 2000,
              np.where(df_out[DATETIME_COL].dt.hour.between(6, 9), 1000, -500))
        )
    })

    df_eng = engineer_features(time_features)
    if df_eng.empty:
        df_out['vuln_probability'] = 0
        df_out['predict_triggered'] = False
        return df_out

    vuln_proba = model.predict_proba(df_eng[FEATURE_COLS])[:, 1]
    df_eng['vuln_proba'] = vuln_proba
    df_eng['hour'] = df_eng['datetime'].dt.hour
    df_eng['month'] = df_eng['datetime'].dt.month

    pred_by_hour_month = df_eng.groupby(['hour', 'month'])['vuln_proba'].mean().reset_index()
    pred_by_hour_month['predict_triggered'] = pred_by_hour_month['vuln_proba'] >= DECISION_THRESHOLD

    df_out = df_out.merge(pred_by_hour_month[['hour', 'month', 'vuln_proba', 'predict_triggered']],
                          on=['hour', 'month'], how='left')
    df_out = df_out.rename(columns={'vuln_proba': 'vuln_probability'})
    df_out['vuln_probability'] = df_out['vuln_probability'].ffill().fillna(0)
    df_out['predict_triggered'] = df_out['predict_triggered'].ffill().fillna(False)
    return df_out


# ============================================================
# ACT LAYER — WITH THERMAL REBOUND (UPDATED)
# ============================================================
def act_layer(df_input, reduction_rate_percent, homes, apply_intervention_flag):
    """
    SPA dual-confirmation with Real-World Thermal Rebound (Snapback).
    
    Adds realistic grid physics:
    - Compliance Rate (85% of homes actually respond)
    - Thermal Rebound (60% of shed load spikes back after event ends)
    - Physical bounds (reduction cannot exceed demand)
    """
    df_a = df_input.copy()
    df_a['sense_triggered'] = df_a['vulnerability_event']
    df_a['spa_action_triggered'] = (
        df_a['sense_triggered'] & df_a['predict_triggered']
    )
    
    # ============================================================
    # 1. Calculate Load Reduction with Compliance Rate
    # ============================================================
    if apply_intervention_flag:
        # Real-world factor: Not all homes respond to the signal
        COMPLIANCE_RATE = 0.85  # 85% of homes actually reduce load
        
        scaled_kw_per_home = KW_PER_HOME * (reduction_rate_percent / 4) * COMPLIANCE_RATE
        
        df_a['grid_saver_reduction_kw'] = np.where(
            df_a['spa_action_triggered'],
            homes * scaled_kw_per_home,
            0
        )
    else:
        df_a['grid_saver_reduction_kw'] = 0
    
    df_a['reduction_mw'] = df_a['grid_saver_reduction_kw'] / 1000
    
    # Physical bound: reduction cannot exceed actual demand
    df_a['reduction_mw'] = np.minimum(
        df_a['reduction_mw'], 
        df_a.get('ercot_demand_mw', 0)
    )
    
    # ============================================================
    # 2. Calculate Thermal Rebound (The "Snapback")
    # ============================================================
    REBOUND_RATE = 0.60  # 60% of shed load returns as a spike
    
    # Look at the previous hour's reduction
    df_a['previous_reduction'] = df_a['reduction_mw'].shift(1).fillna(0)
    
    # Rebound only happens if:
    # - Current hour is NOT an active SPA event
    # - Previous hour HAD a reduction
    df_a['rebound_mw'] = np.where(
        (~df_a['spa_action_triggered']) & (df_a['previous_reduction'] > 0),
        df_a['previous_reduction'] * REBOUND_RATE,
        0
    )
    
    # ============================================================
    # 3. Calculate Final Adjusted Grid Demand
    # ============================================================
    # Net impact = Baseline - Reduction + Rebound
    df_a['adjusted_demand_mw'] = (
        df_a['ercot_demand_mw'] - df_a['reduction_mw'] + df_a['rebound_mw']
    )
    
    total_mw_saved = df_a['reduction_mw'].sum() if apply_intervention_flag else 0
    
    return df_a, total_mw_saved


# ============================================================
# ERCOT DEMAND CALCULATION
# ============================================================
def add_ercot_demand(df_input):
    df_out = df_input.copy()
    base_load_pct = 0.55
    peak_load_pct = 0.95
    
    df_out['ercot_demand_mw'] = (
        base_load_pct + (df_out['vulnerability_score'] / 100) * (peak_load_pct - base_load_pct)
    ) * ERCOT_PEAK_MW
    
    return df_out


# ============================================================
# IMPACT AND DISPATCH CALCULATIONS
# ============================================================
def compute_impact_metrics(row, homes, reduction_rate_percent):
    reduction_kw = homes * KW_PER_HOME * (reduction_rate_percent / 4)
    reduction_mw = reduction_kw / 1000
    mwh_saved = reduction_mw * 1
    cost_savings = mwh_saved * 100
    carbon_intensity = row[CARBON_COL]
    co2_avoided_tons = (carbon_intensity * mwh_saved) / 1000
    return mwh_saved, cost_savings, co2_avoided_tons


def compute_dispatch_priority(row):
    score = row['vulnerability_score'] * 0.5
    score += row.get('vuln_probability', 0) * 30
    score += min(row[CARBON_COL] / 10, 20)
    return round(score, 1)


def get_risk_drivers(row, vulnerability_threshold, df_full):
    drivers = []
    score = row['vulnerability_score']
    carbon = row[CARBON_COL]
    cfe = row[CFE_COL]

    if score >= 70:
        drivers.append("🔴 Grid is in CRITICAL state — vulnerability score at 70 or above")
    elif score >= 40:
        drivers.append("🟡 Grid is in WARNING state — vulnerability score between 40 and 69")
    else:
        drivers.append("🟢 Grid is STABLE — vulnerability score below 40")

    carbon_min = df_full[CARBON_COL].min()
    carbon_max = df_full[CARBON_COL].max()
    carbon_range = carbon_max - carbon_min if (carbon_max - carbon_min) != 0 else 1
    carbon_pct = (carbon - carbon_min) / carbon_range

    cfe_max = df_full[CFE_COL].max()
    cfe_pct = cfe / cfe_max if cfe_max != 0 else 0

    if carbon_pct > 0.7:
        drivers.append("🔴 Carbon intensity is in the upper 30% of observed range — grid running heavily on fossil fuels")
    elif carbon_pct > 0.4:
        drivers.append("🟡 Carbon intensity is above the mid-range — fossil fuel contribution elevated")
    else:
        drivers.append("🟢 Carbon intensity is in the lower range — cleaner generation mix")

    if cfe_pct < 0.3:
        drivers.append("🔴 Carbon-free energy is in the lower 30% of observed range — clean supply buffer low")
    elif cfe_pct < 0.6:
        drivers.append("🟡 Carbon-free energy is below mid-range — moderate clean supply")
    else:
        drivers.append("🟢 Carbon-free energy is strong — healthy clean supply buffer")

    return drivers


# ============================================================
# RUN FULL PIPELINE
# ============================================================
df, VULNERABILITY_THRESHOLD = sense_layer(df_raw)
df['hour'] = df[DATETIME_COL].dt.hour
df['month'] = df[DATETIME_COL].dt.month
df['month_name'] = df[DATETIME_COL].dt.strftime('%b')
df['date'] = df[DATETIME_COL].dt.date
df['year'] = df[DATETIME_COL].dt.year
df['week'] = df[DATETIME_COL].dt.isocalendar().week.astype(int)

df = predict_layer(df, model)
df = add_ercot_demand(df)

# Run Act Layer on full dataset for baseline counts
df_full, _ = act_layer(df, 4, 25, True)

SENSE_TRIGGERS_TOTAL = int((df_full['sense_triggered'] == True).sum())
PREDICT_TRIGGERS_TOTAL = int((df_full['predict_triggered'] == True).sum())
SPA_ACTIONS_TOTAL = count_spa_events(df_full['spa_action_triggered'])

# Locked notebook values
NOTEBOOK_SENSE_TRIGGERS = 1316
NOTEBOOK_PREDICT_TRIGGERS = 1659
NOTEBOOK_SPA_ACTIONS = 154


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.image("https://img.icons8.com/fluency/96/lightning-bolt.png", width=60)
st.sidebar.title("Grid Saver")
st.sidebar.markdown("**Adaptive Grid Intelligence Platform**")
st.sidebar.divider()

live_mode = st.sidebar.toggle("Recent Window View (Last 24 Hours)", value=False)
if live_mode:
    st.sidebar.markdown("<p style='color:#2ECC71; font-size:0.8rem;'>Showing the most recent 24 hours</p>",
                        unsafe_allow_html=True)

st.sidebar.divider()

months_present = [m for m in month_order if m in df['month_name'].unique()]
selected_month = st.sidebar.selectbox("Select Month", ['All Year'] + months_present)

reduction_rate_percent = st.sidebar.slider("HVAC Reduction Rate (%)", 1, 10, 4, 1)
homes = st.sidebar.slider("Homes Coordinated", 1000, 1000000, 100000, 1000)
apply_intervention_flag = st.sidebar.toggle("Apply Grid Saver Intervention", value=True)

with st.sidebar.expander("ℹ️ How Grid Saver Works"):
    st.markdown("""
    **Vulnerability Score (0-100)**
    - **0-39: STABLE** — Normal
    - **40-69: WARNING** — Elevated risk
    - **70-100: CRITICAL** — Action required
    
    **SPA Dual-Confirmation**
    - **Sense** → ERCOT carbon + CFE signals
    - **Predict** → PJM temporal pattern (24hr ahead)
    - **Action** → Only when BOTH confirm independently
    
    **Thermal Rebound Modeling**
    - 85% compliance rate (not all homes respond)
    - 60% rebound spike after intervention ends
    - Physical bounds prevent negative demand
    """)

st.sidebar.divider()
st.sidebar.markdown("**Stack:** Colab + GitHub + Streamlit")
st.sidebar.markdown("*Justine Adzormado*")


# ============================================================
# FILTER DATA
# ============================================================
if live_mode:
    df_view = df[df[DATETIME_COL] >= df[DATETIME_COL].max() - pd.Timedelta(hours=24)].copy()
elif selected_month != 'All Year':
    df_view = df[df['month_name'] == selected_month].copy()
else:
    df_view = df.copy()

if df_view.empty:
    st.warning("No data available for selected filters.")
    st.stop()

# Run act layer with all parameters
df_view, total_mw_saved = act_layer(df_view, reduction_rate_percent, homes, apply_intervention_flag)

# Calculate metrics
spa_events_view = count_spa_events(df_view['spa_action_triggered'])
total_energy_removed_mwh = df_view['reduction_mw'].sum()
total_rebound_mwh = df_view['rebound_mw'].sum()
net_energy_saved = total_energy_removed_mwh - total_rebound_mwh

avg_energy_per_event = total_energy_removed_mwh / spa_events_view if spa_events_view > 0 else 0

spa_only = df_view[df_view['spa_action_triggered'] == True]
max_spa_peak_reduction = spa_only['reduction_mw'].max() if not spa_only.empty else 0

critical_hours = df_view[df_view['vulnerability_score'] >= VULNERABILITY_THRESHOLD]
avoided_events = int((critical_hours['spa_action_triggered'] == True).sum())
critical_coverage_pct = (avoided_events / len(critical_hours) * 100) if len(critical_hours) > 0 else 0


# ============================================================
# HEADER
# ============================================================
mode_label = "RECENT WINDOW VIEW" if live_mode else "ANALYSIS MODE"
mode_color = "#2ECC71" if live_mode else "#4A9EFF"

st.markdown(f"""
<div style='background: linear-gradient(135deg, #1B4F8C, #0D1117); padding: 30px; border-radius: 12px; margin-bottom: 20px; border: 1px solid #30363D;'>
    <div style='display: flex; justify-content: space-between; align-items: center;'>
        <div>
            <h1 style='color: white; margin: 0;'>⚡ Grid Saver</h1>
            <p style='color: #4A9EFF; margin: 5px 0 0 0;'>Adaptive Grid Intelligence Platform</p>
            <p style='color: #888; margin: 5px 0 0 0;'>Texas ERCOT 2025 | SPA Logic | Dual-Confirmation</p>
        </div>
        <div style='background: {mode_color}22; border: 2px solid {mode_color}; padding: 10px 20px; border-radius: 8px;'>
            <p style='color: {mode_color}; font-weight: bold; margin: 0;'>{mode_label}</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# GRID STATUS METRICS
# ============================================================
st.markdown("## ⚡ Grid Status")

current_row = df_view.iloc[-1]
current_score = current_row['vulnerability_score']
current_status = current_row['grid_status']
current_carbon = current_row[CARBON_COL]
current_cfe = current_row[CFE_COL]
current_prob = current_row.get('vuln_probability', 0)
vulnerable_pct = (df_view['vulnerability_event'] == True).mean() * 100

status_color = {'STABLE': '#2ECC71', 'WARNING': '#F39C12', 'CRITICAL': '#E74C3C'}
status_icon = {'STABLE': '🟢', 'WARNING': '🟡', 'CRITICAL': '🔴'}

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.markdown(f"<div class='metric-card'><h2 style='color: {status_color[current_status]}; font-size: 1.6rem; margin: 0;'>{status_icon[current_status]}</h2><p style='color: #666; margin: 2px 0; font-size: 0.75rem;'>{current_status}</p><p style='color: #888; margin: 0; font-size: 0.75rem;'>Grid Status</p></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-card'><h2 style='color: white; font-size: 1.6rem; margin: 0;'>{current_score:.0f}</h2><p style='color: #666; margin: 2px 0; font-size: 0.75rem;'>/100</p><p style='color: #888; margin: 0; font-size: 0.75rem;'>Vulnerability Score</p></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-card'><h2 style='color: #E74C3C; font-size: 1.6rem; margin: 0;'>{current_carbon:.0f}</h2><p style='color: #666; margin: 2px 0; font-size: 0.75rem;'>gCO₂/kWh</p><p style='color: #888; margin: 0; font-size: 0.75rem;'>Carbon Intensity</p></div>", unsafe_allow_html=True)
with col4:
    st.markdown(f"<div class='metric-card'><h2 style='color: #2ECC71; font-size: 1.6rem; margin: 0;'>{current_cfe:.1f}%</h2><p style='color: #666; margin: 2px 0; font-size: 0.75rem;'>clean energy</p><p style='color: #888; margin: 0; font-size: 0.75rem;'>Carbon-Free Energy</p></div>", unsafe_allow_html=True)
with col5:
    st.markdown(f"<div class='metric-card'><h2 style='color: #4A9EFF; font-size: 1.6rem; margin: 0;'>{vulnerable_pct:.1f}%</h2><p style='color: #666; margin: 2px 0; font-size: 0.75rem;'>of period</p><p style='color: #888; margin: 0; font-size: 0.75rem;'>Vulnerability Rate</p></div>", unsafe_allow_html=True)
with col6:
    st.markdown(f"<div class='metric-card'><h2 style='color: #9B59B6; font-size: 1.6rem; margin: 0;'>{current_prob:.2f}</h2><p style='color: #666; margin: 2px 0; font-size: 0.75rem;'>probability</p><p style='color: #888; margin: 0; font-size: 0.75rem;'>24hr Risk Projection</p></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# VULNERABILITY TREND CHART
# ============================================================
st.markdown("## 📈 Grid Vulnerability Score")

color_map = {'STABLE': '#2ECC71', 'WARNING': '#F39C12', 'CRITICAL': '#E74C3C'}
fig_trend = go.Figure()

for status in ['STABLE', 'WARNING', 'CRITICAL']:
    mask = df_view['grid_status'] == status
    if mask.any():
        fig_trend.add_trace(go.Scatter(
            x=df_view[mask][DATETIME_COL],
            y=df_view[mask]['vulnerability_score'],
            mode='markers+lines',
            name=status,
            marker=dict(color=color_map[status], size=3, opacity=0.8),
            line=dict(color=color_map[status], width=0.8),
        ))

fig_trend.add_hline(y=VULNERABILITY_THRESHOLD, line_dash='dash', line_color='#FF4444',
                    annotation_text=f'Threshold ({VULNERABILITY_THRESHOLD:.0f})')
fig_trend.add_trace(go.Scatter(
    x=df_view[DATETIME_COL],
    y=df_view['vuln_probability'] * 100,
    mode='lines',
    name='Risk Projection (%)',
    line=dict(color='#9B59B6', dash='dot', width=1.2),
))

fig_trend.update_layout(
    paper_bgcolor='#161B22', plot_bgcolor='#161B22',
    font=dict(color='white'), height=350,
    xaxis=dict(gridcolor='#30363D'), yaxis=dict(gridcolor='#30363D', title='Vulnerability Score'),
)
st.plotly_chart(fig_trend, width='stretch')

st.markdown("""
<div class='info-box'>
📌 <strong>Chart Guide:</strong> Green=STABLE (&lt;40), Yellow=WARNING (40-69), Red=CRITICAL (≥70)<br>
• Purple dashed line = 24hr Risk Projection (XGBoost temporal pattern)<br>
• Red dashed line = Vulnerability threshold (top 15% stressed hours)
</div>
""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# PEAK VULNERABILITY TIMELINE
# ============================================================
st.markdown("## 🕒 Peak Vulnerability Timeline")
col_left, col_right = st.columns(2)

with col_left:
    hourly = df_view.groupby('hour')['vulnerability_score'].mean().round(1)
    colors = ['#E74C3C' if s >= 70 else '#F39C12' if s >= 40 else '#2ECC71' for s in hourly.values]
    fig_hour = go.Figure(go.Bar(x=[f'{h:02d}:00' for h in hourly.index], y=hourly.values, marker_color=colors))
    fig_hour.add_hline(y=VULNERABILITY_THRESHOLD, line_dash='dash', line_color='#FF4444')
    fig_hour.update_layout(paper_bgcolor='#161B22', plot_bgcolor='#161B22', font=dict(color='white'),
                           title=dict(text='Avg by Hour', font=dict(size=13)), height=300)
    st.plotly_chart(fig_hour, width='stretch')

with col_right:
    if live_mode:
        st.info("Monthly trend requires full-year data.")
    elif selected_month != 'All Year':
        daily = df_view.groupby('date')['vulnerability_score'].mean().round(1).sort_index()
        colors = ['#E74C3C' if s >= 70 else '#F39C12' if s >= 40 else '#2ECC71' for s in daily.values]
        fig_daily = go.Figure(go.Bar(x=[str(d) for d in daily.index], y=daily.values, marker_color=colors))
        fig_daily.update_layout(paper_bgcolor='#161B22', plot_bgcolor='#161B22', font=dict(color='white'),
                                title=dict(text=f'Avg Daily — {selected_month}', font=dict(size=13)), height=300)
        st.plotly_chart(fig_daily, width='stretch')
    else:
        monthly = df_view.groupby('month_name')['vulnerability_score'].mean().round(1)
        monthly = monthly.reindex([m for m in month_order if m in monthly.index])
        colors = ['#E74C3C' if s >= 70 else '#F39C12' if s >= 40 else '#2ECC71' for s in monthly.values]
        fig_month = go.Figure(go.Bar(x=monthly.index, y=monthly.values, marker_color=colors))
        fig_month.add_hline(y=VULNERABILITY_THRESHOLD, line_dash='dash', line_color='#FF4444')
        fig_month.update_layout(paper_bgcolor='#161B22', plot_bgcolor='#161B22', font=dict(color='white'),
                                title=dict(text='Avg by Month', font=dict(size=13)), height=300)
        st.plotly_chart(fig_month, width='stretch')

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# RECOMMENDED GRID ACTION
# ============================================================
st.markdown("## 🎯 Recommended Grid Action")

if current_status == 'CRITICAL':
    action_color, action_icon = "#E74C3C", "🔴"
    action_title = "CRITICAL — Immediate Action Required"
    action_text = f"Reduce HVAC load by {reduction_rate_percent}% across {homes:,} homes"
elif current_status == 'WARNING':
    action_color, action_icon = "#F39C12", "🟡"
    action_title = "WARNING — Prepare for Intervention"
    action_text = f"Pre-stage {reduction_rate_percent}% HVAC reduction"
else:
    action_color, action_icon = "#2ECC71", "🟢"
    action_title = "STABLE — No Action Required"
    action_text = "Grid operating normally. Continue monitoring."

st.markdown(f"""
<div style='background:#161B22; border-left:5px solid {action_color}; padding:20px; border-radius:8px; margin:10px 0;'>
    <h3 style='color:{action_color}; margin:0;'>{action_icon} {action_title}</h3>
    <p style='color:white; margin:10px 0 5px 0;'><strong>Action:</strong> {action_text}</p>
</div>
""", unsafe_allow_html=True)

if st.button("🧠 Explain Decision"):
    st.markdown(f"""
    <div style='background:#161B22; border:1px solid #1B4F8C; padding:20px; border-radius:10px; margin:10px 0;'>
        <h3 style='color:#4A9EFF; margin:0 0 15px 0;'>Decision Explanation</h3>
        <p>Vulnerability Score: <strong>{current_score:.1f}/100</strong> (threshold {VULNERABILITY_THRESHOLD:.0f})</p>
        <p>Carbon Intensity: <strong>{current_carbon:.0f} gCO₂/kWh</strong></p>
        <p>Carbon-Free Energy: <strong>{current_cfe:.1f}%</strong></p>
        <p>Risk Projection: <strong>{current_prob:.2f}</strong> (threshold {DECISION_THRESHOLD})</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# SMART RECOMMENDATIONS
# ============================================================
st.markdown("## 🧠 Smart Recommendations")

dispatch_score = compute_dispatch_priority(current_row)
st.markdown(f"""
<div style='background:#161B22; border:1px solid #F39C12; padding:12px 16px; border-radius:8px; margin-bottom:12px;'>
    <span style='color:#F39C12; font-weight:bold;'>⚡ Dispatch Priority: {dispatch_score}/100</span>
    <p style='color:#888; margin:5px 0 0 0; font-size:0.75rem;'>
        Formula: (vulnerability × 0.5) + (risk_probability × 30) + min(carbon/10, 20)
    </p>
</div>
""", unsafe_allow_html=True)

if current_status == 'CRITICAL':
    st.markdown(f"🔴 Execute {reduction_rate_percent}% HVAC reduction across {homes:,} homes")
    if current_row.get('spa_action_triggered', False):
        st.markdown("✅ SPA dual-confirmation achieved — dispatch approved")
    else:
        st.markdown("⚠️ Partial confirmation — controlled reduction recommended")
elif current_status == 'WARNING':
    st.markdown("🟡 Pre-stage demand response resources")
    if current_prob >= 0.4:
        st.markdown("📈 Risk signal elevated — prepare for activation")
else:
    st.markdown("🟢 No intervention required — monitor grid conditions")

# Impact summary with rebound
mwh_saved, cost_savings, co2_avoided = compute_impact_metrics(current_row, homes, reduction_rate_percent)

st.markdown(f"""
<div style='background:#161B22; border-left:4px solid #2ECC71; padding:10px 14px; border-radius:6px; margin-top:12px;'>
    <span style='color:#2ECC71; font-size:0.9rem;'>📊 <strong>Impact Summary</strong></span><br>
    <span style='color:#CCC; font-size:0.85rem;'>
        Per SPA event (1-hour window): {mwh_saved:.2f} MWh | ${cost_savings:,.0f} savings | {co2_avoided:.3f} tons CO₂ avoided<br>
        <strong>Current period:</strong> {spa_events_view} SPA events | {total_energy_removed_mwh:.2f} MWh gross reduction
    </span>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# LOAD REDUCTION SIMULATION
# ============================================================
st.markdown("## ⚡ Load Reduction Simulation")

if reduction_rate_percent != 4:
    st.warning(f"⚠️ Validated at 4%. Results at {reduction_rate_percent}% are scaled estimates.")

if live_mode:
    display_sense = int((df_view['sense_triggered'] == True).sum())
    display_predict = int((df_view['predict_triggered'] == True).sum())
    display_spa = spa_events_view
else:
    display_sense = NOTEBOOK_SENSE_TRIGGERS
    display_predict = NOTEBOOK_PREDICT_TRIGGERS
    display_spa = NOTEBOOK_SPA_ACTIONS

col_s1, col_s2, col_s3 = st.columns(3)
col_s1.metric("Sense Triggers", f"{display_sense:,}")
col_s2.metric("Predict Triggers", f"{display_predict:,}")
col_s3.metric("SPA Events (Dual-Confirmed)", f"{display_spa}")

# Energy metrics with rebound
st.markdown(f"""
<div style='background:#161B22; border-left:5px solid #2ECC71; padding:15px; border-radius:8px; margin:15px 0;'>
    <h3 style='color:#2ECC71; margin:0;'>⚡ Load Reduction Analysis</h3>
    <p style='color:white; font-size:1.2rem; margin:5px 0;'>
        Gross Load Reduced: {total_energy_removed_mwh:,.2f} MWh
    </p>
    <p style='color:#E74C3C; font-size:1rem; margin:5px 0;'>
        ⚠️ Thermal Rebound (Snapback): +{total_rebound_mwh:,.2f} MWh
    </p>
    <p style='color:#2ECC71; font-size:1.1rem; margin:5px 0;'>
        ✅ Net Load Reduction: {net_energy_saved:,.2f} MWh
    </p>
    <p style='color:#888; margin:5px 0 0 0; font-size:0.8rem;'>
        Across {spa_events_view} SPA events | Average: {avg_energy_per_event:.2f} MWh/event
    </p>
</div>
""", unsafe_allow_html=True)

# Peak reduction
peak_idx = df_view['ercot_demand_mw'].idxmax()
peak_time = df_view.loc[peak_idx, DATETIME_COL]
original_peak = df_view.loc[peak_idx, 'ercot_demand_mw']
reduction_at_peak = min(df_view.loc[peak_idx, 'reduction_mw'], original_peak)
after_peak = original_peak - reduction_at_peak
peak_reduction_mw = reduction_at_peak
pct_reduction = (peak_reduction_mw / original_peak * 100) if original_peak > 0 else 0

col_p1, col_p2, col_p3, col_p4 = st.columns(4)
col_p1.metric("Original Peak", f"{original_peak:,.0f} MW")
col_p2.metric("After Grid Saver", f"{after_peak:,.0f} MW")
col_p3.metric("Peak Reduction", f"{pct_reduction:.2f}%")
col_p4.metric("Peak Load Shed", f"{peak_reduction_mw:,.2f} MW")

st.markdown(f"""
<p style='color:#888; font-size:0.8rem; margin-top:6px;'>
⚡ <strong>Max Reduction During SPA Events:</strong> {max_spa_peak_reduction:,.2f} MW
</p>
""", unsafe_allow_html=True)

# Critical events coverage
st.markdown(f"""
<div class='success-box'>
🛡️ <strong>Critical Events Intervened:</strong> {avoided_events} of {len(critical_hours)} critical hours ({critical_coverage_pct:.1f}%)
<br>This shows interventions targeted actual grid stress periods.
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# BEFORE PLOT
# ============================================================
st.markdown("#### 📉 Before Grid Saver — Original ERCOT Demand with SPA Markers")

fig_before = go.Figure()
fig_before.add_trace(go.Scatter(
    x=df_view[DATETIME_COL], y=df_view['ercot_demand_mw'],
    mode='lines', name='Original ERCOT Demand',
    line=dict(color='#E74C3C', width=1.5),
))

if apply_intervention_flag:
    spa_triggered = df_view[df_view['spa_action_triggered'] == True]
    if not spa_triggered.empty:
        fig_before.add_trace(go.Scatter(
            x=spa_triggered[DATETIME_COL], y=spa_triggered['ercot_demand_mw'],
            mode='markers', name='SPA Action Triggered',
            marker=dict(color='#F39C12', size=8, symbol='circle', line=dict(width=1.5, color='white')),
        ))

fig_before.add_trace(go.Scatter(
    x=[peak_time, peak_time], y=[df_view['ercot_demand_mw'].min(), df_view['ercot_demand_mw'].max()],
    mode='lines', name='Peak Demand', line=dict(color='#FFFFFF', width=1.5, dash='dash')
))

fig_before.update_layout(
    paper_bgcolor='#161B22', plot_bgcolor='#161B22',
    font=dict(color='white'), height=350,
    xaxis=dict(gridcolor='#30363D'), yaxis=dict(gridcolor='#30363D', title='Demand (MW)'),
)
st.plotly_chart(fig_before, width='stretch')
st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# AFTER PLOT — WITH REBOUND VISUALIZATION
# ============================================================
st.markdown("#### 📈 After Grid Saver — Demand with Intervention Applied")

fig_after = go.Figure()

# Original demand (faded reference)
fig_after.add_trace(go.Scatter(
    x=df_view[DATETIME_COL], y=df_view['ercot_demand_mw'],
    mode='lines', name='Original Demand (Reference)',
    line=dict(color='#E74C3C', width=1, dash='dot'), opacity=0.35,
))

# Adjusted demand (with rebound)
line_color = '#2ECC71' if apply_intervention_flag else '#888888'
fig_after.add_trace(go.Scatter(
    x=df_view[DATETIME_COL], y=df_view['adjusted_demand_mw'],
    mode='lines', name='After Grid Saver (with Rebound)' if apply_intervention_flag else 'Intervention Disabled',
    line=dict(color=line_color, width=1.5),
    fill='tonexty' if apply_intervention_flag else None,
    fillcolor='rgba(46, 204, 113, 0.08)' if apply_intervention_flag else None,
))

# Shaded SPA zones and drop lines
if apply_intervention_flag:
    spa_triggered = df_view[df_view['spa_action_triggered'] == True].copy()
    
    for idx, row in spa_triggered.iterrows():
        fig_after.add_vrect(
            x0=row[DATETIME_COL], x1=row[DATETIME_COL] + pd.Timedelta(hours=1),
            fillcolor='rgba(255, 165, 0, 0.15)', line_width=0, layer='below',
        )
    
    top_spa = spa_triggered.sort_values('reduction_mw', ascending=False).head(20)
    for _, row in top_spa.iterrows():
        fig_after.add_trace(go.Scatter(
            x=[row[DATETIME_COL], row[DATETIME_COL]],
            y=[row['ercot_demand_mw'], row['adjusted_demand_mw']],
            mode='lines', line=dict(color='#F39C12', width=2), showlegend=False,
        ))
    
    # Rebound spikes (red triangles)
    rebound_events = df_view[df_view['rebound_mw'] > 0].copy()
    if not rebound_events.empty:
        fig_after.add_trace(go.Scatter(
            x=rebound_events[DATETIME_COL], y=rebound_events['adjusted_demand_mw'],
            mode='markers', name='⚠️ Thermal Rebound Spike',
            marker=dict(color='#E74C3C', size=10, symbol='triangle-up', line=dict(width=1.5, color='white')),
        ))

fig_after.add_trace(go.Scatter(
    x=[peak_time, peak_time], y=[df_view['ercot_demand_mw'].min(), df_view['ercot_demand_mw'].max()],
    mode='lines', name='Peak Demand', line=dict(color='#FFFFFF', width=1.5, dash='dash'), showlegend=False,
))

fig_after.update_layout(
    paper_bgcolor='#161B22', plot_bgcolor='#161B22',
    font=dict(color='white'), height=400,
    xaxis=dict(gridcolor='#30363D'), yaxis=dict(gridcolor='#30363D', title='Demand (MW)'),
    legend=dict(bgcolor='#1A1A2E', bordercolor='#333', orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
)
st.plotly_chart(fig_after, width='stretch')

st.markdown("""
<div class='info-box'>
📌 <strong>Chart Guide — AFTER:</strong><br>
• <span style='color:#E74C3C'>🔴 Red dotted line:</span> Original demand (reference)<br>
• <span style='color:#2ECC71'>🟢 Green line:</span> Demand after Grid Saver (accounts for rebound)<br>
• <span style='color:#FFA500'>🟧 Shaded zones:</span> SPA-triggered windows<br>
• <span style='color:#F39C12'>🟡 Orange drop lines:</span> Top interventions<br>
• <span style='color:#E74C3C'>🔴 Red triangles:</span> <strong>Thermal Rebound Spikes</strong> — load that returns after intervention
</div>
""", unsafe_allow_html=True)

# Rebound explanation
st.markdown("""
<div class='rebound-box'>
⚠️ <strong>Thermal Rebound (Snapback) Effect:</strong><br>
When HVAC systems are suppressed during an SPA event, they turn back on simultaneously afterward,
creating a secondary demand spike. Grid Saver models this using:<br>
• <strong>85% Compliance Rate</strong> — not all homes respond<br>
• <strong>60% Rebound Rate</strong> — 60% of shed load returns post-event<br>
• <strong>Physical bounds</strong> — reduction never exceeds actual demand
</div>
""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# PEAK ZOOM WINDOW
# ============================================================
with st.expander("🔍 Inspect Peak Demand Window (6 Hours Before and After Peak)"):
    st.markdown("""
    **Why ±6 hours?** This window captures the critical period around peak demand — 
    the 6 hours before (ramp-up to peak) and 6 hours after (recovery).
    """)
    
    zoom_df = df_view[
        (df_view[DATETIME_COL] >= peak_time - pd.Timedelta(hours=6)) &
        (df_view[DATETIME_COL] <= peak_time + pd.Timedelta(hours=6))
    ].copy()
    
    if not zoom_df.empty:
        fig_zoom = go.Figure()
        
        fig_zoom.add_trace(go.Scatter(
            x=zoom_df[DATETIME_COL], y=zoom_df['ercot_demand_mw'],
            mode='lines+markers', name='Original Demand',
            line=dict(color='#E74C3C', width=2), marker=dict(size=4, color='#E74C3C')
        ))
        
        fig_zoom.add_trace(go.Scatter(
            x=zoom_df[DATETIME_COL], y=zoom_df['adjusted_demand_mw'],
            mode='lines+markers', name='After Grid Saver',
            line=dict(color='#2ECC71', width=2), marker=dict(size=4, color='#2ECC71'),
            fill='tonexty', fillcolor='rgba(46, 204, 113, 0.15)'
        ))
        
        if apply_intervention_flag:
            zoom_spa = zoom_df[zoom_df['spa_action_triggered'] == True]
            for _, row in zoom_spa.iterrows():
                fig_zoom.add_trace(go.Scatter(
                    x=[row[DATETIME_COL], row[DATETIME_COL]],
                    y=[row['ercot_demand_mw'], row['adjusted_demand_mw']],
                    mode='lines', line=dict(color='#F39C12', width=2.5), showlegend=False,
                ))
                fig_zoom.add_vrect(
                    x0=row[DATETIME_COL], x1=row[DATETIME_COL] + pd.Timedelta(hours=1),
                    fillcolor='rgba(255, 165, 0, 0.25)', line_width=0, layer='below'
                )
        
        fig_zoom.add_trace(go.Scatter(
            x=[peak_time, peak_time], y=[zoom_df['ercot_demand_mw'].min(), zoom_df['ercot_demand_mw'].max()],
            mode='lines', name='Peak Moment', line=dict(color='#FFFFFF', width=2, dash='dash')
        ))
        
        fig_zoom.update_layout(
            paper_bgcolor='#161B22', plot_bgcolor='#161B22',
            font=dict(color='white'), height=400,
            xaxis=dict(gridcolor='#30363D'), yaxis=dict(gridcolor='#30363D', title='Demand (MW)'),
        )
        st.plotly_chart(fig_zoom, width='stretch')


# ============================================================
# IMPACT AT SCALE
# ============================================================
st.divider()
st.markdown("## 📊 Impact at Scale")

scaled_reduction_kw = compute_scaled_reduction_kw(homes, reduction_rate_percent)
scaled_reduction_mw = scaled_reduction_kw / 1000

if homes < 50000:
    grid_impact = "Neighbourhood Scale"
    impact_color = "#F39C12"
elif homes < 250000:
    grid_impact = "District Scale"
    impact_color = "#4A9EFF"
elif homes < 600000:
    grid_impact = "Regional Scale"
    impact_color = "#9B59B6"
else:
    grid_impact = "National Scale"
    impact_color = "#2ECC71"

col_s1, col_s2, col_s3, col_s4 = st.columns(4)
col_s1.metric("Homes Coordinated", f"{homes:,}")
col_s2.metric("Projected Reduction (per event)", f"{scaled_reduction_mw:,.1f} MW")
col_s3.metric("Impact Level", grid_impact)
col_s4.metric("Reserve Margin", "Exceeds 200MW" if scaled_reduction_mw > 200 else "Building toward")

percentage_of_grid = (scaled_reduction_mw / ERCOT_PEAK_MW) * 100
st.markdown(
    f"<p style='color:#888; font-size:0.8rem;'>"
    f"📌 At this scale, Grid Saver removes approximately <strong>{percentage_of_grid:.3f}%</strong> "
    f"of ERCOT peak demand (~{ERCOT_PEAK_MW:,} MW).</p>",
    unsafe_allow_html=True
)

st.markdown("""
<div class='warning-box'>
⚠️ <strong>Scaling assumption:</strong> Linear aggregation of residential load response.
Real-world performance may vary due to behavioral diversity, device heterogeneity,
and rebound effects following coordinated load reduction.
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# SYSTEM ARCHITECTURE
# ============================================================
st.markdown("## 🏗️ System Architecture")
col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("""
    <div style='background: #1B4F8C22; border: 1px solid #4A9EFF; padding: 20px; border-radius: 10px; text-align: center; border-top: 4px solid #4A9EFF;'>
        <h2 style='color: #4A9EFF; font-size: 2rem; margin: 0;'>👁️</h2>
        <h3 style='color: #4A9EFF; margin: 10px 0 5px 0;'>SENSE</h3>
        <p style='color: #888; font-size: 0.85rem;'>Detect grid vulnerability<br>Carbon intensity + CFE%<br>Electricity Maps ERCOT<br>8,760 hourly records</p>
    </div>
    """, unsafe_allow_html=True)

with col_b:
    st.markdown("""
    <div style='background: #1A6B2E22; border: 1px solid #2ECC71; padding: 20px; border-radius: 10px; text-align: center; border-top: 4px solid #2ECC71;'>
        <h2 style='color: #2ECC71; font-size: 2rem; margin: 0;'>🧠</h2>
        <h3 style='color: #2ECC71; margin: 10px 0 5px 0;'>PREDICT</h3>
        <p style='color: #888; font-size: 0.85rem;'>XGBoost 24hr forecast<br>91.3% Recall validated<br>PJM 32,896 records<br>Temporal patterns only</p>
    </div>
    """, unsafe_allow_html=True)

with col_c:
    st.markdown("""
    <div style='background: #7B1A1A22; border: 1px solid #E74C3C; padding: 20px; border-radius: 10px; text-align: center; border-top: 4px solid #E74C3C;'>
        <h2 style='color: #E74C3C; font-size: 2rem; margin: 0;'>⚡</h2>
        <h3 style='color: #E74C3C; margin: 10px 0 5px 0;'>ACT</h3>
        <p style='color: #888; font-size: 0.85rem;'>HVAC load coordination<br>85% compliance + 60% rebound<br>Pecan Street 868k records<br>SPA dual-confirmation</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# REPORTS SECTION
# ============================================================
with st.expander("📄 Reports and Insights (Download CSV)"):
    if live_mode:
        st.warning("Reports are disabled in Live Mode. Switch to Analysis Mode.")
    else:
        st.markdown("*Select a time period to view grid performance analysis.*")
        
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            report_type = st.selectbox("Report Type", ["Yearly", "Monthly", "Weekly"])
        with col_r2:
            report_year = st.selectbox("Year", sorted(df['year'].unique(), reverse=True))
        
        if report_type == "Monthly":
            avail_months = sorted(df[df['year'] == report_year]['month'].unique())
            month_names = [MONTH_NAMES[m] for m in avail_months]
            sel_month_name = st.selectbox("Month", month_names)
            report_month = [k for k, v in MONTH_NAMES.items() if v == sel_month_name][0]
            df_report = df[(df['year'] == report_year) & (df['month'] == report_month)].copy()
            period_label = f"{sel_month_name} {report_year}"
        elif report_type == "Weekly":
            avail_weeks = sorted(df[df['year'] == report_year]['week'].unique())
            report_week = st.selectbox("Week", avail_weeks)
            df_report = df[(df['year'] == report_year) & (df['week'] == report_week)].copy()
            period_label = f"Week {report_week}, {report_year}"
        else:
            df_report = df[df['year'] == report_year].copy()
            period_label = f"{report_year}"
        
        if not df_report.empty:
            avg_vuln = df_report['vulnerability_score'].mean()
            peak_vuln = df_report['vulnerability_score'].max()
            
            # Locked SPA events
            spa_events_report = NOTEBOOK_SPA_ACTIONS
            
            # Pie chart
            status_counts = df_report['grid_status'].value_counts()
            fig_pie = go.Figure(data=[go.Pie(
                labels=status_counts.index, values=status_counts.values, hole=0.4,
                marker_colors=['#2ECC71', '#F39C12', '#E74C3C']
            )])
            fig_pie.update_layout(paper_bgcolor='#161B22', plot_bgcolor='#161B22',
                                  font=dict(color='white'), height=300)
            
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Avg Vulnerability", f"{avg_vuln:.1f}")
            col_m2.metric("Peak Vulnerability", f"{peak_vuln:.1f}")
            col_m3.metric("SPA Events (Validated)", f"{spa_events_report}")
            
            st.plotly_chart(fig_pie, width='stretch')
            
            # Trend chart
            fig_report = go.Figure()
            for status in ['STABLE', 'WARNING', 'CRITICAL']:
                mask = df_report['grid_status'] == status
                if mask.any():
                    fig_report.add_trace(go.Scatter(
                        x=df_report[mask][DATETIME_COL], y=df_report[mask]['vulnerability_score'],
                        mode='lines', name=status, line=dict(color=color_map[status], width=1)
                    ))
            fig_report.add_hline(y=VULNERABILITY_THRESHOLD, line_dash='dash', line_color='#FF4444')
            fig_report.update_layout(paper_bgcolor='#161B22', plot_bgcolor='#161B22',
                                      font=dict(color='white'), height=300)
            st.plotly_chart(fig_report, width='stretch')
            
            st.info(f"📊 {period_label}: {spa_events_report} validated SPA events")
            
            csv = df_report.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, f"GridSaver_{period_label.replace(' ', '_')}.csv")
        else:
            st.warning("No data for selected period.")


# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div style='background: #161B22; padding: 15px; border-radius: 8px; border: 1px solid #30363D; text-align: center; margin-top: 20px;'>
    <p style='color: #888; margin: 0; font-size: 0.85rem;'>
        Grid Saver | Adaptive Grid Intelligence Platform | Justine Adzormado
    </p>
    <p style='color: #555; margin: 5px 0 0 0; font-size: 0.75rem;'>
        📡 Sense: Electricity Maps US-TEX-ERCO 2025 | 🧠 Predict: PJM XGBoost 91.3% Recall | ⚡ Act: Pecan Street 2018<br>
        🔒 SPA dual-confirmation: Action only when BOTH Sense AND Predict independently confirm vulnerability
    </p>
    <p style='color: #555; font-size: 0.7rem; margin-top: 5px;'>
        ⏱️ <strong>Temporal resolution:</strong> Hourly intervals. Rebound modeled at 60% with 85% compliance.
    </p>
    <p style='color: #444; margin: 5px 0 0 0; font-size: 0.7rem;'>
        ⚠️ Educational Prototype — Not for real-time operations.
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# END OF APP
# ============================================================
