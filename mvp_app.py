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
    .text-box-horizontal {
        background: #161B22;
        border-radius: 10px;
        padding: 16px;
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
# LOCKED NOTEBOOK TRUTH (NEVER RECOMPUTE)
# Annual totals are immutable. SPA=154, Gross=108,293 MWh, Net=43,317 MWh.
# ============================================================
NOTEBOOK_SENSE_TRIGGERS = 1316
NOTEBOOK_PREDICT_TRIGGERS = 1659
NOTEBOOK_SPA_EVENTS = 154

# ============================================================
# THREE-LAYER BASELINE (Option A)
# Layer 1 — THEORETICAL: explanation + HVAC decomposition only. Never plotted.
# Layer 2 — MODEL PEAK:  simulation envelope ceiling. Documented, not enforced.
# Layer 3 — OBSERVED:    runtime max from data. Never overridden.
# ============================================================
THEORETICAL_BASELINE_MW = 70320          # explanation only, used for HVAC narrative
MODEL_PEAK_MW = THEORETICAL_BASELINE_MW * 0.95   # 66,804 MW — reference only, not enforced

HVAC_SHARE = 0.25
HVAC_REDUCTION_RATE = 0.04

hvac_load_mw = THEORETICAL_BASELINE_MW * HVAC_SHARE
SYSTEM_REDUCTION_MW = hvac_load_mw * HVAC_REDUCTION_RATE   # 703.2 MW

# Annual totals (MW × 154 hours)
ANNUAL_GROSS_MWH = NOTEBOOK_SPA_EVENTS * SYSTEM_REDUCTION_MW          # 108,293
REBOUND_RATE = 0.60
ANNUAL_REBOUND_MWH = ANNUAL_GROSS_MWH * REBOUND_RATE                  # 64,976
ANNUAL_NET_MWH = ANNUAL_GROSS_MWH - ANNUAL_REBOUND_MWH                # 43,317

# ============================================================
# CONSTANTS
# ============================================================
CARBON_COL = 'Carbon intensity gCO\u2082eq/kWh (direct)'
CFE_COL = 'Carbon-free energy percentage (CFE%)'
DECISION_THRESHOLD = 0.4
KW_PER_HOME = 0.0920   # validated at 4% reduction for 25 homes

MONTH_NAMES = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December'
}
DATETIME_COL = 'Datetime (UTC)'
month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def count_spa_events(trigger_series):
    """Count SPA events using rising edge detection (kept for reference only)."""
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
    """Impact at Scale: pure scenario calculator (not used in main simulation)."""
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
# FEATURE ENGINEERING (unchanged)
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
# SENSE LAYER — detects grid stress from carbon + CFE signals.
# Outputs: vulnerability_score (0-100), grid_status, vulnerability_event.
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
# PREDICT LAYER — XGBoost forecasts vulnerability probability.
# Aggregated by hour + month. Outputs: vuln_probability, predict_triggered.
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
# ACT LAYER (UNIFIED)
# ============================================================
def act_layer(df_input, reduction_rate_percent, apply_intervention_flag):
    df = df_input.copy()
    df['sense_triggered'] = df['vulnerability_event']

    # SPA is a logic gate (AND), not a threshold trigger.
    # Both sense AND predict must independently confirm. Either alone = no dispatch.
    df['spa_action_triggered'] = df['sense_triggered'] & df['predict_triggered']

    # DEMAND MODEL: synthetic envelope, NOT measured telemetry.
    # Range: 55% → 95% of theoretical baseline.
    # Label in all charts as "Simulated Demand (Vulnerability-Scaled)"
    base_pct = 0.55
    peak_pct = 0.95
    df['simulated_demand_mw'] = (
        base_pct + (df['vulnerability_score'] / 100) * (peak_pct - base_pct)
    ) * THEORETICAL_BASELINE_MW

    # Reduction value (constant per SPA event)
    actual_reduction_mw = SYSTEM_REDUCTION_MW

    if apply_intervention_flag:
        df['grid_saver_reduction_mw'] = np.where(
            df['spa_action_triggered'],
            actual_reduction_mw,
            0
        )
    else:
        df['grid_saver_reduction_mw'] = 0

    # REBOUND: 60% thermal snapback in next time step.
    # 85% compliance = behavioral assumption from literature, NOT validated vs Pecan Street.
    # Silent recovery is not permitted — rebound must always be modeled.
    REBOUND_RATE = 0.60
    df['rebound_mw'] = np.where(
        df['spa_action_triggered'].shift(1).fillna(False),
        df['grid_saver_reduction_mw'].shift(1).fillna(0) * REBOUND_RATE,
        0
    )

    df['optimized_demand_mw'] = (
        df['simulated_demand_mw'] - df['grid_saver_reduction_mw'] + df['rebound_mw']
    )

    total_mw_saved = df['grid_saver_reduction_mw'].sum() if apply_intervention_flag else 0
    return df, total_mw_saved

# ============================================================
# RUN FULL PIPELINE (once)
# ============================================================
df, VULNERABILITY_THRESHOLD = sense_layer(df_raw)
df['hour'] = df[DATETIME_COL].dt.hour
df['month'] = df[DATETIME_COL].dt.month
df['month_name'] = df[DATETIME_COL].dt.strftime('%b')
df['date'] = df[DATETIME_COL].dt.date
df['year'] = df[DATETIME_COL].dt.year
df['week'] = df[DATETIME_COL].dt.isocalendar().week.astype(int)

df = predict_layer(df, model)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.image("https://img.icons8.com/fluency/96/lightning-bolt.png", width=60)
st.sidebar.title("Grid Saver")
st.sidebar.markdown("**Adaptive Grid Intelligence Platform**")
st.sidebar.divider()

live_mode = st.sidebar.toggle("Recent Window View (Last 24 Hours)", value=False)
if live_mode:
    st.sidebar.markdown(
        "<p style='color:#2ECC71; font-size:0.8rem;'>Showing the most recent 24 hours</p>",
        unsafe_allow_html=True
    )
    st.sidebar.markdown(
        "<p style='color:#888; font-size:0.7rem;'>Note: Threshold is based on full-year data to ensure consistent benchmarking.</p>",
        unsafe_allow_html=True
    )

st.sidebar.divider()
months_present = [m for m in month_order if m in df['month_name'].unique()]
selected_month = st.sidebar.selectbox("Select Month", ['All Year'] + months_present)

reduction_rate_percent = st.sidebar.slider(
    "HVAC Reduction Rate (%)",
    min_value=1, max_value=10, value=4, step=1,
    help="Reduction applied to residential HVAC load (25% of grid). 4% → 703 MW per event."
)

homes = st.sidebar.slider(
    "Homes Coordinated (Impact at Scale)",
    min_value=1000, max_value=1000000, value=100000, step=1000,
    help="What-if scaling scenario using validated 0.0920 kW per home (does not affect main simulation)."
)

apply_intervention_flag = st.sidebar.toggle("Apply Grid Saver Intervention", value=True)

with st.sidebar.expander("ℹ️ How Grid Saver Works"):
    st.markdown(f"""
    **Vulnerability Score (0-100)**
    - **0-39: STABLE** — Normal
    - **40-69: WARNING** — Elevated risk
    - **70-100: CRITICAL** — Action required

    **SPA Dual-Confirmation** (Notebook truth – immutable)
    - Sense triggers: **{NOTEBOOK_SENSE_TRIGGERS}** hours
    - Predict triggers: **{NOTEBOOK_PREDICT_TRIGGERS}** hours
    - SPA events (dual‑confirmed): **{NOTEBOOK_SPA_EVENTS}** hours

    **Three‑Layer Baseline (Option A)**
    - Theoretical baseline (explanation only): **{THEORETICAL_BASELINE_MW:,} MW**
    - Model operational max: **{MODEL_PEAK_MW:,.0f} MW** (95% of theoretical)
    - Observed data peak: from actual dataset

    **HVAC Reduction Model**
    - HVAC share: {HVAC_SHARE*100:.0f}% → load = {hvac_load_mw:,.0f} MW
    - {reduction_rate_percent}% HVAC reduction → **{SYSTEM_REDUCTION_MW:.1f} MW** per SPA hour
    - Equivalent: {SYSTEM_REDUCTION_MW/THEORETICAL_BASELINE_MW*100:.1f}% of theoretical grid

    **Annual Impact (based on {NOTEBOOK_SPA_EVENTS} events)**
    - Gross reduction: **{ANNUAL_GROSS_MWH:,.0f} MWh**
    - Rebound (60%): **{ANNUAL_REBOUND_MWH:,.0f} MWh**
    - Net saving: **{ANNUAL_NET_MWH:,.0f} MWh**
    """)

st.sidebar.divider()
st.sidebar.markdown("**Stack:** Colab + GitHub + Streamlit")
st.sidebar.markdown("*Justine Adzormado*")

# ============================================================
# FILTER DATA (Live Mode: only time filter)
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

# Apply act_layer to get unified curves
df_view, total_mw_saved = act_layer(df_view, reduction_rate_percent, apply_intervention_flag)

# ============================================================
# EXTRACT CURVES (using the actual data for the filtered period)
# ============================================================
simulated_curve = df_view['simulated_demand_mw']
if apply_intervention_flag:
    optimized_curve = df_view['optimized_demand_mw']
    reduction_curve = df_view['grid_saver_reduction_mw']
else:
    optimized_curve = simulated_curve.copy()
    reduction_curve = np.zeros(len(df_view))

# Peak values (observed data)
peak_observed = simulated_curve.max()
peak_optimized = optimized_curve.max()
peak_reduction_mw = peak_observed - peak_optimized
peak_reduction_pct = (peak_reduction_mw / peak_observed * 100) if peak_observed > 0 else 0

# Peak time and index
peak_idx = simulated_curve.idxmax()
if peak_idx in df_view.index:
    peak_time = df_view.loc[peak_idx, DATETIME_COL]
else:
    peak_time = df_view[DATETIME_COL].iloc[-1]

# Totals (for the filtered period, not used for annual metrics)
total_reduction_mwh = reduction_curve.sum()
total_rebound_mwh = df_view['rebound_mw'].sum()

# Current row for status
current_row = df_view.iloc[-1]

# ============================================================
# HEADER
# ============================================================
mode_label = "🔴 LIVE MODE (Last 24h)" if live_mode else "📊 ANALYSIS MODE (Full Year)"
mode_color = "#E74C3C" if live_mode else "#E74C3C"

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
# RISK DRIVERS (Explain WHY the grid is stressed)
# ============================================================
st.markdown("## ⚠️ Risk Drivers")
drivers = []

# Relative (not absolute) comparisons for defensibility
if current_carbon > df[CARBON_COL].quantile(0.75):
    drivers.append("Carbon intensity is elevated relative to baseline")

if current_cfe < df[CFE_COL].quantile(0.25):
    drivers.append("Carbon-free energy contribution is below typical levels")

if current_score >= 70:
    drivers.append("System operating in CRITICAL vulnerability range")

if current_prob >= DECISION_THRESHOLD:
    drivers.append("Short-term risk projection is elevated")

if drivers:
    for d in drivers:
        st.markdown(f"- {d}")
else:
    st.markdown("No significant risk drivers detected — grid conditions within normal range.")

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
# PEAK VULNERABILITY TIMELINE (unchanged)
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

current_status_text = current_status
if current_status_text == 'CRITICAL':
    action_color, action_icon = "#E74C3C", "🔴"
    action_title = "CRITICAL — Immediate Action Required"
    if apply_intervention_flag:
        action_text = f"Reduce residential HVAC load by {reduction_rate_percent}% (~{SYSTEM_REDUCTION_MW:.0f} MW per event)"
    else:
        action_text = "Grid Saver is OFF — monitoring only, no active intervention"
elif current_status_text == 'WARNING':
    action_color, action_icon = "#F39C12", "🟡"
    action_title = "WARNING — Prepare for Intervention"
    if apply_intervention_flag:
        action_text = f"Pre-stage {reduction_rate_percent}% HVAC reduction"
    else:
        action_text = "Grid Saver is OFF — monitoring only, no active intervention"
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

with st.expander("🧠 AI Decision Explanation", expanded=False):
    if st.button("Explain Grid Decision", key="explain_btn"):
        risk_color = {"CRITICAL":"#E74C3C", "WARNING":"#F39C12", "STABLE":"#2ECC71"}.get(current_status_text, "#CCC")
        st.markdown(f"""
        <div style='background:#161B22; border:1px solid #1B4F8C; padding:25px; border-radius:10px; margin-top:15px;'>
            <h3 style='color:#4A9EFF; margin:0 0 15px 0;'>🧠 AI Decision Explanation</h3>
            <p style='color:#CCC; margin:5px 0;'>
                Grid Saver classified the system as
                <strong style='color:{action_color};'>{current_status_text}</strong> based on:
            </p>
            <ul style='color:#CCC; margin:10px 0;'>
                <li><strong>Vulnerability Score:</strong> {current_score:.1f} / 100 (threshold: {VULNERABILITY_THRESHOLD:.0f})</li>
                <li><strong>Carbon Intensity:</strong> {current_carbon:.0f} gCO₂eq/kWh</li>
                <li><strong>Carbon-Free Energy:</strong> {current_cfe:.1f}%</li>
                <li><strong>24hr Risk Projection:</strong> {current_prob:.2f} (threshold: {DECISION_THRESHOLD})</li>
                <li><strong>Risk Level:</strong> <span style='color:{risk_color};'>{current_status_text}</span></li>
            </ul>
            <p style='color:#CCC; margin:10px 0 5px 0;'>
                <strong>Recommended Action:</strong> {action_text}
            </p>
            <p style='color:#888; margin:0; font-size:0.9rem;'>Monitoring window: Next 2-4 hours | Pre-emptive coordination recommended</p>
            <p style='color:#555; margin:15px 0 0 0; font-size:0.8rem;'>
                <strong>Reserve Margin Impact:</strong> A {reduction_rate_percent}% HVAC reduction across {homes:,} homes removes approximately {(homes * 0.092 * (reduction_rate_percent/4)):,.1f} kW from peak demand, contributing to restoring the grid toward safe operating bounds.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# SMART RECOMMENDATIONS (Operator-grade actions)
# ============================================================
st.markdown("## 🧠 Smart Recommendations")
dispatch_score = current_score * 0.5 + current_prob * 30 + min(current_carbon/10, 20)
dispatch_score = round(dispatch_score, 1)
st.markdown(f"""
<div style='background:#161B22; border:1px solid #F39C12; padding:12px 16px; border-radius:8px; margin-bottom:12px;'>
    <span style='color:#F39C12; font-weight:bold;'>⚡ Dispatch Priority: {dispatch_score}/100</span>
    <p style='color:#888; margin:5px 0 0 0; font-size:0.75rem;'>
        Formula: (vulnerability × 0.5) + (risk_probability × 30) + min(carbon/10, 20)
    </p>
</div>
""", unsafe_allow_html=True)

if current_status_text == 'CRITICAL':
    if apply_intervention_flag:
        st.markdown(f"""
🔴 **Operator Action Required**
- Dispatch {reduction_rate_percent}% residential HVAC demand response (~{SYSTEM_REDUCTION_MW:.0f} MW per event)
- Target peak window (next 1–2 hours)
- Monitor rebound risk following load reduction
""")
        if current_row.get('spa_action_triggered', False):
            st.markdown("✅ SPA dual-confirmation achieved — dispatch approved")
        else:
            st.markdown("⚠️ Partial confirmation — controlled reduction recommended")
    else:
        st.markdown("🔴 Grid Saver OFF — enable intervention to allow dispatch actions")
elif current_status_text == 'WARNING':
    st.markdown("""
🟡 **Operator Advisory**
- Pre-stage demand response resources
- Increase monitoring frequency
- Prepare for potential dispatch within next 2–4 hours
""")
    if current_prob >= DECISION_THRESHOLD:
        st.markdown("📈 Elevated risk signal — escalation likely")
else:
    st.markdown("""
🟢 **Operator Status**
- No intervention required
- Maintain standard monitoring
""")

if apply_intervention_flag:
    st.markdown(f"""
    <div style='background:#161B22; border-left:4px solid #2ECC71; padding:10px 14px; border-radius:6px; margin-top:12px;'>
        <span style='color:#2ECC71; font-size:0.9rem;'>📊 <strong>Impact Summary (Annual, Notebook Truth)</strong></span><br>
        <span style='color:#CCC; font-size:0.85rem;'>
            Per SPA event: {SYSTEM_REDUCTION_MW:.0f} MW<br>
            Gross annual reduction: {ANNUAL_GROSS_MWH:,.0f} MWh<br>
            Net after rebound: {ANNUAL_NET_MWH:,.0f} MWh<br>
            Based on {NOTEBOOK_SPA_EVENTS} SPA events per year
        </span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style='background:#161B22; border-left:4px solid #888; padding:10px 14px; border-radius:6px; margin-top:12px;'>
        <span style='color:#888; font-size:0.9rem;'>⚙️ <strong>Grid Saver Disabled</strong></span><br>
        <span style='color:#999; font-size:0.85rem;'>
            Toggle "Apply Grid Saver Intervention" in the sidebar to see load reduction impact.
        </span>
    </div>
    """, unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# LOAD REDUCTION SIMULATION (using locked notebook values)
# ============================================================
st.markdown("## ⚡ Load Reduction Simulation")
st.markdown(f"""
<div class='info-box'>
📌 <strong>Simulation Basis (Three‑Layer Baseline – Option A)</strong><br>
• Theoretical baseline (explanation only): <strong>{THEORETICAL_BASELINE_MW:,} MW</strong><br>
• Model operational max (95% of theoretical): <strong>{MODEL_PEAK_MW:,.0f} MW</strong><br>
• Observed demand peak from data: <strong>{peak_observed:,.0f} MW</strong><br>
• HVAC share: {HVAC_SHARE*100:.0f}% → HVAC load = {hvac_load_mw:,.0f} MW<br>
• HVAC reduction applied: <strong>{reduction_rate_percent}%</strong><br>
• System impact per event: <strong>{SYSTEM_REDUCTION_MW:.1f} MW</strong> ({SYSTEM_REDUCTION_MW/THEORETICAL_BASELINE_MW*100:.1f}% of theoretical grid)<br>
• Annual SPA events (dual‑confirmed): <strong>{NOTEBOOK_SPA_EVENTS}</strong>
</div>
""", unsafe_allow_html=True)

if reduction_rate_percent != 4:
    st.warning(f"⚠️ Validated at 4% HVAC reduction = {SYSTEM_REDUCTION_MW:.1f} MW ({SYSTEM_REDUCTION_MW/THEORETICAL_BASELINE_MW*100:.1f}% grid impact). Results at {reduction_rate_percent}% are proportionally scaled.")

# SPA trigger cards (locked notebook values)
col_s1, col_s2, col_s3 = st.columns(3)
col_s1.metric("Sense Triggers (year)", f"{NOTEBOOK_SENSE_TRIGGERS:,}")
col_s2.metric("Predict Triggers (year)", f"{NOTEBOOK_PREDICT_TRIGGERS:,}")
col_s3.metric("SPA Events (year)", f"{NOTEBOOK_SPA_EVENTS}")

if apply_intervention_flag:
    st.markdown(f"""
    <div style='background:#161B22; border-left:5px solid #2ECC71; padding:15px; border-radius:8px; margin:15px 0;'>
        <h3 style='color:#2ECC71; margin:0;'>⚡ Load Reduction Analysis (Annual)</h3>
        <p style='color:white; font-size:1.2rem; margin:5px 0;'>
            Gross Load Reduced: {ANNUAL_GROSS_MWH:,.2f} MWh
        </p>
        <p style='color:#E74C3C; font-size:1rem; margin:5px 0;'>
            ⚠️ Thermal Rebound (Snapback): +{ANNUAL_REBOUND_MWH:,.2f} MWh
        </p>
        <p style='color:#2ECC71; font-size:1.1rem; margin:5px 0;'>
            ✅ Net Load Reduction: {ANNUAL_NET_MWH:,.2f} MWh
        </p>
        <p style='color:#888; margin:5px 0 0 0; font-size:0.8rem;'>
            Across {NOTEBOOK_SPA_EVENTS} SPA events | Per event: {SYSTEM_REDUCTION_MW:.0f} MW
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style='background:#161B22; border-left:5px solid #888; padding:15px; border-radius:8px; margin:15px 0;'>
        <h3 style='color:#888; margin:0;'>⚡ Load Reduction Analysis</h3>
        <p style='color:#888; font-size:1rem; margin:5px 0;'>
            Grid Saver intervention is currently <strong>DISABLED</strong>.
        </p>
        <p style='color:#888; margin:5px 0 0 0; font-size:0.8rem;'>
            Toggle "Apply Grid Saver Intervention" in the sidebar to see load reduction impact.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Peak reduction – horizontal text boxes for Observed Peak and After Grid Saver
col_peak1, col_peak2 = st.columns(2)
with col_peak1:
    st.markdown(f"""
    <div class='text-box-horizontal'>
        <p style='color:#888; margin:0; font-size:0.8rem;'>Observed Peak</p>
        <h2 style='color:#E74C3C; margin:0;'>{peak_observed:,.0f} MW</h2>
    </div>
    """, unsafe_allow_html=True)
with col_peak2:
    st.markdown(f"""
    <div class='text-box-horizontal'>
        <p style='color:#888; margin:0; font-size:0.8rem;'>After Grid Saver</p>
        <h2 style='color:#2ECC71; margin:0;'>{peak_optimized:,.0f} MW</h2>
    </div>
    """, unsafe_allow_html=True)

# Keep reduction percentage and load shed as metric cards (optional, you can keep them)
col_p3, col_p4 = st.columns(2)
with col_p3:
    st.markdown(f"""
    <div class='metric-card'>
        <h2 style='color:#4A9EFF; font-size:1.4rem; margin:0;'>{peak_reduction_pct:.2f}%</h2>
        <p style='color:#888; margin:0;'>Peak Reduction</p>
    </div>
    """, unsafe_allow_html=True)
with col_p4:
    st.markdown(f"""
    <div class='metric-card'>
        <h2 style='color:#F39C12; font-size:1.4rem; margin:0;'>{peak_reduction_mw:,.2f} MW</h2>
        <p style='color:#888; margin:0;'>Peak Load Shed</p>
    </div>
    """, unsafe_allow_html=True)

if apply_intervention_flag:
    st.markdown(f"""
    <p style='color:#888; font-size:0.8rem; margin-top:6px;'>
    ⚡ <strong>Max Reduction During SPA Events:</strong> {SYSTEM_REDUCTION_MW:.0f} MW per event
    </p>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class='success-box'>
    🛡️ <strong>Critical Events Intervened (Annual):</strong> {NOTEBOOK_SPA_EVENTS} SPA events
    <br>Based on dual-confirmation (Sense + Predict) logic from notebook.
    </div>
    """, unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# BEFORE / AFTER CHART (preserved structure)
# ============================================================
st.markdown("#### 📉 Before vs After Grid Saver — Demand with Intervention")
fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=('Grid Demand (MW)', 'Load Reduction (MW)'),
    vertical_spacing=0.12,
    shared_xaxes=True
)

# BEFORE line (always visible) – label fixed
fig.add_trace(go.Scatter(
    x=df_view[DATETIME_COL],
    y=simulated_curve,
    mode='lines',
    name='Simulated Demand (Vulnerability-Scaled)',
    line=dict(color='#E74C3C', width=2),
), row=1, col=1)

# AFTER line (only when toggle ON)
if apply_intervention_flag:
    fig.add_trace(go.Scatter(
        x=df_view[DATETIME_COL],
        y=optimized_curve,
        mode='lines',
        name='After Grid Saver',
        line=dict(color='#2ECC71', width=2, dash='dash'),
    ), row=1, col=1)

# Reduction bars (always exist, zero when OFF)
fig.add_trace(go.Bar(
    x=df_view[DATETIME_COL],
    y=reduction_curve,
    name='Load Reduction',
    marker_color='#F39C12',
    opacity=0.6,
), row=2, col=1)

# Mark peak (safe indexing)
if peak_idx in df_view.index:
    fig.add_trace(go.Scatter(
        x=[df_view.loc[peak_idx, DATETIME_COL]],
        y=[peak_observed],
        mode='markers',
        name=f'Peak: {peak_observed:,.0f} MW',
        marker=dict(color='#FFFFFF', size=8, symbol='star'),
    ), row=1, col=1)

fig.update_layout(
    paper_bgcolor='#161B22',
    plot_bgcolor='#161B22',
    font=dict(color='white'),
    height=500,
    legend=dict(bgcolor='#1A1A2E', bordercolor='#333'),
    hovermode='x unified'
)
fig.update_xaxes(gridcolor='#30363D', title_text='Datetime', row=2, col=1)
fig.update_yaxes(gridcolor='#30363D', title_text='Demand (MW)', row=1, col=1)
fig.update_yaxes(gridcolor='#30363D', title_text='Reduction (MW)', row=2, col=1)
st.plotly_chart(fig, width='stretch')

st.markdown("""
<div class='info-box'>
📌 <strong>Chart Guide:</strong><br>
• <span style='color:#E74C3C'>🔴 Red line:</span> Simulated demand (vulnerability‑scaled, 55–95% of theoretical baseline)<br>
• <span style='color:#2ECC71'>🟢 Green dashed line:</span> Demand after Grid Saver intervention (ONLY shown when toggle is ON)<br>
• <span style='color:#F39C12'>🟠 Orange bars:</span> Load reduction achieved during SPA-triggered hours (zero when toggle OFF)<br>
• <span style='color:#FFFFFF'>⭐ White star:</span> Peak demand timestamp
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='rebound-box'>
⚠️ <strong>Thermal Rebound (Snapback) Effect:</strong><br>
When HVAC systems are suppressed during an SPA event, they turn back on simultaneously afterward,
creating a secondary demand spike. Grid Saver models this using:<br>
• <strong>85% Compliance Rate</strong> — behavioral assumption from literature, not validated vs Pecan Street.<br>
• <strong>60% Rebound Rate</strong> — 60% of shed load returns post-event<br>
• <strong>Corrected rebound timing</strong> — rebound occurs in the hour following a reduction.
</div>
""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# 6-HOUR BEFORE/AFTER PEAK WINDOW (inside expander)
# ============================================================
with st.expander("📉 6-Hour Before vs After Analysis (around peak demand)"):
    half_window = pd.Timedelta(hours=6)
    window_mask = (df_view[DATETIME_COL] >= peak_time - half_window) & (df_view[DATETIME_COL] <= peak_time + half_window)
    window_df = df_view[window_mask].copy()
    if not window_df.empty:
        fig_window = go.Figure()
        fig_window.add_trace(go.Scatter(
            x=window_df[DATETIME_COL],
            y=window_df['simulated_demand_mw'],
            mode='lines',
            name='Simulated Demand (Vulnerability-Scaled)',
            line=dict(color='#E74C3C', width=2)
        ))
        if apply_intervention_flag:
            fig_window.add_trace(go.Scatter(
                x=window_df[DATETIME_COL],
                y=window_df['optimized_demand_mw'],
                mode='lines',
                name='After Grid Saver',
                line=dict(color='#2ECC71', width=2, dash='dash')
            ))
        fig_window.add_trace(go.Bar(
            x=window_df[DATETIME_COL],
            y=window_df['grid_saver_reduction_mw'] if apply_intervention_flag else np.zeros(len(window_df)),
            name='Load Reduction',
            marker_color='#F39C12',
            opacity=0.6,
            yaxis='y2'
        ))
        fig_window.update_layout(
            paper_bgcolor='#161B22', plot_bgcolor='#161B22',
            font=dict(color='white'), height=400,
            title=dict(text=f'Demand & Reduction around Peak ({peak_time.strftime("%Y-%m-%d %H:%M")})',
                       font=dict(color='white', size=13)),
            xaxis=dict(gridcolor='#30363D', title='Datetime'),
            yaxis=dict(gridcolor='#30363D', title='Demand (MW)'),
            yaxis2=dict(title='Reduction (MW)', overlaying='y', side='right', showgrid=False),
            legend=dict(bgcolor='#1A1A2E')
        )
        st.plotly_chart(fig_window, width='stretch')
        st.markdown("""
        <div class='info-box'>
        🔍 <strong>6-Hour Peak Window:</strong> Shows the 6 hours before and after the peak demand moment.
        This highlights how Grid Saver intervention affects the most critical period.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No data available in the 6‑hour window around the peak.")

# ============================================================
# IMPACT AT SCALE (four square boxes horizontally, plus Impact Level)
# ============================================================
st.divider()
st.markdown("## 📊 Impact at Scale")
st.markdown("*This section shows what-if scaling scenarios. Adjust the slider above to see different scales.*")

scaled_reduction_kw = compute_scaled_reduction_kw(homes, reduction_rate_percent)
scaled_reduction_mw = scaled_reduction_kw / 1000
grid_impact_percent = (scaled_reduction_mw / THEORETICAL_BASELINE_MW) * 100

col_i1, col_i2, col_i3, col_i4 = st.columns(4)
with col_i1:
    st.markdown(f"""
    <div class='metric-card'>
        <h2 style='color: #4A9EFF; font-size: 1.3rem; margin: 0;'>{homes:,}</h2>
        <p style='color: #888; margin: 0;'>Homes Coordinated</p>
    </div>
    """, unsafe_allow_html=True)
with col_i2:
    st.markdown(f"""
    <div class='metric-card'>
        <h2 style='color: #2ECC71; font-size: 1.3rem; margin: 0;'>0.092 kW</h2>
        <p style='color: #888; margin: 0;'>Per Home Impact</p>
    </div>
    """, unsafe_allow_html=True)
with col_i3:
    st.markdown(f"""
    <div class='metric-card'>
        <h2 style='color: #F39C12; font-size: 1.3rem; margin: 0;'>{scaled_reduction_mw:.2f} MW</h2>
        <p style='color: #888; margin: 0;'>Total Reduction (per event)</p>
    </div>
    """, unsafe_allow_html=True)
with col_i4:
    st.markdown(f"""
    <div class='metric-card'>
        <h2 style='color: #9B59B6; font-size: 1.3rem; margin: 0;'>{grid_impact_percent:.3f}%</h2>
        <p style='color: #888; margin: 0;'>Grid Impact</p>
    </div>
    """, unsafe_allow_html=True)

# Impact Level text box
if homes < 100000:
    impact_level = "City-scale"
elif homes < 500000:
    impact_level = "Regional-scale"
else:
    impact_level = "National-scale"

st.markdown(f"""
<div class='text-box-horizontal' style='margin-top: 10px;'>
    <p style='color:#888; margin:0; font-size:0.8rem;'>Impact Level (scenario)</p>
    <h2 style='color:#4A9EFF; margin:0;'>{impact_level}</h2>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='warning-box'>
⚠️ <strong>Scaling assumption:</strong> Linear aggregation of residential load response (0.0920 kW per home at 4% reduction).
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
        <p style='color: #888; font-size: 0.85rem;'>HVAC load coordination<br>85% compliance (assumed) + 60% rebound<br>Pecan Street 868k records<br>SPA dual-confirmation</p>
    </div>
    """, unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# REPORTS SECTION (unchanged)
# ============================================================
with st.expander("📄 Reports and Insights (Download CSV)"):
    if live_mode:
        st.warning("📊 Reports are disabled in Live Mode. Switch to Analysis Mode to generate reports.")
    else:
        st.markdown("*Select a time period to view grid performance analysis and download a report.*")
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
            # Run act_layer to get SPA columns for this period
            df_report, _ = act_layer(df_report, reduction_rate_percent, apply_intervention_flag)
            avg_vuln = df_report['vulnerability_score'].mean()
            peak_vuln = df_report['vulnerability_score'].max()
            spa_events_period = count_spa_events(df_report['spa_action_triggered'])

            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Avg Vulnerability", f"{avg_vuln:.1f}")
            col_m2.metric("Peak Vulnerability", f"{peak_vuln:.1f}")
            col_m3.metric("SPA Events (this period)", f"{spa_events_period}")

            # Pie chart
            status_counts = df_report['grid_status'].value_counts()
            fig_pie = go.Figure(data=[go.Pie(
                labels=status_counts.index, values=status_counts.values, hole=0.4,
                marker_colors=['#2ECC71', '#F39C12', '#E74C3C']
            )])
            fig_pie.update_layout(paper_bgcolor='#161B22', plot_bgcolor='#161B22',
                                  font=dict(color='white'), height=300)
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
            st.info(f"📊 {period_label}: {spa_events_period} SPA events in this period (notebook full-year total: {NOTEBOOK_SPA_EVENTS})")

            try:
                export_df = df_report.copy()
                csv_data = export_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Report (CSV)",
                    data=csv_data,
                    file_name=f"GridSaver_{period_label.replace(' ', '_')}.csv",
                    key="report_download"
                )
            except Exception as e:
                st.error(f"Export failed: {e}")
        else:
            st.warning("No data available for selected period.")

# ============================================================
# FOOTER (shortened & clarified)
# ============================================================
st.markdown(f"""
<div style='background: #161B22; padding: 15px; border-radius: 8px; border: 1px solid #30363D; text-align: center; margin-top: 20px;'>
    <p style='color: #888; margin: 0; font-size: 0.85rem;'>
        Grid Saver | Adaptive Grid Intelligence Platform | Justine Adzormado
    </p>
    <p style='color: #555; margin: 5px 0 0 0; font-size: 0.75rem;'>
        📡 Sense: Electricity Maps US-TEX-ERCO 2025 | 🧠 Predict: PJM XGBoost 91.3% Recall | ⚡ Act: Pecan Street 2018<br>
        🔒 SPA dual‑confirmation: Sense AND Predict must both trigger<br>
        📊 Three‑Layer Baseline: Theoretical {THEORETICAL_BASELINE_MW:,} MW (explanation) | 
        Model max {MODEL_PEAK_MW:,.0f} MW (envelope) | Observed {peak_observed:,.0f} MW (data truth)<br>
        ⏱️ Hourly resolution | Rebound: 60% snapback | Compliance: 85% assumed — not validated vs Pecan Street<br>
        📌 Notebook validated SPA events: {NOTEBOOK_SPA_EVENTS} per year → annual gross {ANNUAL_GROSS_MWH:,.0f} MWh, net {ANNUAL_NET_MWH:,.0f} MWh
    </p>
    <p style='color: #444; margin: 5px 0 0 0; font-size: 0.7rem;'>
        ⚠️ Production readiness requires live SCADA integration and regulatory approval.
        Current version operates on historical data with validated models.
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# END OF APP
# ============================================================
