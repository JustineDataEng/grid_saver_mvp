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
DATETIME_COL  = 'Datetime (UTC)'   
ERCOT_PEAK_MW = 70000              
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
    # Fix pandas datetime processing to be instantly fast
    df['Datetime (UTC)'] = pd.to_datetime(df['Datetime (UTC)'], utc=True, format='mixed')
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
# SENSE LAYER
# ============================================================
def sense_layer(df_input):
    df_s = df_input.copy()
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
def act_layer(df_input, reduction_rate, homes):
    df_a = df_input.copy()
    df_a['sense_triggered']      = df_a['vulnerability_event']
    df_a['spa_action_triggered'] = (
        df_a['sense_triggered'] & df_a['predict_triggered']
    )
    scaled_kw_per_home = KW_PER_HOME * (reduction_rate / 4)
    df_a['grid_saver_reduction_kw'] = np.where(
        df_a['spa_action_triggered'],
        homes * scaled_kw_per_home,
        0
    )
    df_a['reduction_mw'] = df_a['grid_saver_reduction_kw'] / 1000
    total_mw_saved = homes * scaled_kw_per_home / 1000
    return df_a, total_mw_saved

# ============================================================
# IMPACT AND DISPATCH CALCULATIONS
# ============================================================
def compute_impact_metrics(row, homes, reduction_rate):
    reduction_kw   = homes * KW_PER_HOME * (reduction_rate / 4)
    reduction_mw   = reduction_kw / 1000
    mwh_saved      = reduction_mw * 1 
    cost_savings   = mwh_saved * 100  
    carbon_intensity = row[CARBON_COL] 
    co2_avoided_tons = (carbon_intensity * mwh_saved) / 1000
    return mwh_saved, cost_savings, co2_avoided_tons

def compute_dispatch_priority(row):
    score  = row['vulnerability_score'] * 0.5
    score += row.get('vuln_probability', 0) * 30
    score += min(row[CARBON_COL] / 10, 20)
    return round(score, 1)

def get_risk_drivers(row, vulnerability_threshold, df_full):
    drivers = []
    score  = row['vulnerability_score']
    carbon = row[CARBON_COL]
    cfe    = row[CFE_COL]

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
# RUN FULL SPA PIPELINE ON FULL DATASET (NOW CACHED FOR SPEED)
# ============================================================
@st.cache_data
def run_base_pipeline(df_raw, _ml_model):
    """Caches Sense and Predict so XGBoost only runs ONCE at startup."""
    df_base, thresh = sense_layer(df_raw)
    df_base['hour']       = df_base['Datetime (UTC)'].dt.hour
    df_base['month']      = df_base['Datetime (UTC)'].dt.month
    df_base['month_name'] = df_base['Datetime (UTC)'].dt.strftime('%b')
    df_base['date']       = df_base['Datetime (UTC)'].dt.date
    df_base['day_of_week']= df_base['Datetime (UTC)'].dt.dayofweek
    df_base['year']       = df_base['Datetime (UTC)'].dt.year
    df_base['week']       = df_base['Datetime (UTC)'].dt.isocalendar().week.astype(int)
    
    df_base = predict_layer(df_base, _ml_model)
    return df_base, thresh

# 1. Load the heavy ML data from cache (Instant)
df, VULNERABILITY_THRESHOLD = run_base_pipeline(df, model)

# 2. Run Act layer (Recalculates instantly when sliders move)
df_full, _ = act_layer(df, REDUCTION_RATE, NUM_HOMES)

SENSE_TRIGGERS_TOTAL   = int(df_full['sense_triggered'].sum())
PREDICT_TRIGGERS_TOTAL = int(df_full['predict_triggered'].sum())
SPA_ACTIONS_TOTAL      = int(df_full['spa_action_triggered'].sum())

NOTEBOOK_SENSE_TRIGGERS   = 1316
NOTEBOOK_PREDICT_TRIGGERS = 1659
NOTEBOOK_SPA_ACTIONS      = 154
NOTEBOOK_VULN_THRESHOLD   = 74.6

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.image("https://img.icons8.com/fluency/96/lightning-bolt.png", width=60)
st.sidebar.title("Grid Saver")
st.sidebar.markdown("**Adaptive Grid Intelligence Platform**")
st.sidebar.divider()

live_mode = st.sidebar.toggle("Recent Window View (Last 24 Hours)", value=False)
if live_mode:
    st.sidebar.markdown("<p style='color:#2ECC71; font-size:0.8rem;'>Showing the most recent 24 hours of grid data</p>", unsafe_allow_html=True)

st.sidebar.divider()

months_present = [m for m in month_order if m in df['month_name'].unique()]
month_options  = ['All Year'] + months_present
selected_month = st.sidebar.selectbox("Select Month", month_options)

reduction_rate_input = st.sidebar.slider("HVAC Reduction Rate (%)", min_value=1, max_value=10, value=4, step=1)
homes = st.sidebar.slider("Homes Coordinated", min_value=1000, max_value=1000000, value=100000, step=1000)
apply_intervention = st.sidebar.toggle("Apply Grid Saver Intervention", value=True)

st.sidebar.divider()
st.sidebar.markdown("**Stack**\nColab + GitHub + Streamlit\n*Justine Adzormado*")

# ============================================================
# FILTER DATA FOR DISPLAY
# ============================================================
if live_mode:
    df_view = df[df['Datetime (UTC)'] >= df['Datetime (UTC)'].max() - pd.Timedelta(hours=24)].copy()
elif selected_month != 'All Year':
    df_view = df[df['month_name'] == selected_month].copy()
else:
    df_view = df.copy()

if df_view.empty:
    st.warning("No data available for selected filters.")
    st.stop()

# Run act layer on filtered view for display only
df_view, total_mw_saved = act_layer(df_view, reduction_rate_input, homes)

# ============================================================
# HEADER & STATUS
# ============================================================
mode_label = "RECENT WINDOW VIEW" if live_mode else "ANALYSIS MODE"
mode_color = "#2ECC71" if live_mode else "#4A9EFF"

st.markdown(f"""
<div style='background: linear-gradient(135deg, #1B4F8C, #0D1117); padding: 30px; border-radius: 12px; margin-bottom: 20px; border: 1px solid #30363D;'>
    <div style='display: flex; justify-content: space-between; align-items: center;'>
        <div>
            <h1 style='color: white; margin: 0; font-size: 2.2rem;'>⚡ Grid Saver</h1>
            <p style='color: #4A9EFF; margin: 5px 0 0 0; font-size: 1.1rem;'>Adaptive Grid Intelligence Platform</p>
        </div>
        <div style='background: {mode_color}22; border: 2px solid {mode_color}; padding: 10px 20px; border-radius: 8px; text-align: center;'>
            <p style='color: {mode_color}; font-weight: bold; margin: 0; font-size: 1rem;'>{"🕐 " if live_mode else "📊 "}{mode_label}</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("## ⚡ Grid Status")

current_row    = df_view.iloc[-1]
current_score  = current_row['vulnerability_score']
current_status = current_row['grid_status']
current_carbon = current_row[CARBON_COL]
current_cfe    = current_row[CFE_COL]
current_prob   = current_row['vuln_probability']

vulnerable_hours   = int(df_view['vulnerability_event'].sum())
vulnerable_pct     = df_view['vulnerability_event'].mean() * 100

status_color = {'STABLE': '#2ECC71', 'WARNING': '#F39C12', 'CRITICAL': '#E74C3C'}
status_icon  = {'STABLE': '🟢', 'WARNING': '🟡', 'CRITICAL': '🔴'}

col1, col2, col3, col4, col5, col6 = st.columns(6)
cards = [
    (col1, status_icon[current_status], current_status,   "Grid Status", status_color[current_status]),
    (col2, f"{current_score:.0f}",      "/100",            "Vulnerability Score", "white"),
    (col3, f"{current_carbon:.0f}",     "gCO₂/kWh",  "Carbon Intensity", "#E74C3C"),
    (col4, f"{current_cfe:.1f}%",       "clean energy",    "Carbon-Free Energy", "#2ECC71"),
    (col5, f"{vulnerable_pct:.1f}%",    "of period",       "Vulnerability Rate", "#4A9EFF"),
    (col6, f"{current_prob:.2f}",       "probability",     "24hr Risk Projection", "#9B59B6"),
]
for col, val, sub, label, color in cards:
    with col:
        st.markdown(f"""
        <div class='metric-card'>
            <h2 style='color: {color}; font-size: 1.6rem; margin: 0;'>{val}</h2>
            <p style='color: #666; margin: 2px 0; font-size: 0.75rem;'>{sub}</p>
            <p style='color: #888; margin: 0; font-size: 0.75rem;'>{label}</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# SECTION 2 — RISK DRIVERS
# ============================================================
st.markdown("<br>## Risk Drivers", unsafe_allow_html=True)
drivers = get_risk_drivers(current_row, VULNERABILITY_THRESHOLD, df)
col_d1, col_d2 = st.columns(2)
for i, driver in enumerate(drivers):
    if i % 2 == 0:
        col_d1.markdown(f"**{driver}**")
    else:
        col_d2.markdown(f"**{driver}**")

# ============================================================
# SECTION 3 — GRID DEMAND AND VULNERABILITY WINDOWS (FIXED FOR SPEED)
# ============================================================
st.markdown("<br>## Grid Demand and Vulnerability Windows", unsafe_allow_html=True)

color_map  = {'STABLE': '#2ECC71', 'WARNING': '#F39C12', 'CRITICAL': '#E74C3C'}
fig_demand = go.Figure()

# Fast continuous background line
fig_demand.add_trace(go.Scatter(
    x=df_view['Datetime (UTC)'], y=df_view['vulnerability_score'], mode='lines',
    line=dict(color='#555', width=1), showlegend=False
))

# Overlay colored status dots (Vectorized and instant)
for status in ['STABLE', 'WARNING', 'CRITICAL']:
    mask = df_view['grid_status'] == status
    if mask.any():
        fig_demand.add_trace(go.Scatter(
            x=df_view[mask]['Datetime (UTC)'], y=df_view[mask]['vulnerability_score'],
            mode='markers', name=status, marker=dict(color=color_map[status], size=4, opacity=0.9),
        ))

fig_demand.add_hline(
    y=VULNERABILITY_THRESHOLD, line_dash='dash', line_color='#FF4444',
    annotation_text=f'Vulnerability Threshold ({VULNERABILITY_THRESHOLD:.0f})', annotation_font_color='#FF4444'
)
fig_demand.add_trace(go.Scatter(
    x=df_view['Datetime (UTC)'], y=df_view['vuln_probability'] * 100, mode='lines',
    name='Projected Risk Signal (%)', line=dict(color='#9B59B6', dash='dot', width=1.2), opacity=0.8,
))
fig_demand.update_layout(
    paper_bgcolor='#161B22', plot_bgcolor='#161B22', font=dict(color='white'),
    title=dict(text='Grid Vulnerability Score', font=dict(color='white', size=14)),
    xaxis=dict(gridcolor='#30363D', color='#888'), yaxis=dict(gridcolor='#30363D', color='#888'),
    legend=dict(bgcolor='#1A1A2E', bordercolor='#333'), height=350, margin=dict(t=50, b=30),
)
st.plotly_chart(fig_demand, use_container_width=True)

# ============================================================
# SECTION 5 — RECOMMENDED GRID ACTION & SMART RECOMMENDATIONS
# ============================================================
st.markdown("<br>## Recommended Grid Action", unsafe_allow_html=True)
dispatch_score = compute_dispatch_priority(current_row)
mwh_saved, cost_savings, co2_avoided = compute_impact_metrics(current_row, homes, reduction_rate_input)

# SAFELY compute true SPA-aggregated impact avoiding KeyError
scaled_kw_per_home_recs = KW_PER_HOME * (reduction_rate_input / 4)
spa_events_df = df_view[df_view['spa_action_triggered']].copy() if 'spa_action_triggered' in df_view.columns else pd.DataFrame()
event_count = len(spa_events_df)
total_mwh_all_events = event_count * (homes * scaled_kw_per_home_recs) / 1000
total_co2_all_events = (spa_events_df[CARBON_COL].mean() * total_mwh_all_events) / 1000 if event_count > 0 else 0
total_cost_all_events = total_mwh_all_events * 100

st.markdown(f"""
<div style='background: #161B22; border-left: 5px solid #E74C3C; padding: 20px; border-radius: 8px; margin: 10px 0;'>
    <h3 style='color: #E74C3C; margin: 0;'>🔴 Operational Intelligence</h3>
    <p style='color: white; margin: 10px 0 5px 0; font-size: 1rem;'>
        Total across {event_count} events this period: {total_mwh_all_events:.2f} MWh | ${total_cost_all_events:,.0f} | {total_co2_all_events:.3f} tons CO₂ avoided
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SECTION 6 — LOAD REDUCTION SIMULATION
# ============================================================
st.markdown("## Grid Saver Load Reduction Simulation")

df_view['synthetic_demand_mw'] = 35000 + (
    np.where(df_view['month'].isin([6, 7, 8]), 5000, np.where(df_view['month'].isin([12, 1, 2]), 3000, 0)) +
    np.where(df_view['hour'].between(15, 20), 2000, np.where(df_view['hour'].between(6, 9), 1000, -500))
)
df_view['adjusted_demand_mw'] = df_view['synthetic_demand_mw'] - df_view['reduction_mw']

peak_idx  = df_view['synthetic_demand_mw'].idxmax()
peak_time = df_view.loc[peak_idx, 'Datetime (UTC)']

spa_active_df = df_view[df_view['spa_action_triggered']].copy()
if not spa_active_df.empty:
    spa_peak_idx = spa_active_df['synthetic_demand_mw'].idxmax()
    original_peak = spa_active_df.loc[spa_peak_idx, 'synthetic_demand_mw']
    after_peak = spa_active_df.loc[spa_peak_idx, 'adjusted_demand_mw']
    peak_reduction_mw = original_peak - after_peak
    pct_reduction = (peak_reduction_mw / original_peak) * 100 if original_peak > 0 else 0
else:
    original_peak = df_view['synthetic_demand_mw'].max()
    after_peak = original_peak
    peak_reduction_mw = 0
    pct_reduction = 0

col_p1, col_p2, col_p3, col_p4 = st.columns(4)
peak_cards = [
    (col_p1, f"{original_peak:,.0f} MW", "Original Peak", "white"),
    (col_p2, f"{after_peak:,.0f} MW", "After Grid Saver", "#2ECC71"),
    (col_p3, f"{pct_reduction:.2f}%", "Peak Demand Reduction", "#4A9EFF"),
    (col_p4, f"{peak_reduction_mw:,.2f} MW", "Peak Load Shed", "#F39C12"),
]
for col, val, label, color in peak_cards:
    with col:
        st.markdown(f"""
        <div class='metric-card'>
            <h2 style='color: {color}; font-size: 1.4rem; margin: 0;'>{val}</h2>
            <p style='color: #888; margin: 4px 0 0 0; font-size: 0.78rem;'>{label}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# BEFORE PLOT (Fixed the timestamp crash)
fig_before = go.Figure()
fig_before.add_trace(go.Scatter(
    x=df_view['Datetime (UTC)'], y=df_view['synthetic_demand_mw'], mode='lines', name='Original Demand (Proxy)', line=dict(color='#E74C3C', width=1.5),
))
fig_before.add_trace(go.Scatter(
    x=df_view[df_view['spa_action_triggered']]['Datetime (UTC)'], y=df_view[df_view['spa_action_triggered']]['synthetic_demand_mw'],
    mode='markers', name='SPA Action Triggered', marker=dict(color='#F39C12', size=6, symbol='circle'),
))
# SAFE PEAK LINE
fig_before.add_trace(go.Scatter(
    x=[peak_time, peak_time], y=[df_view['synthetic_demand_mw'].min(), df_view['synthetic_demand_mw'].max()],
    mode='lines', name='Peak Demand', line=dict(color='#FFFFFF', width=1.5, dash='dash')
))
fig_before.update_layout(
    paper_bgcolor='#161B22', plot_bgcolor='#161B22', font=dict(color='white'),
    title=dict(text='Before Grid Saver — Original Demand Proxy', font=dict(color='white', size=13)),
    xaxis=dict(gridcolor='#30363D'), yaxis=dict(gridcolor='#30363D'), height=320, margin=dict(t=50, b=20),
)
st.plotly_chart(fig_before, use_container_width=True)

# AFTER PLOT (Fixed the shape crash)
fig_after = go.Figure()
fig_after.add_trace(go.Scatter(
    x=df_view['Datetime (UTC)'], y=df_view['synthetic_demand_mw'], mode='lines', name='Original Demand', line=dict(color='#E74C3C', width=1, dash='dot'), opacity=0.35,
))
fig_after.add_trace(go.Scatter(
    x=df_view['Datetime (UTC)'], y=df_view['adjusted_demand_mw'], mode='lines', name='After Grid Saver', line=dict(color='#2ECC71', width=1.5), fill='tonexty', fillcolor='rgba(46, 204, 113, 0.08)'
))
fig_after.update_layout(
    paper_bgcolor='#161B22', plot_bgcolor='#161B22', font=dict(color='white'),
    title=dict(text='After Grid Saver — Demand with Intervention Applied', font=dict(color='white', size=13)),
    xaxis=dict(gridcolor='#30363D'), yaxis=dict(gridcolor='#30363D'), height=320, margin=dict(t=50, b=20),
)
st.plotly_chart(fig_after, use_container_width=True)


# ============================================================
# SECTION 8 — SYSTEM ARCHITECTURE (Fixed the typo crash)
# ============================================================
st.markdown("## System Architecture")
col_a, col_b, col_c = st.columns(3)

arch = [
    (col_a, "👁️", "SENSE",   "#1B4F8C", "#4A9EFF",
     "Detect grid vulnerability signals continuously", "Carbon intensity monitoring", "Electricity Maps US-TEX-ERCO", "8,760 hourly records"),
    (col_b, "🧠", "PREDICT", "#1A6B2E", "#2ECC71",
     "Forecast vulnerability windows", "XGBoost model | 91.6% Recall", "PJM 32,896 hourly records", "24hr Risk Projection"),
    (col_c, "⚡", "ACT",     "#7B1A1A", "#E74C3C",
     "Coordinate residential HVAC load reduction", "3 to 5% targeted precision", "Pecan Street 868,096 records", "Human-override safety protocol"),
]

for col, icon, title, bg, color, l1, l2, l3, l4 in arch:
    with col:
        st.markdown(f"""
        <div style='background: {bg}22; border: 1px solid {color}; padding: 20px; border-radius: 10px; text-align: center; border-top: 4px solid {color};'>
            <h2 style='color: {color}; font-size: 2rem; margin: 0;'>{icon}</h2>
            <h3 style='color: {color}; margin: 10px 0 5px 0;'>{title}</h3>
            <p style='color: #888; font-size: 0.85rem; margin: 0;'>{l1}<br>{l2}<br>{l3}<br>{l4}</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# SECTION 9 — REPORTS AND INSIGHTS (Safely Rendered)
# ============================================================
st.markdown("## Reports and Insights")

if live_mode:
    st.info("📊 Reports are disabled in Recent Window View. Toggle 'Recent Window View' off in the sidebar to run full historical reports.")
else:
    st.markdown("*Select a time period to view grid performance analysis and download a report.*")
    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        report_type = st.selectbox("Report Type", ["Yearly", "Monthly", "Weekly"])
    with col_r2:
        report_year = st.selectbox("Year", sorted(df['year'].unique(), reverse=True))
    with col_r3:
        report_month, report_week = None, None
        if report_type == "Monthly":
            available_months = sorted(df[df['year'] == report_year]['month'].unique())
            selected_month_name = st.selectbox("Month", [MONTH_NAMES[m] for m in available_months])
            report_month = [k for k, v in MONTH_NAMES.items() if v == selected_month_name][0]
        elif report_type == "Weekly":
            report_week = st.selectbox("Week", sorted(df[df['year'] == report_year]['week'].unique()))

    if report_type == "Yearly":
        df_report = df[df['year'] == report_year].copy()
        period_label = f"{report_year}"
    elif report_type == "Monthly":
        df_report = df[(df['year'] == report_year) & (df['month'] == report_month)].copy()
        period_label = f"{MONTH_NAMES[report_month]} {report_year}"
    else:
        df_report = df[(df['year'] == report_year) & (df['week'] == report_week)].copy()
        period_label = f"Week {report_week}, {report_year}"

    if df_report.empty:
        st.warning(f"No data available for {period_label}.")
    else:
        spa_report_df, _ = act_layer(df_report, reduction_rate_input, homes)
        avg_vulnerability  = spa_report_df['vulnerability_score'].mean()
        peak_vulnerability = spa_report_df['vulnerability_score'].max()
        spa_events_report  = int(spa_report_df['spa_action_triggered'].sum())

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Avg Vulnerability", f"{avg_vulnerability:.1f}")
        col_m2.metric("Peak Vulnerability", f"{peak_vulnerability:.1f}")
        col_m3.metric("SPA Actions Triggered", f"{spa_events_report:,}")

        fig_report = go.Figure()
        fig_report.add_trace(go.Scatter(
            x=spa_report_df['Datetime (UTC)'], y=spa_report_df['vulnerability_score'], mode='lines',
            name='Vulnerability Score', line=dict(color='#4A9EFF', width=1.5), fill='tozeroy', fillcolor='rgba(74, 158, 255, 0.1)'
        ))
        fig_report.update_layout(
            paper_bgcolor='#161B22', plot_bgcolor='#161B22', font=dict(color='white'),
            title=dict(text=f'Vulnerability Trend — {period_label}', font=dict(color='white', size=13)),
            xaxis=dict(gridcolor='#30363D'), yaxis=dict(gridcolor='#30363D', range=[0, 100]), height=300, margin=dict(t=40, b=20)
        )
        st.plotly_chart(fig_report, use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div style='background: #161B22; padding: 15px; border-radius: 8px; border: 1px solid #30363D; text-align: center; margin-top: 20px;'>
    <p style='color: #888; margin: 0; font-size: 0.85rem;'>Grid Saver | Adaptive Grid Intelligence Platform | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
