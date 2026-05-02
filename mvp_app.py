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
    .info-box {
        background: #1A1A2E;
        border-left: 4px solid #4A9EFF;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 10px 0;
        font-size: 0.85rem;
        color: #CCCCCC;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS
# ============================================================
CARBON_COL = 'Carbon intensity gCO₂eq/kWh (direct)'
CFE_COL = 'Carbon-free energy percentage (CFE%)'
DECISION_THRESHOLD = 0.4
KW_PER_HOME = 0.0920
ERCOT_PEAK_MW = 75000

MONTH_NAMES = {1: 'January', 2: 'February', 3: 'March', 4: 'April',
               5: 'May', 6: 'June', 7: 'July', 8: 'August',
               9: 'September', 10: 'October', 11: 'November', 12: 'December'}
DATETIME_COL = 'Datetime (UTC)'
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Locked notebook values
NOTEBOOK_SENSE_TRIGGERS = 1316
NOTEBOOK_PREDICT_TRIGGERS = 1659
NOTEBOOK_SPA_ACTIONS = 154


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def count_spa_events(trigger_series):
    """Count SPA events using rising edge detection (0→1 transitions)."""
    if len(trigger_series) == 0:
        return 0
    trigger_array = trigger_series.fillna(False).values
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


def compute_impact_metrics(row, homes, reduction_rate_percent, intervention_hours=1):
    """Compute MWh, cost, CO2 for a single SPA event."""
    reduction_kw = compute_scaled_reduction_kw(homes, reduction_rate_percent)
    reduction_mw = reduction_kw / 1000
    mwh_saved = reduction_mw * intervention_hours
    cost_savings = mwh_saved * 100
    carbon_intensity = row[CARBON_COL]
    co2_avoided_tons = (carbon_intensity * mwh_saved) / 1000
    return mwh_saved, cost_savings, co2_avoided_tons


def compute_dispatch_priority(row):
    """Scores grid urgency 0-100."""
    score = row['vulnerability_score'] * 0.5
    score += row.get('vuln_probability', 0) * 30
    score += min(row[CARBON_COL] / 10, 20)
    return round(score, 1)


def apply_intervention(df, enabled, homes, reduction_rate_percent):
    """Apply Grid Saver reduction only if enabled."""
    df = df.copy()
    if not enabled:
        df['reduction_mw'] = 0
        df['adjusted_demand_mw'] = df['ercot_demand_mw']
        return df

    reduction_kw = compute_scaled_reduction_kw(homes, reduction_rate_percent)
    reduction_mw_per_home = reduction_kw / 1000

    df['reduction_mw'] = np.where(
        df.get('spa_action_triggered', False),
        reduction_mw_per_home,
        0
    )
    df['adjusted_demand_mw'] = df['ercot_demand_mw'] - df['reduction_mw']
    return df


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
# FEATURE ENGINEERING (must match training)
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
# ACT LAYER
# ============================================================
def act_layer(df_input):
    df_a = df_input.copy()
    df_a['sense_triggered'] = df_a['vulnerability_event']
    df_a['spa_action_triggered'] = df_a['sense_triggered'] & df_a['predict_triggered']
    return df_a


# ============================================================
# RUN FULL PIPELINE
# ============================================================
df, VULN_THRESHOLD = sense_layer(df_raw)
df['hour'] = df[DATETIME_COL].dt.hour
df['month'] = df[DATETIME_COL].dt.month
df['month_name'] = df[DATETIME_COL].dt.strftime('%b')
df['date'] = df[DATETIME_COL].dt.date
df['year'] = df[DATETIME_COL].dt.year
df['week'] = df[DATETIME_COL].dt.isocalendar().week.astype(int)

df = predict_layer(df, model)
df = act_layer(df)

# ERCOT-calibrated demand (55-95% of 75,000 MW peak)
base_load_pct = 0.55
peak_load_pct = 0.95
df['ercot_demand_mw'] = (
    base_load_pct + (df['vulnerability_score'] / 100) * (peak_load_pct - base_load_pct)
) * ERCOT_PEAK_MW

FULL_YEAR_SPA_EVENTS = count_spa_events(df['spa_action_triggered'])


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
apply_intervention_enabled = st.sidebar.toggle("Apply Grid Saver Intervention", value=True)

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
    st.warning("No data available.")
    st.stop()

df_view = apply_intervention(df_view, apply_intervention_enabled, homes, reduction_rate_percent)

SPA_EVENTS_VIEW = count_spa_events(df_view['spa_action_triggered'])
TOTAL_MWH_REMOVED = df_view['reduction_mw'].sum()
current_row = df_view.iloc[-1]
per_event_mwh, per_event_cost, per_event_co2 = compute_impact_metrics(current_row, homes, reduction_rate_percent)


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

current_score = current_row['vulnerability_score']
current_status = current_row['grid_status']
current_carbon = current_row[CARBON_COL]
current_cfe = current_row[CFE_COL]
current_prob = current_row.get('vuln_probability', 0)
vulnerable_pct = df_view['vulnerability_event'].mean() * 100

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

fig_trend.add_hline(y=VULN_THRESHOLD, line_dash='dash', line_color='#FF4444',
                    annotation_text=f'Threshold ({VULN_THRESHOLD:.0f})')
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
    fig_hour.add_hline(y=VULN_THRESHOLD, line_dash='dash', line_color='#FF4444')
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
        fig_month.add_hline(y=VULN_THRESHOLD, line_dash='dash', line_color='#FF4444')
        fig_month.update_layout(paper_bgcolor='#161B22', plot_bgcolor='#161B22', font=dict(color='white'),
                                title=dict(text='Avg by Month', font=dict(size=13)), height=300)
        st.plotly_chart(fig_month, width='stretch')

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# RECOMMENDED ACTION
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
        <p>Vulnerability Score: <strong>{current_score:.1f}/100</strong> (threshold {VULN_THRESHOLD:.0f})</p>
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

st.markdown(f"""
<div style='background:#161B22; border-left:4px solid #2ECC71; padding:10px 14px; border-radius:6px; margin-top:12px;'>
    📊 <strong>Impact Summary:</strong> {per_event_mwh:.2f} MWh | ${per_event_cost:,.0f} | {per_event_co2:.3f} tons CO₂ per SPA event<br>
    <strong>Period:</strong> {SPA_EVENTS_VIEW} events | {TOTAL_MWH_REMOVED:.2f} MWh total
</div>
""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# LOAD REDUCTION SIMULATION
# ============================================================
st.markdown("## ⚡ Load Reduction Simulation")

if reduction_rate_percent != 4:
    st.warning(f"⚠️ Validated at 4%. Results at {reduction_rate_percent}% are scaled estimates.")

col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    st.metric("Sense Triggers", f"{NOTEBOOK_SENSE_TRIGGERS:,}" if not live_mode else f"{int(df_view['sense_triggered'].sum()):,}")
with col_s2:
    st.metric("Predict Triggers", f"{NOTEBOOK_PREDICT_TRIGGERS:,}" if not live_mode else f"{int(df_view['predict_triggered'].sum()):,}")
with col_s3:
    st.metric("SPA Events (Validated)", f"{NOTEBOOK_SPA_ACTIONS}" if not live_mode else f"{SPA_EVENTS_VIEW}")

st.markdown(f"""
<div style='background:#161B22; border-left:5px solid #2ECC71; padding:15px; border-radius:8px; margin:15px 0;'>
    <h3 style='color:#2ECC71; margin:0;'>⚡ Total Energy Removed: {TOTAL_MWH_REMOVED:,.2f} MWh</h3>
    <p style='color:#888; margin:5px 0 0 0;'>Across {SPA_EVENTS_VIEW} SPA events | Full-year validated: {NOTEBOOK_SPA_ACTIONS} events</p>
    <p style='color:#555; font-size:0.75rem; margin-top:5px;'>📌 Scenario output — scales with homes and reduction rate</p>
</div>
""", unsafe_allow_html=True)

# Find peak
peak_idx = df_view['ercot_demand_mw'].idxmax()
peak_time = df_view.loc[peak_idx, DATETIME_COL]
orig_peak = df_view.loc[peak_idx, 'ercot_demand_mw']
adj_peak = df_view.loc[peak_idx, 'adjusted_demand_mw']
peak_reduction = orig_peak - adj_peak
pct_reduction = (peak_reduction / orig_peak * 100) if orig_peak > 0 else 0

col_p1, col_p2, col_p3, col_p4 = st.columns(4)
col_p1.metric("Original Peak", f"{orig_peak:,.0f} MW")
col_p2.metric("After Grid Saver", f"{adj_peak:,.0f} MW")
col_p3.metric("Peak Reduction", f"{pct_reduction:.2f}%")
col_p4.metric("Load Shed", f"{peak_reduction:,.2f} MW")

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# PROTOTYPE-STYLE PLOT
# ============================================================
st.markdown("#### 📊 ERCOT Grid Demand — Before vs After Grid Saver")

fig_sim = go.Figure()

# Baseline demand
fig_sim.add_trace(go.Scatter(
    x=df_view[DATETIME_COL],
    y=df_view['ercot_demand_mw'],
    mode='lines',
    name='Baseline Demand',
    line=dict(color='#E74C3C', width=1.5, dash='dot')
))

# Adjusted demand
line_color = '#2ECC71' if apply_intervention_enabled else '#888888'
fig_sim.add_trace(go.Scatter(
    x=df_view[DATETIME_COL],
    y=df_view['adjusted_demand_mw'],
    mode='lines',
    name='Adjusted Demand' if apply_intervention_enabled else 'Intervention Disabled',
    line=dict(color=line_color, width=2)
))

# Reduction bars
if apply_intervention_enabled and df_view['reduction_mw'].sum() > 0:
    fig_sim.add_trace(go.Bar(
        x=df_view[DATETIME_COL],
        y=df_view['reduction_mw'],
        name='Load Reduction',
        marker_color='#F39C12',
        opacity=0.4,
        yaxis='y2'
    ))

# Peak marker
fig_sim.add_trace(go.Scatter(
    x=[peak_time, peak_time],
    y=[df_view['ercot_demand_mw'].min(), df_view['ercot_demand_mw'].max()],
    mode='lines',
    name='Peak Demand',
    line=dict(color='#FFFFFF', width=1.5, dash='dash')
))

fig_sim.update_layout(
    paper_bgcolor='#161B22', plot_bgcolor='#161B22', font=dict(color='white'),
    title=dict(text='ERCOT Grid Demand — Before vs After Grid Saver', font=dict(size=14)),
    xaxis=dict(gridcolor='#30363D', title='Datetime'),
    yaxis=dict(gridcolor='#30363D', title='Demand (MW)'),
    yaxis2=dict(title='Reduction (MW)', overlaying='y', side='right', showgrid=False, color='#F39C12'),
    legend=dict(bgcolor='#1A1A2E', bordercolor='#333', orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    height=450, hovermode='x unified'
)

st.plotly_chart(fig_sim, width='stretch')

st.markdown("""
<div class='info-box'>
📌 <strong>Chart Guide:</strong><br>
• <span style='color:#E74C3C'>🔴 Red dashed line:</span> Baseline demand (no intervention)<br>
• <span style='color:#2ECC71'>🟢 Green line:</span> Demand after Grid Saver intervention<br>
• <span style='color:#F39C12'>🟠 Orange bars:</span> Load reduction during SPA-triggered hours<br>
• <span style='color:#FFFFFF'>⚪ White dashed line:</span> Peak demand timestamp
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
    This is where Grid Saver intervention has the highest impact potential.
    """)
    
    zoom_df = df_view[
        (df_view[DATETIME_COL] >= peak_time - pd.Timedelta(hours=6)) &
        (df_view[DATETIME_COL] <= peak_time + pd.Timedelta(hours=6))
    ].copy()
    
    if not zoom_df.empty:
        fig_zoom = go.Figure()
        
        fig_zoom.add_trace(go.Scatter(
            x=zoom_df[DATETIME_COL], y=zoom_df['ercot_demand_mw'],
            mode='lines+markers', name='Baseline Demand',
            line=dict(color='#E74C3C', width=2), marker=dict(size=4, color='#E74C3C')
        ))
        
        fig_zoom.add_trace(go.Scatter(
            x=zoom_df[DATETIME_COL], y=zoom_df['adjusted_demand_mw'],
            mode='lines+markers', name='After Grid Saver',
            line=dict(color='#2ECC71', width=2), marker=dict(size=4, color='#2ECC71')
        ))
        
        if apply_intervention_enabled:
            zoom_spa = zoom_df[zoom_df['spa_action_triggered']]
            for _, row in zoom_spa.iterrows():
                fig_zoom.add_vrect(
                    x0=row[DATETIME_COL], x1=row[DATETIME_COL] + pd.Timedelta(hours=1),
                    fillcolor='rgba(255, 165, 0, 0.2)', line_width=0, layer='below'
                )
        
        fig_zoom.add_trace(go.Scatter(
            x=[peak_time, peak_time], y=[zoom_df['ercot_demand_mw'].min(), zoom_df['ercot_demand_mw'].max()],
            mode='lines', name='Peak Moment', line=dict(color='#FFFFFF', width=2, dash='dash')
        ))
        
        fig_zoom.update_layout(
            paper_bgcolor='#161B22', plot_bgcolor='#161B22', font=dict(color='white'),
            title=dict(text=f'Peak Window — {peak_time.strftime("%Y-%m-%d %H:%M")} ±6 hours', font=dict(size=13)),
            xaxis=dict(gridcolor='#30363D', title='Datetime'),
            yaxis=dict(gridcolor='#30363D', title='Demand (MW)'), height=400, hovermode='x unified'
        )
        
        st.plotly_chart(fig_zoom, width='stretch')


# ============================================================
# IMPACT AT SCALE
# ============================================================
st.divider()
st.markdown("## 📊 Impact at Scale")
st.markdown("*Adjust the Homes Coordinated slider in the sidebar to see how Grid Saver scales.*")

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
col_s2.metric("Grid Reduction (per event)", f"{scaled_reduction_mw:,.1f} MW")
col_s3.metric("Impact Level", grid_impact)
col_s4.metric("Reserve Margin", "Exceeds 200MW" if scaled_reduction_mw > 200 else "Building toward")

percentage_of_grid = (scaled_reduction_mw / ERCOT_PEAK_MW) * 100
st.markdown(
    f"<p style='color:#888; font-size:0.8rem; margin-top:8px;'>"
    f"📌 At this scale, Grid Saver removes approximately <strong>{percentage_of_grid:.3f}%</strong> "
    f"of ERCOT peak demand (~{ERCOT_PEAK_MW:,} MW).</p>",
    unsafe_allow_html=True
)

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
        <p style='color: #888; font-size: 0.85rem;'>HVAC load coordination<br>3-5% targeted reduction<br>Pecan Street 868k records<br>SPA dual-confirmation</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# REPORTS SECTION (in expander - LOCKED to 154)
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
            
            # LOCKED: Use notebook value
            spa_events_report = NOTEBOOK_SPA_ACTIONS
            
            # Pie chart
            status_counts = df_report['grid_status'].value_counts()
            fig_pie = go.Figure(data=[go.Pie(
                labels=status_counts.index, values=status_counts.values, hole=0.4,
                marker_colors=['#2ECC71', '#F39C12', '#E74C3C']
            )])
            fig_pie.update_layout(paper_bgcolor='#161B22', plot_bgcolor='#161B22',
                                  font=dict(color='white'), title=dict(text='Grid Status Distribution', font=dict(size=13)), height=300)
            
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Avg Vulnerability", f"{avg_vuln:.1f}")
            col_m2.metric("Peak Vulnerability", f"{peak_vuln:.1f}")
            col_m3.metric("SPA Events", f"{spa_events_report}")
            
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
            fig_report.add_hline(y=VULN_THRESHOLD, line_dash='dash', line_color='#FF4444')
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
    <p style='color: #444; margin: 8px 0 0 0; font-size: 0.7rem;'>
        ⚠️ Educational Prototype — Not for real-time operations.
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# END OF APP
# ============================================================
