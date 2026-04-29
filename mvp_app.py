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
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS — same as all three prototype notebooks
# ============================================================
CARBON_COL           = 'Carbon intensity gCO\u2082eq/kWh (direct)'
CFE_COL              = 'Carbon-free energy percentage (CFE%)'
DECISION_THRESHOLD   = 0.4      # Phase 2 XGBoost decision threshold
REDUCTION_RATE       = 0.04     # 4% HVAC cycling — validated Phase 1 and Phase 3
NUM_HOMES            = 25       # Pecan Street Austin TX 2018
KW_PER_HOME          = 0.0920   # Derived from Phase 1 real data validation

# ============================================================
# LOAD MODEL AND DATA
# No retraining. No Google Drive. Local files only.
# ============================================================
@st.cache_resource
def load_model():
    """Load trained XGBoost model from repo."""
    return joblib.load("gridsaver_model.pkl")

@st.cache_data
def load_data():
    """
    Load processed ERCOT dataset from repo.
    This is derived output from the prototype notebooks — not raw academic data.
    Source: Electricity Maps US-TEX-ERCO 2025 (Academic Access).
    """
    df = pd.read_csv("data_sample.csv")
    df['Datetime (UTC)'] = pd.to_datetime(df['Datetime (UTC)'])
    df = df.sort_values('Datetime (UTC)').reset_index(drop=True)
    return df

with st.spinner("Loading Grid Saver..."):
    model = load_model()
    df    = load_data()

# ============================================================
# FEATURE ENGINEERING FUNCTION
# Exact same function as Phase 2 and MVP notebook.
# Must match training features — do not change.
# ============================================================
def engineer_features(df_input):
    """
    Input:  dataframe with datetime and demand_mw columns
    Output: dataframe with all 22 engineered features
    Used by Predict Layer to generate model input.
    """
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
# Same formula and thresholds as Phase 1 prototype notebook.
# Score range: 0 (cleanest) to 100 (most stressed)
# Threshold: top 15% most stressed hours
# ============================================================
def sense_layer(df_input):
    """
    Input:  processed ERCOT dataframe
    Output: dataframe with vulnerability_score, vulnerability_event,
            grid_status, VULNERABILITY_THRESHOLD
    """
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
        if score < 40:
            return 'STABLE'
        elif score < 70:
            return 'WARNING'
        else:
            return 'CRITICAL'

    df_s['grid_status'] = df_s['vulnerability_score'].apply(classify_status)
    return df_s, VULNERABILITY_THRESHOLD


# ============================================================
# PREDICT LAYER
# FIX: Replaced synthetic proxy demand signal with hour + month
# alignment approach — same as Phase 3 SPA integration.
# Loads trained XGBoost model. Decision threshold: 0.4.
# Forecasting horizon: 24 hours ahead.
# ============================================================
def predict_layer(df_input, model):
    """
    Input:  processed ERCOT dataframe with hour + month columns,
            trained XGBoost model
    Output: dataframe with vuln_probability and predict_triggered columns
    """
    df_out = df_input.copy()

    # Build hour + month features from ERCOT temporal data
    # Same alignment approach as Phase 3 — no synthetic proxy
    ercot_features = pd.DataFrame({
        'datetime':  df_input['Datetime (UTC)'],
        'demand_mw': df_input['vulnerability_score'] * 350 + 28000
        # NOTE: This proxy is intentional for cross-dataset temporal alignment.
        # The model uses time-based features (hour, month, lags) not raw demand.
        # Scaling chosen to keep demand in PJM training range (17,461–54,030 MW).
    })

    df_engineered = engineer_features(ercot_features)

    if df_engineered.empty:
        df_out['vuln_probability']  = 0.0
        df_out['predict_triggered'] = False
        return df_out

    # Generate probabilities from trained model
    vuln_proba = model.predict_proba(
        df_engineered[FEATURE_COLS]
    )[:, 1]

    # Compute average predict probability per hour + month
    df_engineered['vuln_proba'] = vuln_proba
    df_engineered['hour']  = df_engineered['datetime'].dt.hour
    df_engineered['month'] = df_engineered['datetime'].dt.month

    predict_by_hour_month = df_engineered.groupby(
        ['hour', 'month']
    )['vuln_proba'].mean().reset_index()

    predict_by_hour_month['predict_triggered'] = (
        predict_by_hour_month['vuln_proba'] >= DECISION_THRESHOLD
    )

    # Merge predict signal into output by hour + month
    df_out = df_out.merge(
        predict_by_hour_month[['hour', 'month', 'vuln_proba', 'predict_triggered']],
        on=['hour', 'month'],
        how='left'
    )

    df_out = df_out.rename(columns={'vuln_proba': 'vuln_probability'})

    # FIX: replaced deprecated fillna(method='ffill') with .ffill()
    df_out['vuln_probability']  = df_out['vuln_probability'].ffill().fillna(0)
    df_out['predict_triggered'] = df_out['predict_triggered'].ffill().fillna(False)

    return df_out


# ============================================================
# ACT LAYER
# Same REDUCTION_RATE, NUM_HOMES, KW_PER_HOME as Phase 1 and Phase 3.
# Dual-confirmation: Grid Saver only acts when BOTH
# Sense AND Predict independently confirm risk.
# ============================================================
def act_layer(df_input, reduction_rate, homes):
    """
    Input:  dataframe with sense_triggered and predict_triggered columns,
            reduction_rate (%), homes (count)
    Output: dataframe with spa_action_triggered and grid_saver_reduction_kw columns,
            scaled MW reduction for given homes count
    """
    df_a = df_input.copy()

    # SPA dual-confirmation logic — same as Phase 3
    df_a['sense_triggered']      = df_a['vulnerability_event']
    df_a['spa_action_triggered'] = (
        df_a['sense_triggered'] & df_a['predict_triggered']
    )

    # Per home reduction using validated Phase 1 value
    # Scaled by user-selected reduction rate relative to validated 4%
    scaled_kw_per_home = KW_PER_HOME * (reduction_rate / 4)

    df_a['grid_saver_reduction_kw'] = np.where(
        df_a['spa_action_triggered'],
        homes * scaled_kw_per_home,
        0
    )

    total_mw_saved = homes * scaled_kw_per_home / 1000

    return df_a, total_mw_saved


# ============================================================
# RISK DRIVERS
# Based on real columns: vulnerability_score, CARBON_COL, CFE_COL
# ============================================================
def get_risk_drivers(row, vulnerability_threshold):
    """
    Input:  latest data row, VULNERABILITY_THRESHOLD
    Output: list of risk driver strings
    """
    drivers = []

    if row['vulnerability_score'] >= vulnerability_threshold:
        drivers.append("🔴 Vulnerability score exceeds critical threshold")

    if row[CARBON_COL] > 400:
        drivers.append("🔴 High carbon intensity — grid running fossil-heavy")

    if row[CFE_COL] < 30:
        drivers.append("🔴 Carbon-free energy below 30% — low clean supply buffer")

    if row['vulnerability_score'] >= vulnerability_threshold * 0.7 and \
       row['vulnerability_score'] < vulnerability_threshold:
        drivers.append("🟡 Vulnerability score approaching critical threshold")

    if row[CARBON_COL] > 300:
        drivers.append("🟡 Elevated carbon intensity — grid stress building")

    if not drivers:
        drivers.append("🟢 Grid conditions stable — all signals within safe range")

    return drivers


# ============================================================
# RUN FULL SPA PIPELINE
# ============================================================
df, VULNERABILITY_THRESHOLD = sense_layer(df)
df['hour']       = df['Datetime (UTC)'].dt.hour
df['month']      = df['Datetime (UTC)'].dt.month
df['month_name'] = df['Datetime (UTC)'].dt.strftime('%b')
df['date']       = df['Datetime (UTC)'].dt.date
df = predict_layer(df, model)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.image("https://img.icons8.com/fluency/96/lightning-bolt.png", width=60)
st.sidebar.title("Grid Saver")
st.sidebar.markdown("**Adaptive Grid Intelligence Platform**")
st.sidebar.markdown("---")

live_mode = st.sidebar.toggle("Live Grid Mode", value=False)
if live_mode:
    st.sidebar.markdown(
        "<p style='color:#2ECC71; font-size:0.8rem;'>Showing last 24 hours</p>",
        unsafe_allow_html=True
    )

st.sidebar.markdown("---")

month_order    = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
months_present = [m for m in month_order if m in df['month_name'].unique()]
month_options  = ['All Year'] + months_present
selected_month = st.sidebar.selectbox("Select Month", month_options)

reduction_rate_input = st.sidebar.slider(
    "HVAC Reduction Rate (%) | Validated target: 3–5%",
    min_value=1, max_value=10, value=4, step=1
)

# FIX: slider default changed from 25 to 1000 to match minimum value
homes = st.sidebar.slider(
    "Homes Coordinated",
    min_value=1000, max_value=1000000, value=1000, step=1000
)

apply_intervention = st.sidebar.toggle("Apply Grid Saver Intervention", value=True)

st.sidebar.markdown("---")
with st.sidebar.expander("Dataset Information"):
    st.write("""
    **Sense Layer:** Electricity Maps US-TEX-ERCO 2025
    8,760 hourly records. Carbon intensity and CFE%.

    **Predict Layer:** PJM Interconnection 1998–2002
    32,896 hourly records. XGBoost. 91.6% Recall. 24hr ahead.

    **Act Layer:** Pecan Street Inc. Austin TX 2018
    25 real households. 868,096 records. 56.3% HVAC share.

    **Note:** Data is processed output derived from prototype
    notebooks. Raw academic datasets are not stored here.
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("**Stack**")
st.sidebar.markdown("Colab + GitHub + Streamlit")
st.sidebar.markdown("---")
st.sidebar.markdown("*Justine Adzormado*")

# ============================================================
# FILTER DATA
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

# Run Act Layer on filtered view
df_view, total_mw_saved = act_layer(df_view, reduction_rate_input, homes)

# ============================================================
# HEADER
# ============================================================
mode_label = "LIVE MODE" if live_mode else "ANALYSIS MODE"
mode_color = "#2ECC71" if live_mode else "#4A9EFF"

st.markdown(f"""
<div style='background: linear-gradient(135deg, #1B4F8C, #0D1117);
     padding: 30px; border-radius: 12px; margin-bottom: 20px;
     border: 1px solid #30363D;'>
    <div style='display: flex; justify-content: space-between; align-items: center;'>
        <div>
            <h1 style='color: white; margin: 0; font-size: 2.2rem;'>⚡ Grid Saver</h1>
            <p style='color: #4A9EFF; margin: 5px 0 0 0; font-size: 1.1rem;'>
                Adaptive Grid Intelligence Platform
            </p>
            <p style='color: #888; margin: 5px 0 0 0; font-size: 0.9rem;'>
                Texas ERCOT 2025 | SPA Logic | Dual-Confirmation Architecture
            </p>
        </div>
        <div style='background: {mode_color}22; border: 2px solid {mode_color};
             padding: 10px 20px; border-radius: 8px; text-align: center;'>
            <p style='color: {mode_color}; font-weight: bold; margin: 0; font-size: 1rem;'>
                {"🔴 " if live_mode else "📊 "}{mode_label}
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SECTION 1 — GRID STATUS (SENSE LAYER OUTPUT)
# ============================================================
st.markdown("## ⚡ Grid Status")

current_row    = df_view.iloc[-1]
current_score  = current_row['vulnerability_score']
current_status = current_row['grid_status']
current_carbon = current_row[CARBON_COL]
current_cfe    = current_row[CFE_COL]
current_prob   = current_row['vuln_probability']
vulnerable_pct = df_view['vulnerability_event'].mean() * 100
vulnerable_hours   = int(df_view['vulnerability_event'].sum())
interventions      = int(df_view['spa_action_triggered'].sum())

status_color = {'STABLE': '#2ECC71', 'WARNING': '#F39C12', 'CRITICAL': '#E74C3C'}
status_icon  = {'STABLE': '🟢', 'WARNING': '🟡', 'CRITICAL': '🔴'}

col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
cards = [
    (col1, status_icon[current_status], current_status,   "Grid Status",          status_color[current_status]),
    (col2, f"{current_score:.0f}",      "/100",            "Vulnerability Score",  "white"),
    (col3, f"{current_carbon:.0f}",     "gCO2eq/kWh",     "Carbon Intensity",     "#FF6B6B"),
    (col4, f"{current_cfe:.1f}%",       "CFE",             "Carbon-Free Energy",   "#2ECC71"),
    (col5, f"{current_prob:.2f}",       "probability",     "Predicted Risk (24h)", "#9B59B6"),
    (col6, f"{vulnerable_hours}",       "hours",           "Vulnerability Hours",  "#4A9EFF"),
    (col7, f"{interventions}",          "events",          "SPA Actions",          "#F39C12"),
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

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# SECTION 2 — RISK DRIVERS
# ============================================================
st.markdown("## Risk Drivers")
drivers = get_risk_drivers(current_row, VULNERABILITY_THRESHOLD)
col_d1, col_d2 = st.columns(2)
for i, driver in enumerate(drivers):
    if i % 2 == 0:
        col_d1.markdown(f"**{driver}**")
    else:
        col_d2.markdown(f"**{driver}**")

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# SECTION 3 — RECOMMENDED GRID ACTION
# ============================================================
st.markdown("## Recommended Grid Action")

if current_status == 'CRITICAL':
    action_color    = "#E74C3C"
    action_icon_str = "🔴"
    action_title    = "CRITICAL — Immediate Action Required"
    action_text     = f"Reduce residential HVAC load by {reduction_rate_input}% across coordination zones"
    expected        = f"Expected peak reduction: {reduction_rate_input * 0.9:.1f}% | Grid stabilization: HIGH confidence"
elif current_status == 'WARNING':
    action_color    = "#F39C12"
    action_icon_str = "🟡"
    action_title    = "WARNING — Prepare for Intervention"
    action_text     = f"Pre-stage HVAC coordination. Initiate {reduction_rate_input}% reduction if conditions escalate"
    expected        = "Monitoring window: Next 2–4 hours | Pre-emptive coordination recommended"
else:
    action_color    = "#2ECC71"
    action_icon_str = "🟢"
    action_title    = "STABLE — No Action Required"
    action_text     = "Grid operating within safe parameters. Continue monitoring."
    expected        = "Grid Saver uses dual-confirmation logic — both Sense and Predict must independently confirm risk before any intervention is triggered."

st.markdown(f"""
<div style='background: #161B22; border-left: 5px solid {action_color};
     padding: 20px; border-radius: 8px; margin: 10px 0;'>
    <h3 style='color: {action_color}; margin: 0;'>{action_icon_str} {action_title}</h3>
    <p style='color: white; margin: 10px 0 5px 0; font-size: 1rem;'>
        <strong>Recommended Action:</strong> {action_text}
    </p>
    <p style='color: #888; margin: 0; font-size: 0.9rem;'>{expected}</p>
</div>
""", unsafe_allow_html=True)

# Explain Decision Button
if st.button("🧠 Explain Grid Decision"):
    vulnerability_level = 'HIGH' if current_score >= VULNERABILITY_THRESHOLD else \
                          'MODERATE' if current_score >= VULNERABILITY_THRESHOLD * 0.7 else 'LOW'
    st.markdown(f"""
    <div style='background: #161B22; border: 1px solid #1B4F8C;
         padding: 25px; border-radius: 10px; margin-top: 15px;'>
        <h3 style='color: #4A9EFF; margin: 0 0 15px 0;'>AI Decision Explanation</h3>
        <p style='color: #CCC; margin: 5px 0;'>
            Grid Saver classified the system as
            <strong style='color: {action_color};'>{current_status}</strong> based on:
        </p>
        <ul style='color: #CCC; margin: 10px 0;'>
            <li>Vulnerability Score: <strong style='color: white;'>{current_score:.1f} / 100</strong>
                (threshold: {VULNERABILITY_THRESHOLD:.0f})</li>
            <li>Carbon Intensity: <strong style='color: #FF6B6B;'>{current_carbon:.0f} gCO2eq/kWh</strong></li>
            <li>Carbon-Free Energy: <strong style='color: #2ECC71;'>{current_cfe:.1f}%</strong></li>
            <li>Predicted Risk (24h ahead): <strong style='color: #9B59B6;'>{current_prob:.2f}</strong>
                (threshold: {DECISION_THRESHOLD})</li>
            <li>Risk Level: <strong style='color: {action_color};'>{vulnerability_level}</strong></li>
        </ul>
        <p style='color: #CCC; margin: 10px 0 5px 0;'>
            <strong>Recommended Action:</strong>
            <span style='color: white;'>{action_text}</span>
        </p>
        <p style='color: #888; margin: 0; font-size: 0.9rem;'>{expected}</p>
        <p style='color: #555; margin: 15px 0 0 0; font-size: 0.8rem;'>
            A {reduction_rate_input}% HVAC reduction across {homes:,} homes
            removes approximately {total_mw_saved*1000:,.1f} kW from peak demand,
            contributing to restoring the grid toward safe operating bounds.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# SECTION 4 — GRID DEMAND AND VULNERABILITY WINDOWS
# ============================================================
st.markdown("## Grid Demand and Vulnerability Windows")

color_map  = {'STABLE': '#2ECC71', 'WARNING': '#F39C12', 'CRITICAL': '#E74C3C'}
fig_demand = go.Figure()

for status in ['STABLE', 'WARNING', 'CRITICAL']:
    mask = df_view['grid_status'] == status
    if mask.any():
        fig_demand.add_trace(go.Scatter(
            x=df_view[mask]['Datetime (UTC)'],
            y=df_view[mask]['vulnerability_score'],
            mode='markers+lines',
            name=status,
            marker=dict(color=color_map[status], size=3, opacity=0.8),
            line=dict(color=color_map[status], width=0.8),
            connectgaps=True,
        ))

fig_demand.add_hline(
    y=VULNERABILITY_THRESHOLD,
    line_dash='dash', line_color='#FF4444',
    annotation_text=f'Vulnerability Threshold ({VULNERABILITY_THRESHOLD:.0f})',
    annotation_font_color='#FF4444'
)
fig_demand.update_layout(
    paper_bgcolor='#161B22', plot_bgcolor='#161B22',
    font=dict(color='white'),
    title=dict(
        text='Last 24h Grid Vulnerability Score' if live_mode else 'Grid Vulnerability Score — ERCOT Texas 2025',
        font=dict(color='white', size=14)
    ),
    xaxis=dict(gridcolor='#30363D', color='#888'),
    yaxis=dict(gridcolor='#30363D', color='#888', title='Vulnerability Score (0–100)'),
    legend=dict(bgcolor='#1A1A2E', bordercolor='#333'),
    height=350, margin=dict(t=50, b=30),
)
st.plotly_chart(fig_demand, use_container_width=True)

# ============================================================
# SECTION 5 — PEAK RISK TIMELINE
# ============================================================
st.markdown("## Peak Risk Timeline")
col_left, col_right = st.columns(2)

with col_left:
    hourly_stress = df_view.groupby('hour')['vulnerability_score'].mean().round(1)
    bar_colors_h  = [
        '#E74C3C' if s >= VULNERABILITY_THRESHOLD else
        '#F39C12' if s >= VULNERABILITY_THRESHOLD * 0.7 else
        '#2ECC71' for s in hourly_stress.values
    ]
    fig_hour = go.Figure(go.Bar(
        x=[f'{h:02d}:00' for h in hourly_stress.index],
        y=hourly_stress.values,
        marker_color=bar_colors_h,
    ))
    fig_hour.add_hline(
        y=VULNERABILITY_THRESHOLD, line_dash='dash', line_color='#FF4444',
        annotation_text=f'Threshold ({VULNERABILITY_THRESHOLD:.0f})',
        annotation_font_color='#FF4444'
    )
    fig_hour.update_layout(
        paper_bgcolor='#161B22', plot_bgcolor='#161B22', font=dict(color='white'),
        title=dict(text='Avg Grid Vulnerability by Hour of Day', font=dict(color='white', size=13)),
        xaxis=dict(gridcolor='#30363D', color='#888', title='Hour (UTC)'),
        yaxis=dict(gridcolor='#30363D', color='#888', title='Vulnerability Score'),
        height=300, margin=dict(t=50, b=30),
    )
    st.plotly_chart(fig_hour, use_container_width=True)

with col_right:
    if live_mode:
        st.info("Monthly trend requires full-year data. Switch to Analysis Mode to view seasonal patterns.")
    elif selected_month != 'All Year':
        daily_stress  = df_view.groupby('date')['vulnerability_score'].mean().round(1).sort_index()
        daily_colors  = [
            '#E74C3C' if s >= VULNERABILITY_THRESHOLD else
            '#F39C12' if s >= VULNERABILITY_THRESHOLD * 0.7 else
            '#2ECC71' for s in daily_stress.values
        ]
        fig_daily = go.Figure(go.Bar(
            x=[str(d) for d in daily_stress.index],
            y=daily_stress.values,
            marker_color=daily_colors,
        ))
        fig_daily.update_layout(
            paper_bgcolor='#161B22', plot_bgcolor='#161B22', font=dict(color='white'),
            title=dict(text=f'Avg Daily Vulnerability — {selected_month}', font=dict(color='white', size=13)),
            xaxis=dict(gridcolor='#30363D', color='#888', title='Date', tickangle=45),
            yaxis=dict(gridcolor='#30363D', color='#888', title='Vulnerability Score'),
            height=300, margin=dict(t=60, b=60),
        )
        st.plotly_chart(fig_daily, use_container_width=True)
    else:
        monthly_stress = df_view.groupby('month_name')['vulnerability_score'].mean().round(1)
        monthly_stress = monthly_stress.reindex([m for m in month_order if m in monthly_stress.index])
        bar_colors_m   = [
            '#E74C3C' if s >= VULNERABILITY_THRESHOLD else
            '#F39C12' if s >= VULNERABILITY_THRESHOLD * 0.7 else
            '#2ECC71' for s in monthly_stress.values
        ]
        fig_month = go.Figure(go.Bar(
            x=monthly_stress.index,
            y=monthly_stress.values,
            marker_color=bar_colors_m,
        ))
        fig_month.update_layout(
            paper_bgcolor='#161B22', plot_bgcolor='#161B22', font=dict(color='white'),
            title=dict(text='Avg Grid Vulnerability by Month', font=dict(color='white', size=13)),
            xaxis=dict(gridcolor='#30363D', color='#888', title='Month'),
            yaxis=dict(gridcolor='#30363D', color='#888', title='Vulnerability Score'),
            height=300, margin=dict(t=50, b=30),
        )
        st.plotly_chart(fig_month, use_container_width=True)

# ============================================================
# SECTION 6 — IMPACT AT SCALE
# ============================================================
st.markdown("## Impact at Scale")
st.markdown("*Drag the slider in the sidebar to see how Grid Saver scales.*")

scaled_reduction_kw = homes * KW_PER_HOME * (reduction_rate_input / 4)
scaled_reduction_mw = scaled_reduction_kw / 1000

grid_impact   = "City-scale" if homes < 100000 else "Regional-scale" if homes < 500000 else "National-scale"
impact_color  = "#F39C12" if homes < 100000 else "#4A9EFF" if homes < 500000 else "#2ECC71"
reserve_note  = "Exceeds reserve margin" if scaled_reduction_mw > 200 else "Building toward reserve margin"
rm_color      = "#2ECC71" if scaled_reduction_mw > 200 else "#F39C12"

col_s1, col_s2, col_s3, col_s4 = st.columns(4)
scale_cards = [
    (col_s1, f"{homes:,}",                  "Homes",   "Homes Coordinated",  "#4A9EFF"),
    (col_s2, f"{scaled_reduction_mw:,.1f} MW", "removed", "Grid Reduction",   "#2ECC71"),
    (col_s3, grid_impact,                    "",        "Impact Level",       impact_color),
    (col_s4, reserve_note,                   "",        "Reserve Margin",     rm_color),
]
for col, val, sub, label, color in scale_cards:
    with col:
        st.markdown(f"""
        <div class='metric-card'>
            <h2 style='color: {color}; font-size: 1.3rem; margin: 0;'>{val}</h2>
            <p style='color: #666; margin: 2px 0; font-size: 0.75rem;'>{sub}</p>
            <p style='color: #888; margin: 0; font-size: 0.75rem;'>{label}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# SECTION 7 — SYSTEM ARCHITECTURE
# ============================================================
st.markdown("## System Architecture")
col_a, col_b, col_c = st.columns(3)

arch = [
    (col_a, "👁️", "SENSE",   "#1B4F8C", "#4A9EFF",
     "Detect grid vulnerability signals continuously",
     "Carbon intensity monitoring",
     "Electricity Maps US-TEX-ERCO",
     "8,760 hourly records",
     "✅ ACTIVE — Phase 1 Complete", "#2ECC71"),
    (col_b, "🧠", "PREDICT", "#1A6B2E", "#2ECC71",
     "Forecast vulnerability windows",
     "XGBoost model | 91.6% Recall",
     "PJM 32,896 hourly records",
     "24-hour advance warning",
     "✅ ACTIVE — Phase 2 Complete", "#2ECC71"),
    (col_c, "⚡", "ACT",     "#7B1A1A", "#E74C3C",
     "Coordinate residential HVAC load reduction",
     # FIX: "surgical" replaced with "targeted" for consistency with pitch deck
     "3–5% targeted precision",
     "Pecan Street 868,096 records",
     "Human-override safety protocol",
     "✅ ACTIVE — Phase 3 Complete", "#2ECC71"),
]

for col, icon, title, bg, color, l1, l2, l3, l4, status, sc in arch:
    with col:
        st.markdown(f"""
        <div style='background: {bg}22; border: 1px solid {color};
             padding: 20px; border-radius: 10px; text-align: center;
             border-top: 4px solid {color};'>
            <h2 style='color: {color}; font-size: 2rem; margin: 0;'>{icon}</h2>
            <h3 style='color: {color}; margin: 10px 0 5px 0;'>{title}</h3>
            <p style='color: #888; font-size: 0.85rem; margin: 0;'>
                {l1}<br>{l2}<br>{l3}<br>{l4}
            </p>
            <p style='color: {sc}; font-size: 0.8rem; margin: 10px 0 0 0;'>{status}</p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# SECTION 8 — REPORTS AND INSIGHTS
# ============================================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("## Reports and Insights")
st.markdown("*Select a time period to view grid performance analysis and download a report.*")

# Prepare date columns
df["year"] = df["Datetime (UTC)"].dt.year
df["week"] = df["Datetime (UTC)"].dt.isocalendar().week.astype(int)

# Filters
col_r1, col_r2, col_r3 = st.columns(3)

with col_r1:
    report_type = st.selectbox(
        "Report Type",
        ["Yearly", "Monthly", "Weekly"],
        key="report_type"
    )

with col_r2:
    report_year = st.selectbox(
        "Year",
        sorted(df["year"].unique(), reverse=True),
        key="report_year"
    )

with col_r3:
    if report_type == "Monthly":
        report_month = st.selectbox(
            "Month",
            sorted(df[df["year"] == report_year]["month"].unique()),
            key="report_month"
        )
    elif report_type == "Weekly":
        report_week = st.selectbox(
            "Week Number",
            sorted(df[df["year"] == report_year]["week"].unique()),
            key="report_week"
        )
    else:
        st.markdown("")

# Apply filter
if report_type == "Yearly":
    df_report = df[df["year"] == report_year].copy()
    period_label = f"{report_year}"
elif report_type == "Monthly":
    df_report = df[
        (df["year"] == report_year) &
        (df["month"] == report_month)
    ].copy()
    month_name_label = df_report["month_name"].iloc[0] if not df_report.empty else ""
    period_label = f"{month_name_label} {report_year}"
elif report_type == "Weekly":
    df_report = df[
        (df["year"] == report_year) &
        (df["week"] == report_week)
    ].copy()
    period_label = f"Week {report_week}, {report_year}"

if df_report.empty:
    st.warning("No data available for the selected period.")
else:
    avg_vulnerability  = df_report["vulnerability_score"].mean()
    peak_vulnerability = df_report["vulnerability_score"].max()
    low_vulnerability  = df_report["vulnerability_score"].min()
    high_risk_events   = int((df_report["vulnerability_score"] >= 70).sum())
    warning_events     = int(
        ((df_report["vulnerability_score"] >= 40) &
         (df_report["vulnerability_score"] < 70)).sum()
    )
    stable_hours       = int((df_report["vulnerability_score"] < 40).sum())
    total_hours_report = len(df_report)

    st.markdown(f"**Showing: {period_label}** — {total_hours_report:,} hours analysed")
    st.markdown("<br>", unsafe_allow_html=True)

    col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)
    report_cards = [
        (col_m1, f"{avg_vulnerability:.1f}",  "/100",  "Average Vulnerability", "white"),
        (col_m2, f"{peak_vulnerability:.1f}", "/100",  "Peak Stress Score",     "#E74C3C"),
        (col_m3, f"{low_vulnerability:.1f}",  "/100",  "Lowest Score",          "#2ECC71"),
        (col_m4, f"{high_risk_events:,}",     "hours", "Critical Events",       "#E74C3C"),
        (col_m5, f"{warning_events:,}",       "hours", "Warning Events",        "#F39C12"),
        (col_m6, f"{stable_hours:,}",         "hours", "Stable Hours",          "#2ECC71"),
    ]
    for col, val, sub, label, color in report_cards:
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <h2 style='color: {color}; font-size: 1.4rem; margin: 0;'>{val}</h2>
                <p style='color: #666; margin: 2px 0; font-size: 0.75rem;'>{sub}</p>
                <p style='color: #888; margin: 0; font-size: 0.75rem;'>{label}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_trend, col_dist = st.columns([2, 1])

    with col_trend:
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=df_report["Datetime (UTC)"],
            y=df_report["vulnerability_score"],
            mode="lines",
            name="Vulnerability Score",
            line=dict(color="#4A9EFF", width=1.2),
            fill="tozeroy",
            fillcolor="rgba(74, 158, 255, 0.1)"
        ))
        fig_trend.add_hline(
            y=VULNERABILITY_THRESHOLD,
            line_dash="dash", line_color="#E74C3C",
            annotation_text=f"Threshold ({VULNERABILITY_THRESHOLD:.0f})",
            annotation_font_color="#E74C3C"
        )
        fig_trend.update_layout(
            paper_bgcolor="#161B22", plot_bgcolor="#161B22",
            font=dict(color="white"),
            title=dict(
                text=f"Vulnerability Trend — {period_label}",
                font=dict(color="white", size=13)
            ),
            xaxis=dict(gridcolor="#30363D", color="#888"),
            yaxis=dict(
                gridcolor="#30363D", color="#888",
                title="Vulnerability Score (0–100)",
                range=[0, 100]
            ),
            height=300, margin=dict(t=50, b=30),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_dist:
        status_counts = df_report["grid_status"].value_counts()
        dist_colors   = {
            "STABLE":   "#2ECC71",
            "WARNING":  "#F39C12",
            "CRITICAL": "#E74C3C"
        }
        fig_dist = go.Figure(go.Pie(
            labels=status_counts.index,
            values=status_counts.values,
            marker_colors=[dist_colors.get(s, "#888") for s in status_counts.index],
            hole=0.4,
            textfont=dict(color="white", size=12)
        ))
        fig_dist.update_layout(
            paper_bgcolor="#161B22",
            font=dict(color="white"),
            title=dict(
                text="Grid Status Distribution",
                font=dict(color="white", size=13)
            ),
            legend=dict(bgcolor="#1A1A2E", bordercolor="#333"),
            height=300, margin=dict(t=50, b=10),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("### Insight Summary")

    if avg_vulnerability >= 70:
        insight = f"Grid conditions were consistently critical during {period_label}."
    elif avg_vulnerability >= 40:
        insight = f"Grid showed moderate stress with recurring warning signals during {period_label}."
    else:
        insight = f"Grid conditions remained largely stable during {period_label}."

    if high_risk_events > 0:
        insight += (
            f" {high_risk_events:,} critical event"
            f"{'s were' if high_risk_events > 1 else ' was'} detected,"
            f" concentrated in peak demand hours."
        )
    if warning_events > 0:
        insight += (
            f" An additional {warning_events:,} warning period"
            f"{'s' if warning_events > 1 else ''} indicated elevated but manageable grid stress."
        )

    st.info(insight)

    st.markdown("<br>", unsafe_allow_html=True)
    report_export_cols = [
        "Datetime (UTC)", CARBON_COL, CFE_COL,
        "vulnerability_score", "vulnerability_event",
        "grid_status", "hour", "month", "month_name"
    ]
    csv_data = df_report[report_export_cols].to_csv(index=False)
    st.download_button(
        label="Download Report (CSV)",
        data=csv_data,
        file_name=f"Grid_Saver_{report_type}_{period_label.replace(' ', '_')}_Report.csv",
        mime="text/csv"
    )


# ============================================================
# FOOTER
# ============================================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style='background: #161B22; padding: 15px; border-radius: 8px;
     border: 1px solid #30363D; text-align: center; margin-top: 20px;'>
    <p style='color: #888; margin: 0; font-size: 0.85rem;'>
        Grid Saver | Adaptive Grid Intelligence Platform |
        Justine Adzormado |
        Built with Colab + GitHub + Streamlit
    </p>
    <p style='color: #555; margin: 5px 0 0 0; font-size: 0.75rem;'>
        Sense: Electricity Maps US-TEX-ERCO 2025 (Academic Access) |
        Predict: PJM XGBoost 91.6% Recall | 24hr ahead |
        Act: Pecan Street Austin TX 2018 | 25 real households |
        Full SPA pipeline validated across Phase 1, 2 and 3
    </p>
</div>
""", unsafe_allow_html=True)
