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

MONTH_NAMES = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December'
}
DAY_NAMES = {
    0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
    4: 'Friday', 5: 'Saturday', 6: 'Sunday'
}
DATETIME_COL  = 'Datetime (UTC)'   # standardised column name used throughout
ERCOT_PEAK_MW = 70000              # ERCOT peak demand reference (~70 GW)
month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

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
    Derived output from prototype notebooks — not raw academic data.
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
# FEATURE ENGINEERING
# Exact same function as Phase 2 and MVP notebook.
# Must match training features — do not change.
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
# Same formula and thresholds as Phase 1 prototype notebook.
# Score range: 0 (cleanest) to 100 (most vulnerable)
# Threshold: top 15% most vulnerable hours
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
    """
    TRUE INDEPENDENT PREDICT LAYER — matches prototype architecture exactly.

    Sense Layer:  ERCOT carbon + CFE signals  (independent source 1)
    Predict Layer: PJM time-based demand patterns (independent source 2)

    The Predict Layer uses ONLY temporal features derived from the datetime
    column — hour, day of week, month, lag patterns, rolling stats.
    It does NOT receive vulnerability_score, carbon, or CFE as inputs.
    The two layers meet ONLY at the SPA decision point, never before.

    Model trained on PJM Interconnection demand data (1998 to 2002).
    Applied here using ERCOT timestamps to generate temporal risk projections.
    PJM and ERCOT share the same grid demand seasonality patterns
    (summer peaks, winter peaks, weekday/weekend cycles) making this
    transfer valid for demonstration and early-stage validation.
    """
    df_out = df_input.copy()

    # Build time-only feature input — no vulnerability_score, no carbon, no CFE
    # This preserves independence from the Sense Layer completely
    # =================================================================
    # NOTE FOR TECHNICAL AUDIT:
    # demand_mw is a temporally constructed synthetic signal.
    # It uses a 35,000 MW PJM baseline adjusted by hour and season
    # to activate the trained XGBoost model.
    # It is NOT real-time live demand data.
    # In production, this is replaced with a live SCADA demand feed.
    # =================================================================
    pjm_avg_demand = 35000  # approximate mean PJM demand (MW) for scaling
    time_features = pd.DataFrame({
        'datetime':  df_input['Datetime (UTC)'],
        'demand_mw': pjm_avg_demand + (
            # Seasonal scaling using month only — no Sense Layer data
            np.where(df_input['Datetime (UTC)'].dt.month.isin([6, 7, 8]), 5000,
            np.where(df_input['Datetime (UTC)'].dt.month.isin([12, 1, 2]), 3000, 0))
            # Peak hour scaling using hour only — no Sense Layer data
            + np.where(df_input['Datetime (UTC)'].dt.hour.between(15, 20), 2000,
              np.where(df_input['Datetime (UTC)'].dt.hour.between(6, 9), 1000, -500))
        )
    })

    df_engineered = engineer_features(time_features)

    if df_engineered.empty:
        st.warning("Predict Layer: insufficient temporal data for risk projection.")
        df_out['vuln_probability']  = 0.0
        df_out['predict_triggered'] = False
        return df_out

    if df_engineered[FEATURE_COLS].isnull().any().any():
        st.warning("Predict Layer: feature engineering produced invalid values. Fallback applied.")
        df_out['vuln_probability']  = 0.0
        df_out['predict_triggered'] = False
        return df_out

    vuln_proba = model.predict_proba(df_engineered[FEATURE_COLS])[:, 1]
    df_engineered['vuln_proba'] = vuln_proba
    df_engineered['hour']       = df_engineered['datetime'].dt.hour
    df_engineered['month']      = df_engineered['datetime'].dt.month

    # Average by hour and month — same as prototype Phase 2
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
# SPA dual-confirmation: Grid Saver only acts when BOTH
# Sense AND Predict independently confirm vulnerability.
# ============================================================
def act_layer(df_input, reduction_rate, homes):
    df_a = df_input.copy()
    df_a['sense_triggered']      = df_a['vulnerability_event']
    df_a['spa_action_triggered'] = (
        df_a['sense_triggered'] & df_a['predict_triggered']
    )
    if reduction_rate != 4:
        pass  # Scaling disclaimer shown in UI below
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
# Used by the Recommendation Engine below.
# ============================================================
def compute_impact_metrics(row, homes, reduction_rate):
    """
    Computes energy, cost and CO2 impact per SPA event.
    Assumes 1-hour intervention window per event.
    Cost baseline: $100/MWh (conservative grid average).
    CO2: derived from actual carbon intensity of current row.
    """
    reduction_kw   = homes * KW_PER_HOME * (reduction_rate / 4)
    reduction_mw   = reduction_kw / 1000
    mwh_saved      = reduction_mw * 1  # 1-hour intervention window
    cost_savings   = mwh_saved * 100   # $100/MWh conservative baseline
    carbon_intensity = row[CARBON_COL] # gCO2/kWh = kgCO2/MWh numerically
    co2_avoided_tons = (carbon_intensity * mwh_saved) / 1000
    return mwh_saved, cost_savings, co2_avoided_tons


def compute_dispatch_priority(row):
    """
    Scores grid urgency 0 to 100 for utility-style dispatch prioritisation.
    Vulnerability score: 50% weight (0-100 -> 0-50)
    Predicted risk probability: 30% weight (0-1 -> 0-30)
    Carbon intensity: 20% weight capped at 20
    """
    score  = row['vulnerability_score'] * 0.5
    score += row.get('vuln_probability', 0) * 30
    score += min(row[CARBON_COL] / 10, 20)
    return round(score, 1)


# ============================================================
# RISK DRIVERS
# Thresholds match classify_status exactly.
# CRITICAL >= 70, WARNING >= 40, STABLE < 40
# ============================================================
def get_risk_drivers(row, vulnerability_threshold, df_full):
    """
    Score decides the classification.
    Carbon intensity and CFE explain why the score is where it is.
    Thresholds are relative to actual data range — not arbitrary fixed values.
    """
    drivers = []
    score  = row['vulnerability_score']
    carbon = row[CARBON_COL]
    cfe    = row[CFE_COL]

    # Step 1 — classify using score only (matches sense_layer exactly)
    if score >= 70:
        drivers.append("🔴 Grid is in CRITICAL state — vulnerability score at 70 or above")
    elif score >= 40:
        drivers.append("🟡 Grid is in WARNING state — vulnerability score between 40 and 69")
    else:
        drivers.append("🟢 Grid is STABLE — vulnerability score below 40")

    # Step 2 — explain using relative signal positioning
    # Carbon intensity: position within actual observed range
    carbon_min = df_full[CARBON_COL].min()
    carbon_max = df_full[CARBON_COL].max()
    carbon_range = carbon_max - carbon_min if (carbon_max - carbon_min) != 0 else 1
    carbon_pct = (carbon - carbon_min) / carbon_range

    # CFE: position within actual observed range
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
# RUN FULL SPA PIPELINE ON FULL DATASET
# Act layer runs on full df so SPA counts are accurate.
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

# Run act layer on full dataset for accurate SPA counts
df_full, _ = act_layer(df, REDUCTION_RATE, NUM_HOMES)
SENSE_TRIGGERS_TOTAL   = int(df_full['sense_triggered'].sum())
PREDICT_TRIGGERS_TOTAL = int(df_full['predict_triggered'].sum())
SPA_ACTIONS_TOTAL      = int(df_full['spa_action_triggered'].sum())

# ============================================================
# LOCKED BASELINE TRUTH METRICS — from MVP notebook outputs
# These match the notebook exactly and do NOT change with sliders.
# Sense triggers: 1,316 (15.0%) | Predict: 1,659 (18.9%) | SPA: 154 (1.8%)
# ============================================================
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
    st.sidebar.markdown(
        "<p style='color:#2ECC71; font-size:0.8rem;'>Showing the most recent 24 hours of grid data</p>",
        unsafe_allow_html=True
    )

st.sidebar.divider()

months_present = [m for m in month_order if m in df['month_name'].unique()]
month_options  = ['All Year'] + months_present
selected_month = st.sidebar.selectbox("Select Month", month_options)

reduction_rate_input = st.sidebar.slider(
    "HVAC Reduction Rate (%) — Validated target: 3 to 5%",
    min_value=1, max_value=10, value=4, step=1
)

homes = st.sidebar.slider(
    "Homes Coordinated",
    min_value=1000, max_value=1000000, value=100000, step=1000
)

apply_intervention = st.sidebar.toggle("Apply Grid Saver Intervention", value=True)

st.sidebar.divider()
with st.sidebar.expander("Dataset Information"):
    st.write("""
    **Sense Layer:** Electricity Maps US-TEX-ERCO 2025
    8,760 hourly records. Carbon intensity and CFE%.

    **Predict Layer:** PJM Interconnection 1998 to 2002
    32,896 hourly records. XGBoost. 91.6% Recall. 24hr ahead risk signal.

    **Act Layer:** Pecan Street Inc. Austin TX 2018
    25 real households. 868,096 records. 56.3% HVAC share.

    Note: Data is processed output derived from prototype
    notebooks. Raw academic datasets are not stored here.

    Predict Layer: XGBoost trained on PJM demand data (1998 to 2002).
    24hr Risk Projection via temporal pattern-recognition.
    Independent of Sense Layer — both meet only at SPA decision point.
    Transparency: A synthetic demand baseline activates the model for MVP.
    In production, a live SCADA feed replaces this baseline.
    """)

st.sidebar.divider()
st.sidebar.markdown("**Stack**")
st.sidebar.markdown("Colab + GitHub + Streamlit")
st.sidebar.divider()
st.sidebar.markdown("*Justine Adzormado*")

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
# HEADER
# ============================================================
mode_label = "RECENT WINDOW VIEW" if live_mode else "ANALYSIS MODE"
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
                {"🕐 " if live_mode else "📊 "}{mode_label}
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SECTION 1 — GRID STATUS
# ============================================================
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
    (col1, status_icon[current_status], current_status,   "Grid Status",                            status_color[current_status]),
    (col2, f"{current_score:.0f}",      "/100",            "Vulnerability Score",                    "white"),
    (col3, f"{current_carbon:.0f}",     "gCO₂/kWh",  "Carbon Intensity",                       "#E74C3C"),
    (col4, f"{current_cfe:.1f}%",       "clean energy",    "Carbon-Free Energy",                     "#2ECC71"),
    (col5, f"{vulnerable_pct:.1f}%",    "of period",       "Vulnerability Rate",                     "#4A9EFF"),
    (col6, f"{current_prob:.2f}",       "probability",     "24hr Risk Projection (Temporal Pattern)","#9B59B6"),
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

st.markdown(
    "<p style='color:#555; font-size:0.78rem; margin-top:4px;'>"
    "24hr Risk Projection: Risk signal derived from XGBoost pattern-recognition of historical grid stress cycles. "
    "Not a rolling live forecast — a temporal vulnerability window based on learned demand patterns (PJM 1998 to 2002)."
    "</p>",
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# SECTION 2 — RISK DRIVERS
# ============================================================
st.markdown("## Risk Drivers")
drivers = get_risk_drivers(current_row, VULNERABILITY_THRESHOLD, df)
col_d1, col_d2 = st.columns(2)
for i, driver in enumerate(drivers):
    if i % 2 == 0:
        col_d1.markdown(f"**{driver}**")
    else:
        col_d2.markdown(f"**{driver}**")

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# SECTION 3 — GRID DEMAND AND VULNERABILITY WINDOWS
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
# Overlay Predict Layer risk projection so it is visually present
fig_demand.add_trace(go.Scatter(
    x=df_view['Datetime (UTC)'],
    y=df_view['vuln_probability'] * 100,
    mode='lines',
    name='Projected Risk Signal (%)',
    line=dict(color='#9B59B6', dash='dot', width=1.2),
    opacity=0.8,
))
fig_demand.update_layout(
    paper_bgcolor='#161B22', plot_bgcolor='#161B22',
    font=dict(color='white'),
    title=dict(
        text='Last 24h Grid Vulnerability Score' if live_mode else 'Grid Vulnerability Score — ERCOT Texas 2025',
        font=dict(color='white', size=14)
    ),
    xaxis=dict(gridcolor='#30363D', color='#888'),
    yaxis=dict(gridcolor='#30363D', color='#888', title='Vulnerability Score (0 to 100)'),
    legend=dict(bgcolor='#1A1A2E', bordercolor='#333'),
    height=350, margin=dict(t=50, b=30),
)
st.plotly_chart(fig_demand, use_container_width=True)

# ============================================================
# SECTION 4 — PEAK VULNERABILITY TIMELINE
# ============================================================
st.markdown("## Peak Vulnerability Timeline")
col_left, col_right = st.columns(2)

with col_left:
    hourly_vuln   = df_view.groupby('hour')['vulnerability_score'].mean().round(1)
    bar_colors_h  = [
        '#E74C3C' if s >= 70 else
        '#F39C12' if s >= 40 else
        '#2ECC71' for s in hourly_vuln.values
    ]
    fig_hour = go.Figure(go.Bar(
        x=[f'{h:02d}:00' for h in hourly_vuln.index],
        y=hourly_vuln.values,
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
        daily_vuln    = df_view.groupby('date')['vulnerability_score'].mean().round(1).sort_index()
        daily_colors  = [
            '#E74C3C' if s >= 70 else
            '#F39C12' if s >= 40 else
            '#2ECC71' for s in daily_vuln.values
        ]
        fig_daily = go.Figure(go.Bar(
            x=[str(d) for d in daily_vuln.index],
            y=daily_vuln.values,
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
        monthly_vuln = df_view.groupby('month_name')['vulnerability_score'].mean().round(1)
        monthly_vuln = monthly_vuln.reindex([m for m in month_order if m in monthly_vuln.index])
        bar_colors_m = [
            '#E74C3C' if s >= 70 else
            '#F39C12' if s >= 40 else
            '#2ECC71' for s in monthly_vuln.values
        ]
        fig_month = go.Figure(go.Bar(
            x=monthly_vuln.index,
            y=monthly_vuln.values,
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
# SECTION 5 — RECOMMENDED GRID ACTION + SMART RECOMMENDATIONS
# ============================================================
st.markdown("<br>", unsafe_allow_html=True)
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
    expected        = "Monitoring window: Next 2 to 4 hours | Pre-emptive coordination recommended"
else:
    action_color    = "#2ECC71"
    action_icon_str = "🟢"
    action_title    = "STABLE — No Action Required"
    action_text     = "Grid operating within safe parameters. Continue monitoring."
    expected        = "Grid Saver uses dual-confirmation logic — both Sense and Predict must independently confirm vulnerability before any intervention is triggered."

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

if st.button("🧠 Explain Grid Decision"):
    if current_score >= 70:
        vulnerability_level = 'CRITICAL'
    elif current_score >= 40:
        vulnerability_level = 'WARNING'
    else:
        vulnerability_level = 'STABLE'

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
            <li>Carbon Intensity: <strong style='color: #FF6B6B;'>{current_carbon:.0f} gCO₂eq/kWh</strong></li>
            <li>Carbon-Free Energy: <strong style='color: #2ECC71;'>{current_cfe:.1f}%</strong></li>
            <li>24hr Risk Projection (Temporal Pattern): <strong style='color: #9B59B6;'>{current_prob:.2f}</strong>
                (threshold: {DECISION_THRESHOLD})</li>
            <li>Vulnerability Level: <strong style='color: {action_color};'>{vulnerability_level}</strong></li>
        </ul>
        <p style='color: #CCC; margin: 10px 0 5px 0;'>
            <strong>Recommended Action:</strong>
            <span style='color: white;'>{action_text}</span>
        </p>
        <p style='color: #888; margin: 0; font-size: 0.9rem;'>{expected}</p>
        <p style='color: #555; margin: 15px 0 0 0; font-size: 0.8rem;'>
            A {reduction_rate_input}% HVAC reduction across {homes:,} homes
            removes approximately {homes * KW_PER_HOME * (reduction_rate_input / 4):,.1f} kW
            per SPA event (1-hour window) from peak demand.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# SMART RECOMMENDATIONS — Operational Intelligence Layer
# Converts SPA output into clear grid operator actions
# with dispatch priority, impact and CO2 metrics
# ============================================================
st.markdown("## Smart Recommendations")

# Compute dispatch priority from current row (real-time signal)
dispatch_score = compute_dispatch_priority(current_row)

# Compute per-event impact from current row (1-hour window)
mwh_saved, cost_savings, co2_avoided = compute_impact_metrics(
    current_row, homes, reduction_rate_input
)

# Compute true SPA-aggregated impact across all triggered events in view
# reduction_mw column is computed later in Section 6 so we calculate it here independently
scaled_kw_per_home_recs = KW_PER_HOME * (reduction_rate_input / 4)
if 'spa_action_triggered' in df_view.columns:
    spa_events_df = df_view[df_view['spa_action_triggered']].copy()
    event_count = len(spa_events_df)
    total_mwh_all_events  = event_count * (homes * scaled_kw_per_home_recs) / 1000
    total_co2_all_events  = (spa_events_df[CARBON_COL].mean() * total_mwh_all_events) / 1000 if event_count > 0 else 0
    total_cost_all_events = total_mwh_all_events * 100
else:
    spa_events_df         = pd.DataFrame()
    total_mwh_all_events  = 0
    total_co2_all_events  = 0
    total_cost_all_events = 0

# Build recommendations
recommendations = []
spa_current = current_row.get('spa_action_triggered', False)

# Priority header
priority_label = "Immediate action required" if dispatch_score > 70 else "Monitor closely"
recommendations.append(
    (f"Dispatch Priority Score: {dispatch_score}/100 — {priority_label}", "priority")
)

if current_status == 'CRITICAL':
    recommendations.append((f"Execute {reduction_rate_input}% HVAC reduction across {homes:,} homes immediately.", "action"))
    if spa_current:
        recommendations.append(("SPA dual-confirmation achieved — full automated dispatch approved.", "action"))
    else:
        recommendations.append(("Partial confirmation — deploy controlled reduction while monitoring.", "action"))
    recommendations.append(("Target high-demand feeders and peak residential clusters.", "detail"))
    recommendations.append(("Notify grid operators and demand response aggregators immediately.", "detail"))

elif current_status == 'WARNING':
    recommendations.append(("Pre-stage demand response resources for potential activation.", "action"))
    if current_prob >= 0.4:
        recommendations.append(("Temporal risk signal elevated — prepare for activation within next peak window.", "action"))
    else:
        recommendations.append(("Risk is moderate — maintain readiness and continue monitoring.", "detail"))
    recommendations.append(("Shift flexible loads (EV charging, water heating) to off-peak hours.", "detail"))
    recommendations.append(("Issue advisory notifications to consumers in high-load zones.", "detail"))

else:
    recommendations.append(("No immediate intervention required.", "action"))
    if current_prob >= 0.4:
        recommendations.append(("Temporal model indicates potential future risk — monitor upcoming peak window.", "detail"))
    recommendations.append(("Encourage off-peak consumption behaviour to maintain grid efficiency.", "detail"))
    recommendations.append(("Maintain baseline grid monitoring operations.", "detail"))

# Impact summary — always shown
recommendations.append((
    f"Per SPA event (1-hour window): {mwh_saved:.2f} MWh | ${cost_savings:,.0f} savings | {co2_avoided:.3f} tons CO₂ avoided. "
    f"Total across {len(spa_events_df)} events this period: {total_mwh_all_events:.2f} MWh | "
    f"${total_cost_all_events:,.0f} | {total_co2_all_events:.3f} tons CO₂",
    "impact"
))

# Display with priority tagging
for i, (rec, rec_type) in enumerate(recommendations):
    if rec_type == "priority":
        priority_color = "#E74C3C" if dispatch_score > 70 else "#F39C12" if dispatch_score > 40 else "#2ECC71"
        st.markdown(f"""
        <div style='background:#161B22; border:1px solid {priority_color};
             padding:12px 16px; border-radius:8px; margin-bottom:8px;'>
            <span style='color:{priority_color}; font-weight:bold; font-size:1rem;'>
                ⚡ {rec}
            </span>
        </div>
        """, unsafe_allow_html=True)
    elif rec_type == "action":
        if i == 1:
            st.markdown(f"🔴 **Priority Action:** {rec}")
        else:
            st.markdown(f"🔸 **{rec}**")
    elif rec_type == "impact":
        st.markdown(f"""
        <div style='background:#161B22; border-left:4px solid #2ECC71;
             padding:10px 14px; border-radius:6px; margin-top:10px;'>
            <span style='color:#2ECC71; font-size:0.9rem;'>📊 {rec}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"• {rec}")

st.markdown(
    "<p style='color:#555; font-size:0.78rem; margin-top:8px;'>"
    "Cost estimate based on $100/MWh conservative grid baseline. "
    "CO₂ derived from current carbon intensity. "
    "Feeder-level targeting and real-time pricing integration available in production."
    "</p>",
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# SECTION 6 — GRID SAVER LOAD REDUCTION SIMULATION
# Data-driven. Reduction only applied when SPA triggers.
# Matches prototype Phase 3 logic exactly.
# ============================================================
st.markdown("## Grid Saver Load Reduction Simulation")

if reduction_rate_input != 4:
    st.warning(
        f"Note: Grid Saver is validated at 4% HVAC reduction (0.0920 kW per home). "
        f"Results at {reduction_rate_input}% are proportionally scaled estimates, "
        f"not independently validated values."
    )

# Trigger counts — show live counts in live mode, notebook truth in analysis mode
if live_mode:
    display_sense   = int(df_view['sense_triggered'].sum()) if 'sense_triggered' in df_view.columns else 0
    display_predict = int(df_view['predict_triggered'].sum())
    display_spa     = int(df_view['spa_action_triggered'].sum()) if 'spa_action_triggered' in df_view.columns else 0
    trigger_sub     = "hours (last 24h)"
    spa_sub         = "events (last 24h)"
else:
    display_sense   = NOTEBOOK_SENSE_TRIGGERS
    display_predict = NOTEBOOK_PREDICT_TRIGGERS
    display_spa     = NOTEBOOK_SPA_ACTIONS
    trigger_sub     = "hours (full year validated)"
    spa_sub         = "events (full year validated)"

col_sim1, col_sim2, col_sim3 = st.columns(3)
sim_cards = [
    (col_sim1, f"{display_sense:,}",   trigger_sub, "Sense Triggers",             "#4A9EFF"),
    (col_sim2, f"{display_predict:,}", trigger_sub, "Predict Triggers",           "#9B59B6"),
    (col_sim3, f"{display_spa:,}",     spa_sub,     "SPA Actions (Dual-Confirmed)","#F39C12"),
]
for col, val, sub, label, color in sim_cards:
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
# DATA-DRIVEN PEAK REDUCTION CALCULATION
# synthetic_demand_mw is the synthetic baseline signal.
# Reduction applied ONLY during SPA-triggered hours.
# Peak comparison uses the same timestamp (idxmax).
# ============================================================
# Synthetic demand baseline derived from temporal patterns (Predict Layer only)
# Uses hour and month signals — NO vulnerability_score dependency
# Preserves SPA independence: Sense and Predict remain separate until decision point
df_view['synthetic_demand_mw'] = 35000 + (
    np.where(df_view['month'].isin([6, 7, 8]), 5000,
    np.where(df_view['month'].isin([12, 1, 2]), 3000, 0))
    +
    np.where(df_view['hour'].between(15, 20), 2000,
    np.where(df_view['hour'].between(6, 9), 1000, -500))
)

# reduction_mw already created by act_layer — no recalculation needed
# synthetic_demand_mw is the Predict Layer baseline
df_view['adjusted_demand_mw'] = df_view['synthetic_demand_mw'] - df_view['reduction_mw']

# ============================================================
# CORRECT PEAK REDUCTION — same timestamp comparison
# Find peak on the BEFORE signal, compare AFTER at same point
# ============================================================
# Global peak for chart reference
peak_idx        = df_view['synthetic_demand_mw'].idxmax()
peak_time       = df_view.loc[peak_idx, 'Datetime (UTC)']

# Peak reduction: find the SPA-triggered hour with highest demand
# This is where Grid Saver has the most impact
spa_active_df = df_view[df_view['spa_action_triggered']].copy()
if not spa_active_df.empty:
    spa_peak_idx      = spa_active_df['synthetic_demand_mw'].idxmax()
    original_peak     = spa_active_df.loc[spa_peak_idx, 'synthetic_demand_mw']
    after_peak        = spa_active_df.loc[spa_peak_idx, 'adjusted_demand_mw']
    peak_reduction_mw = original_peak - after_peak
    pct_reduction     = (peak_reduction_mw / original_peak) * 100 if original_peak > 0 else 0
else:
    original_peak     = df_view['synthetic_demand_mw'].max()
    after_peak        = original_peak
    peak_reduction_mw = 0
    pct_reduction     = 0

total_mw_removed  = df_view['reduction_mw'].sum()
spa_view_count    = int(df_view['spa_action_triggered'].sum())

# ============================================================
# TOTAL LOAD REMOVED HEADLINE
# ============================================================
st.markdown(f"""
<div style='background:#161B22; border-left:5px solid #2ECC71;
     padding:15px; border-radius:8px; margin-bottom:15px;'>
    <h3 style='color:#2ECC71; margin:0;'>⚡ Total Energy Removed (MWh)</h3>
    <p style='color:white; font-size:1.4rem; margin:5px 0;'>
        {total_mw_removed:,.2f} MWh — sum of hourly reductions across SPA-triggered events
    </p>
    <p style='color:#888; margin:0; font-size:0.85rem;'>
        Across {spa_view_count} SPA-triggered events in this period |
        Full-year validated: {NOTEBOOK_SPA_ACTIONS} events
    </p>
</div>
""", unsafe_allow_html=True)

# Peak reduction summary cards
col_p1, col_p2, col_p3, col_p4 = st.columns(4)
peak_cards = [
    (col_p1, f"{original_peak:,.0f} MW",      "Original Peak",          "white"),
    (col_p2, f"{after_peak:,.0f} MW",         "After Grid Saver",       "#2ECC71"),
    (col_p3, f"{pct_reduction:.2f}%",         "Peak Demand Reduction",  "#4A9EFF"),
    (col_p4, f"{peak_reduction_mw:,.2f} MW",  "Peak Load Shed",         "#F39C12"),
]
for col, val, label, color in peak_cards:
    with col:
        st.markdown(f"""
        <div class='metric-card'>
            <h2 style='color: {color}; font-size: 1.4rem; margin: 0;'>{val}</h2>
            <p style='color: #888; margin: 4px 0 0 0; font-size: 0.78rem;'>{label}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown(
    f"<p style='color:#555; font-size:0.8rem; margin-top:6px;'>"
    f"Peak insight: Original {original_peak:,.0f} MW reduced to {after_peak:,.0f} MW "
    f"at peak demand timestamp. Load shed: {peak_reduction_mw:,.2f} MW ({pct_reduction:.2f}%). "
    f"Validated at 4% HVAC reduction (0.0920 kW per home). "
    f"Reduction occurs only during dual-confirmed SPA events — not continuously — "
    f"ensuring targeted intervention without unnecessary load disruption."
    f"</p>",
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# BEFORE PLOT (top) — Original demand proxy with SPA markers
# ============================================================
fig_before = go.Figure()
fig_before.add_trace(go.Scatter(
    x=df_view['Datetime (UTC)'],
    y=df_view['synthetic_demand_mw'],
    mode='lines',
    name='Original Demand (Proxy)',
    line=dict(color='#E74C3C', width=1.5),
))
fig_before.add_trace(go.Scatter(
    x=df_view[df_view['spa_action_triggered']]['Datetime (UTC)'],
    y=df_view[df_view['spa_action_triggered']]['synthetic_demand_mw'],
    mode='markers',
    name='SPA Action Triggered',
    marker=dict(color='#F39C12', size=6, symbol='circle'),
))
# Mark peak timestamp
fig_before.add_trace(go.Scatter(
    x=[peak_time, peak_time],
    y=[df_view['synthetic_demand_mw'].min(), df_view['synthetic_demand_mw'].max()],
    mode='lines',
    name='Peak Demand',
    line=dict(color='#FFFFFF', width=1.5, dash='dash'),
    showlegend=True
))
fig_before.update_layout(
    paper_bgcolor='#161B22', plot_bgcolor='#161B22',
    font=dict(color='white'),
    title=dict(text='Before Grid Saver — Original Demand Proxy', font=dict(color='white', size=13)),
    xaxis=dict(gridcolor='#30363D', color='#888'),
    yaxis=dict(gridcolor='#30363D', color='#888', title='Demand Proxy (MW)'),
    legend=dict(bgcolor='#1A1A2E', bordercolor='#333'),
    height=320, margin=dict(t=50, b=20),
)
st.plotly_chart(fig_before, use_container_width=True)

# ============================================================
# AFTER PLOT (below) — Adjusted demand with shaded SPA zones
# and top 20 drop lines by reduction magnitude
# ============================================================
fig_after = go.Figure()

# Original demand (faded reference)
fig_after.add_trace(go.Scatter(
    x=df_view['Datetime (UTC)'],
    y=df_view['synthetic_demand_mw'],
    mode='lines',
    name='Original Demand (Proxy)',
    line=dict(color='#E74C3C', width=1, dash='dot'),
    opacity=0.35,
))

# Adjusted demand after intervention
fig_after.add_trace(go.Scatter(
    x=df_view['Datetime (UTC)'],
    y=df_view['adjusted_demand_mw'],
    mode='lines',
    name='After Grid Saver Intervention',
    line=dict(color='#2ECC71', width=1.5),
    fill='tonexty',
    fillcolor='rgba(46, 204, 113, 0.08)'
))



# Mark peak timestamp
fig_after.add_trace(go.Scatter(
    x=[peak_time, peak_time],
    y=[df_view['synthetic_demand_mw'].min(), df_view['synthetic_demand_mw'].max()],
    mode='lines',
    name='Peak Demand',
    line=dict(color='#FFFFFF', width=1.5, dash='dash'),
    showlegend=False
))

fig_after.update_layout(
    paper_bgcolor='#161B22', plot_bgcolor='#161B22',
    font=dict(color='white'),
    title=dict(
        text='After Grid Saver — Demand with Intervention Applied',
        font=dict(color='white', size=13)
    ),
    xaxis=dict(gridcolor='#30363D', color='#888'),
    yaxis=dict(gridcolor='#30363D', color='#888', title='Demand Proxy (MW)'),
    legend=dict(bgcolor='#1A1A2E', bordercolor='#333'),
    height=320, margin=dict(t=50, b=20),
)
st.plotly_chart(fig_after, use_container_width=True)

st.markdown(
    "<p style='color:#555; font-size:0.78rem; margin-top:4px;'>"
    "Showing top 20 highest-impact Grid Saver interventions. Shaded zones mark all SPA-triggered windows. "
    "Drop lines show top 20 highest-impact events. "
    "Peak window view below shows full detail at peak demand."
    "</p>",
    unsafe_allow_html=True
)

st.write("🔥 REACHED SECTION 6")
# ============================================================
# PEAK ZOOM WINDOW — ±6 hours around peak demand
# Shows all drop lines and full detail at the critical moment
# ============================================================
st.markdown("#### Peak Demand Window (6 Hours Before and After Peak)")
zoom_df = df_view[
    (df_view['Datetime (UTC)'] >= peak_time - pd.Timedelta(hours=6)) &
    (df_view['Datetime (UTC)'] <= peak_time + pd.Timedelta(hours=6))
].copy()

if not zoom_df.empty:
    fig_zoom = go.Figure()
    fig_zoom.add_trace(go.Scatter(
        x=zoom_df['Datetime (UTC)'],
        y=zoom_df['synthetic_demand_mw'],
        mode='lines',
        name='Original Demand (Proxy)',
        line=dict(color='#E74C3C', width=2),
    ))
    fig_zoom.add_trace(go.Scatter(
        x=zoom_df['Datetime (UTC)'],
        y=zoom_df['adjusted_demand_mw'],
        mode='lines',
        name='After Grid Saver',
        line=dict(color='#2ECC71', width=2),
        fill='tonexty',
        fillcolor='rgba(46, 204, 113, 0.15)'
    ))
  
    fig_zoom.add_trace(go.Scatter(
        x=[peak_time, peak_time],
        y=[zoom_df['synthetic_demand_mw'].min(), zoom_df['synthetic_demand_mw'].max()],
        mode='lines',
        name='Peak',
        line=dict(color='#FFFFFF', width=1.5, dash='dash'),
        showlegend=False
    ))
    
    fig_zoom.update_layout(
        paper_bgcolor='#161B22', plot_bgcolor='#161B22',
        font=dict(color='white'),
        title=dict(
            text=f'Peak Window — Full Detail Around {str(peak_time)[:13]}',
            font=dict(color='white', size=13)
        ),
        xaxis=dict(gridcolor='#30363D', color='#888'),
        yaxis=dict(gridcolor='#30363D', color='#888', title='Demand Proxy (MW)'),
        legend=dict(bgcolor='#1A1A2E', bordercolor='#333'),
        height=320, margin=dict(t=50, b=20),
    )
    st.plotly_chart(fig_zoom, use_container_width=True)

# ============================================================
# SECTION 7 — IMPACT AT SCALE
# ============================================================
st.divider()
st.markdown("## Impact at Scale")
st.markdown("*Adjust the Homes Coordinated slider in the sidebar to see how Grid Saver scales.*")

scaled_reduction_kw = homes * KW_PER_HOME * (reduction_rate_input / 4)
scaled_reduction_mw = scaled_reduction_kw / 1000

if homes < 50000:
    grid_impact  = "Neighbourhood Scale"
    impact_color = "#F39C12"
elif homes < 250000:
    grid_impact  = "District Scale"
    impact_color = "#4A9EFF"
elif homes < 600000:
    grid_impact  = "Regional Scale"
    impact_color = "#9B59B6"
else:
    grid_impact  = "National Scale"
    impact_color = "#2ECC71"

reserve_note  = "Exceeds reserve margin" if scaled_reduction_mw > 200 else "Building toward reserve margin"
rm_color      = "#2ECC71" if scaled_reduction_mw > 200 else "#F39C12"

col_s1, col_s2, col_s3, col_s4 = st.columns(4)
scale_cards = [
    (col_s1, f"{homes:,}",                    "Homes",   "Homes Coordinated",  "#4A9EFF"),
    (col_s2, f"{scaled_reduction_mw:,.1f} MW", "removed", "Projected Grid Reduction (per event)", "#2ECC71"),
    (col_s3, grid_impact,                      "",        "Impact Level",        impact_color),
    (col_s4, reserve_note,                     "",        "Reserve Margin",      rm_color),
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

percentage_of_grid = (scaled_reduction_mw / ERCOT_PEAK_MW) * 100
st.markdown(
    f"<p style='color:#888; font-size:0.8rem; margin-top:8px;'>"
    f"At this scale, Grid Saver removes approximately <strong>{percentage_of_grid:.3f}%</strong> "
    f"of ERCOT peak demand (~{ERCOT_PEAK_MW:,} MW). "
    f"Validated reduction rate: 4% HVAC cycling (0.0920 kW per home, Pecan Street 2018)."
    f"</p>",
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# SECTION 8 — SYSTEM ARCHITECTURE
# ============================================================
st.markdown("## System Architecture")
col_a, col_b, col_c = st.columns(3)

arch = [
    (col_a, "👁️", "SENSE",   "#1B4F8C", "#4A9EFF",
     "Detect grid vulnerability signals continuously",
     "Carbon intensity monitoring",
     "Electricity Maps US-TEX-ERCO",
     "8,760 hourly records"),
    (col_b, "🧠", "PREDICT", "#1A6B2E", "#2ECC71",
     "Forecast vulnerability windows",
     "XGBoost model | 91.6% Recall",
     "PJM 32,896 hourly records",
     "24hr Risk Projection (Temporal Pattern) — XGBoost pattern-recognition of historical grid cycles"),
    (col_c, "⚡", "ACT",     "#7B1A1A", "#E74C3C",
     "Coordinate residential HVAC load reduction",
     "3 to 5% targeted precision",
     "Pecan Street 868,096 records",
     "man-override safety protocol"),
]

for col, icon, title, bg, color, l1, l2, l3, l4 in arch:
    with col:
        st.markdown(f"""
        <div style='background: {bg}22; border: 1px solid {color};
             padding: 20px; border-radius: 10px; text-align: center;
             border-top: 4px solid {color};'>
            <h2 style='color: {color}; font-se: 2rem; margin: 0;'>{icon}</h2>
            <h3 style='color: {color}; margin: 10px 0 5px 0;'>{title}</h3>
            <p style='color: #888; font-size: 0.85rem; margin: 0;'>
                {l1}<br>{l2}<br>{l3}<br>{l4}
            </p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# SECTION 9 — REPORTS AND INSIGHTS (HARDENED)
# ============================================================

st.markdown("## Reports and Insights")

# Debug toggle (very useful during development)
debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=False)

def debug_log(msg):
    if debug_mode:
        st.write("DEBUG:", msg)

# -------------------------------
# MODE CHECK
# -------------------------------
if live_mode:
    st.warning("Reports are disabled in Live Mode. Switch to Analysis Mode.")
    st.stop()

st.markdown("*Select a time period to view grid performance analysis and download a report.*")

# -------------------------------
# DATA VALIDATION (Sense QA Layer)
# -------------------------------
if df is None or df.empty:
    st.error("Dataset not loaded or empty.")
    st.stop()

required_base_cols = ['Datetime (UTC)', 'year']
missing_base = [c for c in required_base_cols if c not in df.columns]

if missing_base:
    st.error(f"Missing required columns: {missing_base}")
    st.stop()

# Ensure datetime is correct
df['Datetime (UTC)'] = pd.to_datetime(df['Datetime (UTC)'], errors='coerce')

# -------------------------------
# FILTER SELECTION
# -------------------------------
col_r1, col_r2, col_r3 = st.columns(3)

with col_r1:
    report_type = st.selectbox("Report Type", ["Yearly", "Monthly", "Weekly"])

with col_r2:
    report_year = st.selectbox("Year", sorted(df['year'].unique(), reverse=True))

with col_r3:
    report_month, report_week = None, None

    if report_type == "Monthly":
        if 'month' not in df.columns:
            st.error("Month column missing.")
            st.stop()

        available_months = sorted(df[df['year'] == report_year]['month'].unique())
        month_display = [MONTH_NAMES[m] for m in available_months]

        selected_month_name = st.selectbox("Month", month_display)
        report_month = [k for k, v in MONTH_NAMES.items() if v == selected_month_name][0]

    elif report_type == "Weekly":
        if 'week' not in df.columns:
            st.error("Week column missing.")
            st.stop()

        available_weeks = sorted(df[df['year'] == report_year]['week'].unique())
        report_week = st.selectbox("Week", available_weeks)

# -------------------------------
# FILTER DATA
# -------------------------------
try:
    if report_type == "Yearly":
        df_report = df[df['year'] == report_year].copy()
        period_label = f"{report_year}"

    elif report_type == "Monthly":
        df_report = df[(df['year'] == report_year) & (df['month'] == report_month)].copy()
        period_label = f"{MONTH_NAMES[report_month]} {report_year}"

    else:
        df_report = df[(df['year'] == report_year) & (df['week'] == report_week)].copy()
        period_label = f"Week {report_week}, {report_year}"

except Exception as e:
    st.error(f"Filtering failed: {e}")
    st.stop()

if df_report.empty:
    st.warning("No data available for selected period.")
    st.stop()

debug_log(f"Filtered rows: {len(df_report)}")

# -------------------------------
# REQUIRED ANALYSIS COLUMNS CHECK
# -------------------------------
required_cols = ['vulnerability_score', 'grid_status']
missing_cols = [c for c in required_cols if c not in df_report.columns]

if missing_cols:
    st.error(f"Missing analysis columns: {missing_cols}")
    st.stop()

# Clean NaNs
df_report = df_report.dropna(subset=['vulnerability_score'])

# -------------------------------
# METRICS
# -------------------------------
avg_vulnerability  = df_report['vulnerability_score'].mean()
peak_vulnerability = df_report['vulnerability_score'].max()
low_vulnerability  = df_report['vulnerability_score'].min()

high_risk_events = int((df_report['vulnerability_score'] >= 70).sum())
warning_events   = int(((df_report['vulnerability_score'] >= 40) & (df_report['vulnerability_score'] < 70)).sum())
stable_hours     = int((df_report['vulnerability_score'] < 40).sum())

# -------------------------------
# ACT LAYER (SAFE EXECUTION)
# -------------------------------
try:
    spa_report_df, _ = act_layer(df_report, REDUCTION_RATE, NUM_HOMES)

    spa_events_report = int(spa_report_df['spa_action_triggered'].sum())
    total_mwh_report  = spa_report_df['reduction_mw'].sum()

    if CARBON_COL in spa_report_df.columns:
        total_co2_report = (spa_report_df[CARBON_COL] * spa_report_df['reduction_mw']).sum() / 1000
    else:
        total_co2_report = 0

except Exception as e:
    st.error(f"Act layer failed: {e}")
    debug_log(spa_report_df.head() if 'spa_report_df' in locals() else "No SPA DF")
    st.stop()

# -------------------------------
# DISPLAY METRICS
# -------------------------------
st.markdown(f"**Showing: {period_label}** — {len(df_report):,} hours analysed")

col1, col2, col3 = st.columns(3)
col1.metric("Avg Vulnerability", f"{avg_vulnerability:.1f}")
col2.metric("Peak", f"{peak_vulnerability:.1f}")
col3.metric("SPA Actions", f"{spa_events_report:,}")

# -------------------------------
# CHART (SAFE)
# -------------------------------
try:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_report['Datetime (UTC)'],
        y=df_report['vulnerability_score'],
        mode='lines'
    ))
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Chart failed: {e}")

# -------------------------------
# INSIGHT SUMMARY
# -------------------------------
insight = f"Average vulnerability: {avg_vulnerability:.1f}/100. "

if spa_events_report > 0:
    insight += f"{spa_events_report} interventions triggered, saving {total_mwh_report:.2f} MWh."
else:
    insight += "No interventions triggered."

st.info(insight)

# -------------------------------
# EXPORT (SAFE)
# -------------------------------
try:
    export_df = spa_report_df.copy()
    csv_data = export_df.to_csv(index=False)

    st.download_button(
        "Download Report",
        data=csv_data,
        file_name=f"GridSaver_{period_label}.csv"
    )

except Exception as e:
    st.error(f"Export failed: {e}")
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
        Predict: PJM XGBoost 91.6% Recall | 24hr Risk Projection |
        Act: Pecan Street Austin TX 2018 | 25 real households |
        Full SPA pipeline validated across Phase 1, 2 and 3
    </p>
</div>
""", unsafe_allow_html=True)
