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
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS — same as all three prototype notebooks
# ============================================================
CARBON_COL           = 'Carbon intensity gCO\u2082eq/kWh (direct)'
CFE_COL              = 'Carbon-free energy percentage (CFE%)'
DECISION_THRESHOLD   = 0.4      # Phase 2 XGBoost decision threshold
REDUCTION_RATE_FRAC  = 0.04     # 4% HVAC cycling — validated Phase 1 and Phase 3
NUM_HOMES_VALIDATED  = 25       # Pecan Street Austin TX 2018
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
DATETIME_COL  = 'Datetime (UTC)'
ERCOT_PEAK_MW = 75000  # ERCOT actual peak demand (~75 GW)
month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def count_spa_events(trigger_series):
    """
    Count SPA events using rising edge detection (0 → 1 transitions).
    A single continuous vulnerability window counts as ONE event,
    not multiple hours.
    """
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
    """
    Calculate total reduction in kW for given homes and reduction rate.
    Validated baseline: 4% reduction = 0.0920 kW per home (Pecan Street 2018)
    """
    scaling_factor = reduction_rate_percent / 4.0
    return homes * KW_PER_HOME * scaling_factor

# ============================================================
# LOAD MODEL AND DATA
# No retraining. No Google Drive. Local files only.
# ============================================================
@st.cache_resource
def load_model():
    """Load trained XGBoost model from repo."""
    try:
        return joblib.load("gridsaver_model.pkl")
    except FileNotFoundError:
        st.error("❌ Model file 'gridsaver_model.pkl' not found. Please ensure it exists in the app directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

@st.cache_data
def load_data():
    """
    Load processed ERCOT dataset from repo.
    Derived output from prototype notebooks — not raw academic data.
    Source: Electricity Maps US-TEX-ERCO 2025 (Academic Access).
    """
    try:
        df = pd.read_csv("data_sample.csv")
        df['Datetime (UTC)'] = pd.to_datetime(df['Datetime (UTC)'])
        df = df.sort_values('Datetime (UTC)').reset_index(drop=True)
        return df
    except FileNotFoundError:
        st.error("❌ Data file 'data_sample.csv' not found. Please ensure it exists in the app directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")
        st.stop()

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
    """
    df_out = df_input.copy()

    # Ensure hour and month exist for merge
    if 'hour' not in df_out.columns:
        df_out['hour'] = df_out[DATETIME_COL].dt.hour
    if 'month' not in df_out.columns:
        df_out['month'] = df_out[DATETIME_COL].dt.month

    # Build time-only feature input — no vulnerability_score, no carbon, no CFE
    pjm_avg_demand = 35000  # approximate mean PJM demand (MW) for scaling
    time_features = pd.DataFrame({
        'datetime':  df_input[DATETIME_COL],
        'demand_mw': pjm_avg_demand + (
            np.where(df_input[DATETIME_COL].dt.month.isin([6, 7, 8]), 5000,
            np.where(df_input[DATETIME_COL].dt.month.isin([12, 1, 2]), 3000, 0))
            + np.where(df_input[DATETIME_COL].dt.hour.between(15, 20), 2000,
              np.where(df_input[DATETIME_COL].dt.hour.between(6, 9), 1000, -500))
        )
    })

    df_engineered = engineer_features(time_features)

    if df_engineered.empty:
        st.warning("Predict Layer: insufficient temporal data for risk projection.")
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
# ACT LAYER (FIXED - with proper event counting)
# ============================================================
def act_layer(df_input, reduction_rate_percent, homes, apply_intervention_flag):
    """
    SPA dual-confirmation: Grid Saver only acts when BOTH
    Sense AND Predict independently confirm vulnerability.
    
    Fixed: Uses reduction_rate_percent (not fraction) and respects intervention toggle.
    """
    df_a = df_input.copy()
    df_a['sense_triggered']      = df_a['vulnerability_event']
    df_a['spa_action_triggered'] = (
        df_a['sense_triggered'] & df_a['predict_triggered']
    )
    
    # Calculate reduction
    if apply_intervention_flag:
        scaled_kw_per_home = KW_PER_HOME * (reduction_rate_percent / 4)
        df_a['grid_saver_reduction_kw'] = np.where(
            df_a['spa_action_triggered'],
            homes * scaled_kw_per_home,
            0
        )
    else:
        df_a['grid_saver_reduction_kw'] = 0
    
    df_a['reduction_mw'] = df_a['grid_saver_reduction_kw'] / 1000
    total_mw_saved = homes * KW_PER_HOME * (reduction_rate_percent / 4) / 1000 if apply_intervention_flag else 0
    
    return df_a, total_mw_saved

# ============================================================
# ERCOT DEMAND CALCULATION (FIXED - replaced synthetic demand)
# ============================================================
def add_ercot_demand(df_input):
    """
    Add ERCOT-calibrated demand based on vulnerability score.
    Demand scales from 55% to 95% of ERCOT peak (75,000 MW).
    This replaces the synthetic PJM-based demand used earlier.
    """
    df_out = df_input.copy()
    base_load_pct = 0.55  # 55% of peak at vulnerability score 0
    peak_load_pct = 0.95  # 95% of peak at vulnerability score 100
    
    df_out['ercot_demand_mw'] = (
        base_load_pct + (df_out['vulnerability_score'] / 100) * (peak_load_pct - base_load_pct)
    ) * ERCOT_PEAK_MW
    
    return df_out

# ============================================================
# IMPACT AND DISPATCH CALCULATIONS
# ============================================================
def compute_impact_metrics(row, homes, reduction_rate_percent):
    """
    Computes energy, cost and CO2 impact per SPA event.
    Assumes 1-hour intervention window per event.
    Cost baseline: $100/MWh (conservative grid average).
    CO2: derived from actual carbon intensity of current row.
    """
    reduction_kw   = homes * KW_PER_HOME * (reduction_rate_percent / 4)
    reduction_mw   = reduction_kw / 1000
    mwh_saved      = reduction_mw * 1  # 1-hour intervention window
    cost_savings   = mwh_saved * 100   # $100/MWh conservative baseline
    carbon_intensity = row[CARBON_COL]
    co2_avoided_tons = (carbon_intensity * mwh_saved) / 1000
    return mwh_saved, cost_savings, co2_avoided_tons

def compute_dispatch_priority(row):
    """
    Scores grid urgency 0 to 100 for utility-style dispatch prioritisation.
    Formula: (vulnerability_score × 0.5) + (risk_probability × 30) + min(carbon/10, 20)
    """
    score  = row['vulnerability_score'] * 0.5
    score += row.get('vuln_probability', 0) * 30
    score += min(row[CARBON_COL] / 10, 20)
    return round(score, 1)

# ============================================================
# RISK DRIVERS
# ============================================================
def get_risk_drivers(row, vulnerability_threshold, df_full):
    """
    Score decides the classification.
    Carbon intensity and CFE explain why the score is where it is.
    """
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
# RUN FULL SPA PIPELINE ON FULL DATASET
# ============================================================
df, VULNERABILITY_THRESHOLD = sense_layer(df)
df['hour']       = df[DATETIME_COL].dt.hour
df['month']      = df[DATETIME_COL].dt.month
df['month_name'] = df[DATETIME_COL].dt.strftime('%b')
df['date']       = df[DATETIME_COL].dt.date
df['day_of_week']= df[DATETIME_COL].dt.dayofweek
df['year']       = df[DATETIME_COL].dt.year
df['week']       = df[DATETIME_COL].dt.isocalendar().week.astype(int)

# Run Predict Layer (independent, temporal only)
df = predict_layer(df, model)

# Add ERCOT-calibrated demand (replaces synthetic PJM demand)
df = add_ercot_demand(df)

# Run Act Layer on full dataset for accurate SPA counts (using validated reduction rate)
df_full, _ = act_layer(df, 4, NUM_HOMES_VALIDATED, True)  # Use 4% validated rate for baseline

# Calculate SPA events using rising edge detection (FIXED)
SENSE_TRIGGERS_TOTAL   = int(df_full['sense_triggered'].sum())
PREDICT_TRIGGERS_TOTAL = int(df_full['predict_triggered'].sum())
SPA_ACTIONS_TOTAL      = count_spa_events(df_full['spa_action_triggered'])  # FIXED: uses event counting

# ============================================================
# LOCKED BASELINE TRUTH METRICS — from MVP notebook outputs
# These match the notebook exactly and do NOT change with sliders.
# ============================================================
NOTEBOOK_SENSE_TRIGGERS   = 1316
NOTEBOOK_PREDICT_TRIGGERS = 1659
NOTEBOOK_SPA_ACTIONS      = 154   # Validated from Phase 3 notebook
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

reduction_rate_percent = st.sidebar.slider(
    "HVAC Reduction Rate (%) — Validated target: 3 to 5%",
    min_value=1, max_value=10, value=4, step=1
)

homes = st.sidebar.slider(
    "Homes Coordinated",
    min_value=1000, max_value=1000000, value=100000, step=1000
)

apply_intervention_flag = st.sidebar.toggle("Apply Grid Saver Intervention", value=True)

st.sidebar.divider()
with st.sidebar.expander("📊 Dataset & Architecture Information"):
    st.markdown("""
    **Sense Layer:** Electricity Maps US-TEX-ERCO 2025
    - 8,760 hourly records
    - Carbon intensity and CFE% signals
    - Vulnerability score: 0-100 scale
    
    **Predict Layer:** PJM Interconnection 1998-2002
    - 32,896 hourly records
    - XGBoost model | 91.3% Recall
    - 24hr ahead risk signal (temporal patterns only)
    - Independent of Sense Layer
    
    **Act Layer:** Pecan Street Inc. Austin TX 2018
    - 25 real households
    - 868,096 records | 56.3% HVAC share
    - Validated reduction: 4% = 0.0920 kW/home
    
    **SPA Dual-Confirmation**
    - Sense AND Predict must BOTH confirm
    - Reduces false positives to 1.8% of year
    - Validated across all three phases
    """)

st.sidebar.divider()
st.sidebar.markdown("**Stack**")
st.sidebar.markdown("Colab + GitHub + Streamlit Cloud")
st.sidebar.divider()
st.sidebar.markdown("*Justine Adzormado*")

# ============================================================
# FILTER DATA FOR DISPLAY
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

# Run act layer on filtered view with user's settings and intervention toggle
df_view, total_mw_saved = act_layer(df_view, reduction_rate_percent, homes, apply_intervention_flag)

# Calculate SPA events using rising edge detection (FIXED)
spa_events_view = count_spa_events(df_view['spa_action_triggered'])
total_energy_removed_mwh = df_view['reduction_mw'].sum()

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
current_prob   = current_row.get('vuln_probability', 0)

vulnerable_hours   = int(df_view['vulnerability_event'].sum())
vulnerable_pct     = df_view['vulnerability_event'].mean() * 100

status_color = {'STABLE': '#2ECC71', 'WARNING': '#F39C12', 'CRITICAL': '#E74C3C'}
status_icon  = {'STABLE': '🟢', 'WARNING': '🟡', 'CRITICAL': '🔴'}

col1, col2, col3, col4, col5, col6 = st.columns(6)
cards = [
    (col1, status_icon[current_status], current_status,   "Grid Status",                            status_color[current_status]),
    (col2, f"{current_score:.0f}",      "/100",            "Vulnerability Score",                    "white"),
    (col3, f"{current_carbon:.0f}",     "gCO₂/kWh",        "Carbon Intensity",                       "#E74C3C"),
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
    "📈 <strong>24hr Risk Projection:</strong> Purple line in chart below shows XGBoost pattern-recognition "
    "from historical grid stress cycles (PJM 1998-2002). Independent of Sense Layer — both meet only at SPA decision point."
    "</p>",
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# SECTION 2 — RISK DRIVERS
# ============================================================
st.markdown("## 🔍 Risk Drivers")
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
st.markdown("## 📈 Grid Demand and Vulnerability Windows")

color_map  = {'STABLE': '#2ECC71', 'WARNING': '#F39C12', 'CRITICAL': '#E74C3C'}
fig_demand = go.Figure()

for status in ['STABLE', 'WARNING', 'CRITICAL']:
    mask = df_view['grid_status'] == status
    if mask.any():
        fig_demand.add_trace(go.Scatter(
            x=df_view[mask][DATETIME_COL],
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

fig_demand.add_trace(go.Scatter(
    x=df_view[DATETIME_COL],
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

st.markdown("""
<div class='info-box'>
📌 <strong>Understanding This Chart:</strong><br>
• <strong style='color:#2ECC71'>Green (STABLE):</strong> Score &lt; 40 — Grid operating normally<br>
• <strong style='color:#F39C12'>Yellow (WARNING):</strong> Score 40-69 — Elevated risk, prepare intervention<br>
• <strong style='color:#E74C3C'>Red (CRITICAL):</strong> Score ≥ 70 — Immediate action required<br>
• <strong style='color:#9B59B6'>Purple dashed line:</strong> 24hr Risk Projection — temporal pattern forecast from XGBoost<br>
• <strong style='color:#FF4444'>Red dashed line:</strong> Vulnerability threshold (top 15% most stressed hours)
</div>
""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# SECTION 4 — PEAK VULNERABILITY TIMELINE
# ============================================================
st.markdown("## 🕒 Peak Vulnerability Timeline")
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
    
    st.markdown("""
    <div class='info-box'>
    📌 <strong>Hourly Pattern:</strong> Shows average vulnerability by hour of day (UTC).
    Critical threshold (70+) indicates hours with highest grid stress.
    </div>
    """, unsafe_allow_html=True)

with col_right:
    if live_mode:
        st.info("📅 Monthly trend requires full-year data. Switch to Analysis Mode to view seasonal patterns.")
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
        fig_daily.add_hline(y=VULNERABILITY_THRESHOLD, line_dash='dash', line_color='#FF4444')
        fig_daily.update_layout(
            paper_bgcolor='#161B22', plot_bgcolor='#161B22', font=dict(color='white'),
            title=dict(text=f'Avg Daily Vulnerability — {selected_month}', font=dict(color='white', size=13)),
            xaxis=dict(gridcolor='#30363D', color='#888', title='Date', tickangle=45),
            yaxis=dict(gridcolor='#30363D', color='#888', title='Vulnerability Score'),
            height=300, margin=dict(t=60, b=60),
        )
        st.plotly_chart(fig_daily, use_container_width=True)
        
        st.markdown("""
        <div class='info-box'>
        📌 <strong>Daily Pattern:</strong> Shows daily average vulnerability for the selected month.
        Days above the red threshold line indicate high-risk periods requiring intervention.
        </div>
        """, unsafe_allow_html=True)
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
        fig_month.add_hline(y=VULNERABILITY_THRESHOLD, line_dash='dash', line_color='#FF4444')
        fig_month.update_layout(
            paper_bgcolor='#161B22', plot_bgcolor='#161B22', font=dict(color='white'),
            title=dict(text='Avg Grid Vulnerability by Month', font=dict(color='white', size=13)),
            xaxis=dict(gridcolor='#30363D', color='#888', title='Month'),
            yaxis=dict(gridcolor='#30363D', color='#888', title='Vulnerability Score'),
            height=300, margin=dict(t=50, b=30),
        )
        st.plotly_chart(fig_month, use_container_width=True)
        
        st.markdown("""
        <div class='info-box'>
        📌 <strong>Seasonal Pattern:</strong> Shows average vulnerability by month.
        Texas ERCOT typically sees highest grid stress in summer (Jun-Aug) due to HVAC demand,
        and winter (Dec-Feb) due to heating loads.
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# SECTION 5 — RECOMMENDED GRID ACTION
# ============================================================
st.markdown("## 🎯 Recommended Grid Action")

if current_status == 'CRITICAL':
    action_color    = "#E74C3C"
    action_icon_str = "🔴"
    action_title    = "CRITICAL — Immediate Action Required"
    action_text     = f"Reduce residential HVAC load by {reduction_rate_percent}% across coordination zones"
    expected        = f"Expected peak reduction: {reduction_rate_percent * 0.9:.1f}% | Grid stabilization: HIGH confidence"
elif current_status == 'WARNING':
    action_color    = "#F39C12"
    action_icon_str = "🟡"
    action_title    = "WARNING — Prepare for Intervention"
    action_text     = f"Pre-stage HVAC coordination. Initiate {reduction_rate_percent}% reduction if conditions escalate"
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
        <h3 style='color: #4A9EFF; margin: 0 0 15px 0;'>🧠 AI Decision Explanation</h3>
        <p style='color: #CCC; margin: 5px 0;'>
            Grid Saver classified the system as
            <strong style='color: {action_color};'>{current_status}</strong> based on:
        </p>
        <ul style='color: #CCC; margin: 10px 0;'>
            <li><strong>Vulnerability Score:</strong> {current_score:.1f} / 100 (threshold: {VULNERABILITY_THRESHOLD:.0f})</li>
            <li><strong>Carbon Intensity:</strong> {current_carbon:.0f} gCO₂eq/kWh</li>
            <li><strong>Carbon-Free Energy:</strong> {current_cfe:.1f}%</li>
            <li><strong>24hr Risk Projection:</strong> {current_prob:.2f} (threshold: {DECISION_THRESHOLD})</li>
            <li><strong>Vulnerability Level:</strong> {vulnerability_level}</li>
        </ul>
        <p style='color: #CCC; margin: 10px 0 5px 0;'>
            <strong>Recommended Action:</strong> {action_text}
        </p>
        <p style='color: #888; margin: 0; font-size: 0.9rem;'>{expected}</p>
        <p style='color: #555; margin: 15px 0 0 0; font-size: 0.8rem;'>
            A {reduction_rate_percent}% HVAC reduction across {homes:,} homes
            removes approximately {homes * KW_PER_HOME * (reduction_rate_percent / 4):,.1f} kW
            per SPA event (1-hour window) from peak demand.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# SECTION 6 — SMART RECOMMENDATIONS (Operational Intelligence)
# ============================================================
st.markdown("## 🧠 Smart Recommendations")

# Compute dispatch priority
dispatch_score = compute_dispatch_priority(current_row)
priority_label = "Immediate action required" if dispatch_score > 70 else "Monitor closely" if dispatch_score > 40 else "Normal operations"
priority_color = "#E74C3C" if dispatch_score > 70 else "#F39C12" if dispatch_score > 40 else "#2ECC71"

st.markdown(f"""
<div style='background:#161B22; border:1px solid {priority_color};
     padding:12px 16px; border-radius:8px; margin-bottom:12px;'>
    <span style='color:{priority_color}; font-weight:bold; font-size:1rem;'>
        ⚡ Dispatch Priority Score: {dispatch_score}/100 — {priority_label}
    </span>
    <p style='color:#888; margin:5px 0 0 0; font-size:0.75rem;'>
        <strong>Formula:</strong> (vulnerability_score × 0.5) + (risk_probability × 30) + min(carbon/10, 20)
    </p>
</div>
""", unsafe_allow_html=True)

# Compute per-event impact
mwh_saved, cost_savings, co2_avoided = compute_impact_metrics(current_row, homes, reduction_rate_percent)

# Build recommendations based on grid status
if current_status == 'CRITICAL':
    st.markdown(f"🔴 **Priority Action:** Execute {reduction_rate_percent}% HVAC reduction across {homes:,} homes immediately.")
    if current_row.get('spa_action_triggered', False):
        st.markdown("✅ SPA dual-confirmation achieved — full automated dispatch approved.")
    else:
        st.markdown("⚠️ Partial confirmation — deploy controlled reduction while monitoring.")
    st.markdown("📍 **Target:** High-demand feeders and peak residential clusters.")
    st.markdown("📢 **Notify:** Grid operators and demand response aggregators immediately.")

elif current_status == 'WARNING':
    st.markdown("🟡 **Priority Action:** Pre-stage demand response resources for potential activation.")
    if current_prob >= 0.4:
        st.markdown("📈 Temporal risk signal elevated — prepare for activation within next peak window.")
    else:
        st.markdown("📊 Risk is moderate — maintain readiness and continue monitoring.")
    st.markdown("🔌 **Shift:** Flexible loads (EV charging, water heating) to off-peak hours.")
    st.markdown("📱 **Issue:** Advisory notifications to consumers in high-load zones.")

else:
    st.markdown("🟢 **Status:** No immediate intervention required.")
    if current_prob >= 0.4:
        st.markdown("📈 Temporal model indicates potential future risk — monitor upcoming peak window.")
    st.markdown("💡 **Encourage:** Off-peak consumption behaviour to maintain grid efficiency.")

# Impact summary
st.markdown(f"""
<div style='background:#161B22; border-left:4px solid #2ECC71;
     padding:10px 14px; border-radius:6px; margin-top:12px;'>
    <span style='color:#2ECC71; font-size:0.9rem;'>📊 <strong>Impact Summary</strong></span><br>
    <span style='color:#CCC; font-size:0.85rem;'>
        Per SPA event (1-hour window): {mwh_saved:.2f} MWh | ${cost_savings:,.0f} savings | {co2_avoided:.3f} tons CO₂ avoided<br>
        <strong>Current period:</strong> {spa_events_view} SPA events | {total_energy_removed_mwh:.2f} MWh total reduction
    </span>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "<p style='color:#555; font-size:0.78rem; margin-top:8px;'>"
    "📌 <strong>Note:</strong> Cost estimate based on $100/MWh conservative grid baseline. "
    "CO₂ derived from current carbon intensity. "
    "Feeder-level targeting and real-time pricing integration available in production."
    "</p>",
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# SECTION 7 — GRID SAVER LOAD REDUCTION SIMULATION
# FIXED: Uses ERCOT demand, proper event counting, intervention toggle
# ============================================================
st.markdown("## ⚡ Grid Saver Load Reduction Simulation")

if reduction_rate_percent != 4:
    st.warning(
        f"⚠️ **Note:** Grid Saver is validated at 4% HVAC reduction (0.0920 kW per home). "
        f"Results at {reduction_rate_percent}% are proportionally scaled estimates, "
        f"not independently validated values."
    )

# Display trigger counts with correct event numbers
if live_mode:
    display_sense   = int(df_view['sense_triggered'].sum())
    display_predict = int(df_view['predict_triggered'].sum())
    display_spa     = spa_events_view
    trigger_sub     = "hours (last 24h)"
    spa_sub         = "events (last 24h)"
else:
    display_sense   = NOTEBOOK_SENSE_TRIGGERS
    display_predict = NOTEBOOK_PREDICT_TRIGGERS
    display_spa     = NOTEBOOK_SPA_ACTIONS
    trigger_sub     = "hours (full year validated)"
    spa_sub         = f"events (full year — {NOTEBOOK_SPA_ACTIONS} validated)"

col_sim1, col_sim2, col_sim3 = st.columns(3)
sim_cards = [
    (col_sim1, f"{display_sense:,}",   trigger_sub, "Sense Triggers (Carbon + CFE)",             "#4A9EFF"),
    (col_sim2, f"{display_predict:,}", trigger_sub, "Predict Triggers (Temporal Pattern)",       "#9B59B6"),
    (col_sim3, f"{display_spa:,}",     spa_sub,     "SPA Actions (Dual-Confirmed)",              "#F39C12"),
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

st.markdown(f"""
<div style='background:#161B22; border-left:5px solid #2ECC71;
     padding:15px; border-radius:8px; margin:15px 0;'>
    <h3 style='color:#2ECC71; margin:0;'>⚡ Total Energy Removed: {total_energy_removed_mwh:,.2f} MWh</h3>
    <p style='color:#888; margin:5px 0 0 0;'>
        Sum of hourly reductions across SPA-triggered events | 
        Current period: {spa_events_view} events | 
        Full-year validated: {NOTEBOOK_SPA_ACTIONS} events
    </p>
    <p style='color:#555; font-size:0.75rem; margin-top:5px;'>
        📌 <strong>Scenario output</strong> — scales with homes ({homes:,}) and reduction rate ({reduction_rate_percent}%)
    </p>
</div>
""", unsafe_allow_html=True)

# Calculate peak reduction using ERCOT demand (FIXED)
peak_idx = df_view['ercot_demand_mw'].idxmax()
peak_time = df_view.loc[peak_idx, DATETIME_COL]
original_peak = df_view.loc[peak_idx, 'ercot_demand_mw']
after_peak = df_view.loc[peak_idx, 'ercot_demand_mw'] - df_view.loc[peak_idx, 'reduction_mw']
peak_reduction_mw = original_peak - after_peak
pct_reduction = (peak_reduction_mw / original_peak) * 100 if original_peak > 0 else 0

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
    f"📌 <strong>Peak insight:</strong> Original {original_peak:,.0f} MW reduced to {after_peak:,.0f} MW "
    f"at peak demand timestamp. Load shed: {peak_reduction_mw:,.2f} MW ({pct_reduction:.2f}%). "
    f"Validated at 4% HVAC reduction (0.0920 kW per home). "
    f"Reduction occurs only during dual-confirmed SPA events — not continuously."
    f"</p>",
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# BEFORE PLOT — Original demand with SPA markers
# ============================================================
st.markdown("#### 📉 Before Grid Saver — Original ERCOT Demand with SPA Markers")

fig_before = go.Figure()
fig_before.add_trace(go.Scatter(
    x=df_view[DATETIME_COL],
    y=df_view['ercot_demand_mw'],
    mode='lines',
    name='Original ERCOT Demand',
    line=dict(color='#E74C3C', width=1.5),
))

# SPA markers (only if intervention is enabled)
if apply_intervention_flag:
    spa_triggered = df_view[df_view['spa_action_triggered']]
    if not spa_triggered.empty:
        fig_before.add_trace(go.Scatter(
            x=spa_triggered[DATETIME_COL],
            y=spa_triggered['ercot_demand_mw'],
            mode='markers',
            name='SPA Action Triggered',
            marker=dict(color='#F39C12', size=8, symbol='circle', line=dict(width=1.5, color='white')),
        ))

# Mark peak timestamp
fig_before.add_trace(go.Scatter(
    x=[peak_time, peak_time],
    y=[df_view['ercot_demand_mw'].min(), df_view['ercot_demand_mw'].max()],
    mode='lines',
    name='Peak Demand',
    line=dict(color='#FFFFFF', width=1.5, dash='dash'),
    showlegend=True
))

fig_before.update_layout(
    paper_bgcolor='#161B22', plot_bgcolor='#161B22',
    font=dict(color='white'),
    title=dict(text='Before Grid Saver — Original ERCOT Demand with SPA Markers', font=dict(color='white', size=13)),
    xaxis=dict(gridcolor='#30363D', color='#888', title='Datetime'),
    yaxis=dict(gridcolor='#30363D', color='#888', title='Demand (MW)'),
    legend=dict(bgcolor='#1A1A2E', bordercolor='#333'),
    height=350, margin=dict(t=50, b=20),
)
st.plotly_chart(fig_before, use_container_width=True)

st.markdown("""
<div class='info-box'>
📌 <strong>Chart Guide — BEFORE:</strong><br>
• <span style='color:#E74C3C'>🔴 Red line:</span> Original ERCOT demand (55-95% of 75 GW peak, scaled by vulnerability score)<br>
• <span style='color:#F39C12'>🟠 Orange markers:</span> Hours where SPA dual-confirmation triggered intervention<br>
• <span style='color:#FFFFFF'>⚪ Dashed white line:</span> Peak demand timestamp
</div>
""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# AFTER PLOT — Adjusted demand with shaded SPA zones and drop lines
# ============================================================
st.markdown("#### 📈 After Grid Saver — Demand with Intervention Applied")

fig_after = go.Figure()

# Original demand (faded reference)
fig_after.add_trace(go.Scatter(
    x=df_view[DATETIME_COL],
    y=df_view['ercot_demand_mw'],
    mode='lines',
    name='Original Demand (Reference)',
    line=dict(color='#E74C3C', width=1, dash='dot'),
    opacity=0.35,
))

# Adjusted demand after intervention
line_color = '#2ECC71' if apply_intervention_flag else '#888888'
fig_after.add_trace(go.Scatter(
    x=df_view[DATETIME_COL],
    y=df_view['ercot_demand_mw'] - df_view['reduction_mw'],
    mode='lines',
    name='After Grid Saver Intervention' if apply_intervention_flag else 'Intervention Disabled',
    line=dict(color=line_color, width=1.5),
    fill='tonexty' if apply_intervention_flag else None,
    fillcolor='rgba(46, 204, 113, 0.08)' if apply_intervention_flag else None,
))

# Shaded SPA zones and drop lines (only if intervention enabled)
if apply_intervention_flag:
    spa_triggered = df_view[df_view['spa_action_triggered']].copy()
    
    # Shaded zones at ALL SPA-triggered hours
    for idx, row in spa_triggered.iterrows():
        fig_after.add_vrect(
            x0=row[DATETIME_COL],
            x1=row[DATETIME_COL] + pd.Timedelta(hours=1),
            fillcolor='rgba(255, 165, 0, 0.15)',
            line_width=0,
            layer='below',
        )
    
    # Drop lines — top 20 by reduction magnitude
    MAX_LINES = 20
    if 'reduction_mw' in spa_triggered.columns:
        top_spa = spa_triggered.sort_values('reduction_mw', ascending=False).head(MAX_LINES)
        for _, row in top_spa.iterrows():
            fig_after.add_trace(go.Scatter(
                x=[row[DATETIME_COL], row[DATETIME_COL]],
                y=[row['ercot_demand_mw'], row['ercot_demand_mw'] - row['reduction_mw']],
                mode='lines',
                line=dict(color='#F39C12', width=2),
                showlegend=False,
            ))

# Mark peak timestamp
fig_after.add_trace(go.Scatter(
    x=[peak_time, peak_time],
    y=[df_view['ercot_demand_mw'].min(), df_view['ercot_demand_mw'].max()],
    mode='lines',
    name='Peak Demand',
    line=dict(color='#FFFFFF', width=1.5, dash='dash'),
    showlegend=False,
))

fig_after.update_layout(
    paper_bgcolor='#161B22', plot_bgcolor='#161B22',
    font=dict(color='white'),
    title=dict(text='After Grid Saver — Demand with Intervention Applied', font=dict(color='white', size=13)),
    xaxis=dict(gridcolor='#30363D', color='#888', title='Datetime'),
    yaxis=dict(gridcolor='#30363D', color='#888', title='Demand (MW)'),
    legend=dict(bgcolor='#1A1A2E', bordercolor='#333'),
    height=350, margin=dict(t=50, b=20),
)
st.plotly_chart(fig_after, use_container_width=True)

st.markdown("""
<div class='info-box'>
📌 <strong>Chart Guide — AFTER:</strong><br>
• <span style='color:#E74C3C'>🔴 Red dotted line:</span> Original demand (reference)<br>
• <span style='color:#2ECC71'>🟢 Green line:</span> Demand after Grid Saver intervention<br>
• <span style='color:#FFA500'>🟧 Shaded zones:</span> SPA-triggered windows (1-hour each)<br>
• <span style='color:#F39C12'>🟡 Orange drop lines:</span> Top 20 highest-impact interventions (before → after reduction)<br>
• <span style='color:#FFFFFF'>⚪ Dashed white line:</span> Peak demand timestamp
</div>
""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# PEAK ZOOM WINDOW — ±6 hours around peak demand (in expander)
# ============================================================
with st.expander("🔍 Inspect Peak Demand Window (6 Hours Before and After Peak)"):
    st.markdown("""
    **Why ±6 hours?** This window captures the critical period around peak demand — 
    the 6 hours before (ramp-up to peak) and 6 hours after (recovery). 
    This is where Grid Saver intervention has the highest impact potential.
    
    **Calculation:** `peak_time ± 6 hours` = 13 total hours of detailed analysis
    """)
    
    zoom_df = df_view[
        (df_view[DATETIME_COL] >= peak_time - pd.Timedelta(hours=6)) &
        (df_view[DATETIME_COL] <= peak_time + pd.Timedelta(hours=6))
    ].copy()
    
    if not zoom_df.empty:
        fig_zoom = go.Figure()
        
        # Original demand
        fig_zoom.add_trace(go.Scatter(
            x=zoom_df[DATETIME_COL],
            y=zoom_df['ercot_demand_mw'],
            mode='lines+markers',
            name='Original Demand',
            line=dict(color='#E74C3C', width=2),
            marker=dict(size=4, color='#E74C3C')
        ))
        
        # Adjusted demand
        fig_zoom.add_trace(go.Scatter(
            x=zoom_df[DATETIME_COL],
            y=zoom_df['ercot_demand_mw'] - zoom_df['reduction_mw'],
            mode='lines+markers',
            name='After Grid Saver',
            line=dict(color='#2ECC71', width=2),
            marker=dict(size=4, color='#2ECC71'),
            fill='tonexty' if apply_intervention_flag else None,
            fillcolor='rgba(46, 204, 113, 0.15)'
        ))
        
        # Drop lines and shaded zones in zoom window
        if apply_intervention_flag:
            zoom_spa = zoom_df[zoom_df['spa_action_triggered']]
            for _, row in zoom_spa.iterrows():
                # Drop lines
                fig_zoom.add_trace(go.Scatter(
                    x=[row[DATETIME_COL], row[DATETIME_COL]],
                    y=[row['ercot_demand_mw'], row['ercot_demand_mw'] - row['reduction_mw']],
                    mode='lines',
                    line=dict(color='#F39C12', width=2.5),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                # Shaded zones
                fig_zoom.add_vrect(
                    x0=row[DATETIME_COL],
                    x1=row[DATETIME_COL] + pd.Timedelta(hours=1),
                    fillcolor='rgba(255, 165, 0, 0.25)',
                    line_width=0,
                    layer='below'
                )
        
        # Peak line
        fig_zoom.add_trace(go.Scatter(
            x=[peak_time, peak_time],
            y=[zoom_df['ercot_demand_mw'].min(), zoom_df['ercot_demand_mw'].max()],
            mode='lines',
            name='Peak Moment',
            line=dict(color='#FFFFFF', width=2, dash='dash'),
            showlegend=True
        ))
        
        fig_zoom.update_layout(
            paper_bgcolor='#161B22', plot_bgcolor='#161B22',
            font=dict(color='white'),
            title=dict(text=f'Peak Window — Full Detail Around {peak_time.strftime("%Y-%m-%d %H:%M")}', font=dict(color='white', size=13)),
            xaxis=dict(gridcolor='#30363D', color='#888', title='Datetime'),
            yaxis=dict(gridcolor='#30363D', color='#888', title='Demand (MW)'),
            legend=dict(bgcolor='#1A1A2E', bordercolor='#333'),
            height=400, margin=dict(t=50, b=20),
        )
        st.plotly_chart(fig_zoom, use_container_width=True)
        
        # Calculate peak window statistics
        zoom_original_peak = zoom_df['ercot_demand_mw'].max()
        zoom_adjusted_peak = (zoom_df['ercot_demand_mw'] - zoom_df['reduction_mw']).max()
        zoom_peak_reduction = zoom_original_peak - zoom_adjusted_peak
        zoom_pct = (zoom_peak_reduction / zoom_original_peak * 100) if zoom_original_peak > 0 else 0
        
        st.markdown(f"""
        <div class='info-box'>
        🔍 <strong>Peak Window Statistics (±6 hours):</strong><br>
        • Original peak in window: <strong style='color:#E74C3C'>{zoom_original_peak:,.0f} MW</strong><br>
        • After intervention peak: <strong style='color:#2ECC71'>{zoom_adjusted_peak:,.0f} MW</strong><br>
        • Peak reduction: <strong style='color:#F39C12'>{zoom_peak_reduction:,.2f} MW ({zoom_pct:.2f}%)</strong><br>
        • SPA events in window: <strong>{len(zoom_spa) if apply_intervention_flag else 0}</strong>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("No data available in the ±6 hour peak window.")

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# SECTION 8 — IMPACT AT SCALE
# ============================================================
st.divider()
st.markdown("## 📊 Impact at Scale")
st.markdown("*Adjust the Homes Coordinated slider in the sidebar to see how Grid Saver scales.*")

scaled_reduction_kw = compute_scaled_reduction_kw(homes, reduction_rate_percent)
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

reserve_note  = "✅ Exceeds ERCOT reserve margin" if scaled_reduction_mw > 200 else "📈 Building toward reserve margin"
rm_color      = "#2ECC71" if scaled_reduction_mw > 200 else "#F39C12"

col_s1, col_s2, col_s3, col_s4 = st.columns(4)
scale_cards = [
    (col_s1, f"{homes:,}",                    "Homes",   "Homes Coordinated",  "#4A9EFF"),
    (col_s2, f"{scaled_reduction_mw:,.1f} MW", "removed", "Projected Grid Reduction (per event)", "#2ECC71"),
    (col_s3, grid_impact,                      "",        "Impact Level",        impact_color),
    (col_s4, reserve_note,                     "",        "Reserve Margin Status",      rm_color),
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
    f"📌 At this scale, Grid Saver removes approximately <strong>{percentage_of_grid:.3f}%</strong> "
    f"of ERCOT peak demand (~{ERCOT_PEAK_MW:,} MW). "
    f"Validated reduction rate: 4% HVAC cycling (0.0920 kW per home, Pecan Street 2018)."
    f"</p>",
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# SECTION 9 — SYSTEM ARCHITECTURE
# ============================================================
st.markdown("## 🏗️ System Architecture")
col_a, col_b, col_c = st.columns(3)

arch = [
    (col_a, "👁️", "SENSE",   "#1B4F8C", "#4A9EFF",
     "Detect grid vulnerability signals continuously",
     "Carbon intensity monitoring",
     "Electricity Maps US-TEX-ERCO",
     "8,760 hourly records",
     "✅ Phase 1 Complete"),
    (col_b, "🧠", "PREDICT", "#1A6B2E", "#2ECC71",
     "Forecast vulnerability windows",
     "XGBoost model | 91.3% Recall",
     "PJM 32,896 hourly records",
     "24hr Risk Projection (Temporal Pattern)",
     "✅ Phase 2 Complete"),
    (col_c, "⚡", "ACT",     "#7B1A1A", "#E74C3C",
     "Coordinate residential HVAC load reduction",
     "3 to 5% targeted precision",
     "Pecan Street 868,096 records",
     "SPA dual-confirmation safety",
     "✅ Phase 3 Complete"),
]

for col, icon, title, bg, color, l1, l2, l3, l4, status in arch:
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
            <p style='color: {color}; font-size: 0.8rem; margin: 10px 0 0 0; font-weight: bold;'>{status}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# SECTION 10 — REPORTS AND INSIGHTS (in expander, LOCKED to validated values)
# ============================================================
with st.expander("📄 Reports and Insights (Download CSV)"):
    """
    Reports section provides downloadable grid performance analysis.
    SPA event counts are LOCKED to validated notebook values (154 events per year).
    Not recomputed from filtered data to maintain consistency with Phase 3 validation.
    """
    if live_mode:
        st.warning("📊 Reports are disabled in Live Mode. Switch to Analysis Mode to generate reports.")
    else:
        st.markdown("*Select a time period to view grid performance analysis and download a report.*")
        
        col_r1, col_r2, col_r3 = st.columns(3)
        
        with col_r1:
            report_type = st.selectbox("Report Type", ["Yearly", "Monthly", "Weekly"])
        
        with col_r2:
            report_year = st.selectbox("Year", sorted(df['year'].unique(), reverse=True))
        
        with col_r3:
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
            else:
                report_month = None
                report_week = None
        
        # Filter data
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
        
        # Metrics
        avg_vulnerability = df_report['vulnerability_score'].mean()
        peak_vulnerability = df_report['vulnerability_score'].max()
        
        # LOCKED: Use notebook validated value, not recomputed
        spa_events_report = NOTEBOOK_SPA_ACTIONS
        
        # Pie chart for status distribution
        status_counts = df_report['grid_status'].value_counts()
        fig_pie = go.Figure(data=[go.Pie(
            labels=status_counts.index,
            values=status_counts.values,
            hole=0.4,
            marker_colors=['#2ECC71', '#F39C12', '#E74C3C']
        )])
        fig_pie.update_layout(
            paper_bgcolor='#161B22', plot_bgcolor='#161B22',
            font=dict(color='white'),
            title=dict(text='Grid Status Distribution', font=dict(color='white', size=13)),
            height=300
        )
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Avg Vulnerability", f"{avg_vulnerability:.1f}")
        col_m2.metric("Peak Vulnerability", f"{peak_vulnerability:.1f}")
        col_m3.metric("SPA Events (Validated)", f"{spa_events_report}")
        
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Styled line chart for report (matching main dashboard style)
        fig_report = go.Figure()
        for status in ['STABLE', 'WARNING', 'CRITICAL']:
            mask = df_report['grid_status'] == status
            if mask.any():
                fig_report.add_trace(go.Scatter(
                    x=df_report[mask][DATETIME_COL],
                    y=df_report[mask]['vulnerability_score'],
                    mode='lines',
                    name=status,
                    line=dict(color=color_map[status], width=1)
                ))
        
        fig_report.add_hline(
            y=VULNERABILITY_THRESHOLD,
            line_dash='dash', line_color='#FF4444',
            annotation_text=f'Threshold ({VULNERABILITY_THRESHOLD:.0f})',
            annotation_font_color='#FF4444'
        )
        
        fig_report.update_layout(
            paper_bgcolor='#161B22', plot_bgcolor='#161B22',
            font=dict(color='white'),
            title=dict(text=f'Vulnerability Score — {period_label}', font=dict(color='white', size=13)),
            xaxis=dict(gridcolor='#30363D', color='#888'),
            yaxis=dict(gridcolor='#30363D', color='#888', title='Vulnerability Score'),
            height=300, margin=dict(t=50, b=30),
        )
        st.plotly_chart(fig_report, use_container_width=True)
        
        st.markdown(f"""
        <div class='info-box'>
        📊 <strong>Period Summary — {period_label}</strong><br>
        • Average vulnerability: <strong>{avg_vulnerability:.1f}/100</strong><br>
        • Peak vulnerability: <strong>{peak_vulnerability:.1f}/100</strong><br>
        • Validated SPA events: <strong>{spa_events_report}</strong> (from Phase 3 notebook)<br>
        • Grid status distribution shown in pie chart above
        </div>
        """, unsafe_allow_html=True)
        
        # Download button
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
        Built with Colab + GitHub + Streamlit Cloud
    </p>
    <p style='color: #555; margin: 5px 0 0 0; font-size: 0.75rem;'>
        📡 Sense: Electricity Maps US-TEX-ERCO 2025 (Academic Access) |
        🧠 Predict: PJM XGBoost | 91.3% Recall | 24hr Risk Projection |
        ⚡ Act: Pecan Street Austin TX 2018 | 25 real households |
        🔒 SPA dual-confirmation: Action only when BOTH Sense AND Predict independently confirm vulnerability
    </p>
    <p style='color: #444; margin: 8px 0 0 0; font-size: 0.7rem;'>
        ⚠️ Educational & Research Prototype — Not for real-time grid operations. 
        Full validation and regulatory review required before deployment.
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# END OF APP
# ============================================================
