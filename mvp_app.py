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
# CONSTANTS
# ============================================================
CARBON_COL           = 'Carbon intensity gCO₂eq/kWh (direct)'
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
ERCOT_PEAK_MW = 70000
month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def count_spa_events(trigger_series):
    """
    Count SPA events using rising edge detection (0 → 1 transitions).
    A single event may span multiple consecutive hours.
    """
    if len(trigger_series) == 0:
        return 0
    trigger_array = trigger_series.fillna(False).values
    # Count transitions from False to True
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
    reduction_rate_percent: 1-10 (percent)
    Validated baseline: 4% = 0.0920 kW per home
    """
    scaling_factor = reduction_rate_percent / 4.0
    return homes * KW_PER_HOME * scaling_factor


def compute_impact_metrics(row, homes, reduction_rate_percent, intervention_hours=1):
    """
    Compute energy (MWh), cost savings ($), and CO2 avoided (tons) for a single SPA event.
    Assumes 1-hour intervention window by default.
    """
    reduction_kw = compute_scaled_reduction_kw(homes, reduction_rate_percent)
    reduction_mw = reduction_kw / 1000
    mwh_saved = reduction_mw * intervention_hours
    cost_savings = mwh_saved * 100  # $100/MWh baseline
    carbon_intensity = row[CARBON_COL]
    co2_avoided_tons = (carbon_intensity * mwh_saved) / 1000
    return mwh_saved, cost_savings, co2_avoided_tons


def compute_dispatch_priority(row, carbon_col=CARBON_COL):
    """
    Scores grid urgency 0-100 for utility-style dispatch prioritisation.
    Formula: (vulnerability_score × 0.5) + (risk_probability × 30) + min(carbon/10, 20)
    """
    score = row['vulnerability_score'] * 0.5
    score += row.get('vuln_probability', 0) * 30
    score += min(row[carbon_col] / 10, 20)
    return round(score, 1)


def apply_intervention(df, enabled, homes, reduction_rate_percent):
    """
    Apply Grid Saver reduction only if enabled.
    Returns df with reduction_mw and adjusted_demand_mw columns.
    """
    df = df.copy()
    if not enabled:
        df['reduction_mw'] = 0
        if 'synthetic_demand_mw' in df.columns:
            df['adjusted_demand_mw'] = df['synthetic_demand_mw']
        return df
    
    # Intervention enabled — calculate reduction
    reduction_kw = compute_scaled_reduction_kw(homes, reduction_rate_percent)
    reduction_mw_per_home = reduction_kw / 1000
    
    # Only apply during SPA-triggered hours
    df['reduction_mw'] = np.where(
        df.get('spa_action_triggered', False),
        reduction_mw_per_home,
        0
    )
    
    if 'synthetic_demand_mw' in df.columns:
        df['adjusted_demand_mw'] = df['synthetic_demand_mw'] - df['reduction_mw']
    else:
        df['adjusted_demand_mw'] = df.get('simulated_demand_mw', 0) - df['reduction_mw']
    
    return df


# ============================================================
# LOAD MODEL AND DATA
# ============================================================
@st.cache_resource
def load_model():
    """Load trained XGBoost model from repo."""
    try:
        model = joblib.load("gridsaver_model.pkl")
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'gridsaver_model.pkl' not found. Please ensure it exists in the app directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

@st.cache_data
def load_data():
    """Load processed ERCOT dataset from repo."""
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
    df_raw = load_data()


# ============================================================
# FEATURE ENGINEERING (must match training)
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
# PREDICT LAYER (Independent — PJM temporal patterns only)
# ============================================================
def predict_layer(df_input, model):
    """
    Independent Predict Layer using ONLY temporal features.
    No vulnerability_score, no carbon, no CFE.
    Trained on PJM demand data (1998-2002).
    """
    df_out = df_input.copy()
    
    # Ensure hour and month exist
    if 'hour' not in df_out.columns:
        df_out['hour'] = df_out[DATETIME_COL].dt.hour
    if 'month' not in df_out.columns:
        df_out['month'] = df_out[DATETIME_COL].dt.month
    
    # Build time-only feature input
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
    
    df_engineered = engineer_features(time_features)
    
    if df_engineered.empty:
        st.warning("Predict Layer: insufficient temporal data for risk projection.")
        df_out['vuln_probability'] = 0.0
        df_out['predict_triggered'] = False
        return df_out
    
    vuln_proba = model.predict_proba(df_engineered[FEATURE_COLS])[:, 1]
    df_engineered['vuln_proba'] = vuln_proba
    df_engineered['hour'] = df_engineered['datetime'].dt.hour
    df_engineered['month'] = df_engineered['datetime'].dt.month
    
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
    df_out['vuln_probability'] = df_out['vuln_probability'].ffill().fillna(0)
    df_out['predict_triggered'] = df_out['predict_triggered'].ffill().fillna(False)
    return df_out


# ============================================================
# ACT LAYER (SPA Dual-Confirmation)
# ============================================================
def act_layer(df_input):
    """Apply SPA dual-confirmation logic."""
    df_a = df_input.copy()
    df_a['sense_triggered'] = df_a['vulnerability_event']
    df_a['spa_action_triggered'] = (
        df_a['sense_triggered'] & df_a['predict_triggered']
    )
    return df_a


def get_risk_drivers(row, vulnerability_threshold, df_full):
    """Explain why grid is in current state."""
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
# RUN FULL PIPELINE ON FULL DATASET
# ============================================================
df, VULNERABILITY_THRESHOLD = sense_layer(df_raw)
df['hour'] = df[DATETIME_COL].dt.hour
df['month'] = df[DATETIME_COL].dt.month
df['month_name'] = df[DATETIME_COL].dt.strftime('%b')
df['date'] = df[DATETIME_COL].dt.date
df['day_of_week'] = df[DATETIME_COL].dt.dayofweek
df['year'] = df[DATETIME_COL].dt.year
df['week'] = df[DATETIME_COL].dt.isocalendar().week.astype(int)

df = predict_layer(df, model)
df = act_layer(df)

# Synthetic demand baseline (Predict Layer only — no Sense contamination)
df['synthetic_demand_mw'] = 35000 + (
    np.where(df['month'].isin([6, 7, 8]), 5000,
    np.where(df['month'].isin([12, 1, 2]), 3000, 0))
    +
    np.where(df['hour'].between(15, 20), 2000,
    np.where(df['hour'].between(6, 9), 1000, -500))
)

# Full-year baseline metrics (for locked reference)
NOTEBOOK_SENSE_TRIGGERS = 1316
NOTEBOOK_PREDICT_TRIGGERS = 1659
NOTEBOOK_SPA_ACTIONS = 154
NOTEBOOK_VULN_THRESHOLD = 74.6

# Compute true SPA event count on full year (rising edge detection)
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
    st.sidebar.markdown(
        "<p style='color:#2ECC71; font-size:0.8rem;'>Showing the most recent 24 hours of grid data</p>",
        unsafe_allow_html=True
    )

st.sidebar.divider()

months_present = [m for m in month_order if m in df['month_name'].unique()]
month_options = ['All Year'] + months_present
selected_month = st.sidebar.selectbox("Select Month", month_options)

reduction_rate_percent = st.sidebar.slider(
    "HVAC Reduction Rate (%) — Validated target: 3 to 5%",
    min_value=1, max_value=10, value=4, step=1
)

homes = st.sidebar.slider(
    "Homes Coordinated",
    min_value=1000, max_value=1000000, value=100000, step=1000
)

apply_intervention_enabled = st.sidebar.toggle("Apply Grid Saver Intervention", value=True)

st.sidebar.divider()
with st.sidebar.expander("ℹ️ How Grid Saver Works"):
    st.markdown("""
    **Vulnerability Score (0-100)**
    - **0-39: STABLE** — Grid operating normally
    - **40-69: WARNING** — Elevated risk, prepare intervention
    - **70-100: CRITICAL** — Immediate action required
    
    **SPA Dual-Confirmation**
    - **Sense Layer** → ERCOT carbon + CFE signals
    - **Predict Layer** → PJM temporal pattern (24hr ahead)
    - **Action** → Only when BOTH confirm independently
    
    **Load Reduction**
    - Validated: 4% HVAC cycling = 0.0920 kW per home
    - Pecan Street Austin TX (25 households, 2018)
    - Scales to 92 MW at 1M homes
    """)

st.sidebar.divider()
st.sidebar.markdown("**Stack:** Colab + GitHub + Streamlit")
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

# Apply intervention toggle
df_view = apply_intervention(df_view, apply_intervention_enabled, homes, reduction_rate_percent)

# Compute SPA event count on filtered view (rising edge detection)
SPA_EVENTS_VIEW = count_spa_events(df_view['spa_action_triggered'])

# Compute total energy removed (MWh) — sum of hourly reduction_mw (which is MW per hour = MWh)
TOTAL_MWH_REMOVED = df_view['reduction_mw'].sum()

# Compute per-event metrics for current state (last row)
current_row = df_view.iloc[-1]
per_event_mwh, per_event_cost, per_event_co2 = compute_impact_metrics(
    current_row, homes, reduction_rate_percent, intervention_hours=1
)


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

current_score = current_row['vulnerability_score']
current_status = current_row['grid_status']
current_carbon = current_row[CARBON_COL]
current_cfe = current_row[CFE_COL]
current_prob = current_row.get('vuln_probability', 0)
vulnerable_pct = df_view['vulnerability_event'].mean() * 100

status_color = {'STABLE': '#2ECC71', 'WARNING': '#F39C12', 'CRITICAL': '#E74C3C'}
status_icon = {'STABLE': '🟢', 'WARNING': '🟡', 'CRITICAL': '🔴'}

col1, col2, col3, col4, col5, col6 = st.columns(6)
cards = [
    (col1, status_icon[current_status], current_status, "Grid Status", status_color[current_status]),
    (col2, f"{current_score:.0f}", "/100", "Vulnerability Score", "white"),
    (col3, f"{current_carbon:.0f}", "gCO₂/kWh", "Carbon Intensity", "#E74C3C"),
    (col4, f"{current_cfe:.1f}%", "clean energy", "Carbon-Free Energy", "#2ECC71"),
    (col5, f"{vulnerable_pct:.1f}%", "of period", "Vulnerability Rate", "#4A9EFF"),
    (col6, f"{current_prob:.2f}", "probability", "24hr Risk Projection", "#9B59B6"),
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
    "📈 <strong>24hr Risk Projection:</strong> Purple line shows XGBoost pattern-recognition from historical grid stress cycles (PJM 1998-2002). "
    "Independent of Sense Layer — both meet only at SPA decision point."
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

color_map = {'STABLE': '#2ECC71', 'WARNING': '#F39C12', 'CRITICAL': '#E74C3C'}
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
• <strong style='color:#9B59B6'>Purple dashed line:</strong> 24hr Risk Projection — temporal pattern forecast from XGBoost (independent of Sense Layer)<br>
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
    hourly_vuln = df_view.groupby('hour')['vulnerability_score'].mean().round(1)
    bar_colors_h = [
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
        daily_vuln = df_view.groupby('date')['vulnerability_score'].mean().round(1).sort_index()
        daily_colors = [
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
        fig_month.add_hline(
            y=VULNERABILITY_THRESHOLD, line_dash='dash', line_color='#FF4444',
            annotation_text=f'Threshold ({VULNERABILITY_THRESHOLD:.0f})',
            annotation_font_color='#FF4444'
        )
        fig_month.update_layout(
            paper_bgcolor='#161B22', plot_bgcolor='#161B22', font=dict(color='white'),
            title=dict(text='Avg Grid Vulnerability by Month', font=dict(color='white', size=13)),
            xaxis=dict(gridcolor='#30363D', color='#888', title='Month'),
            yaxis=dict(gridcolor='#30363D', color='#888', title='Vulnerability Score'),
            height=300, margin=dict(t=50, b=30),
        )
        st.plotly_chart(fig_month, use_container_width=True)


# ============================================================
# SECTION 5 — RECOMMENDED GRID ACTION
# ============================================================
st.markdown("## 🎯 Recommended Grid Action")

if current_status == 'CRITICAL':
    action_color = "#E74C3C"
    action_icon_str = "🔴"
    action_title = "CRITICAL — Immediate Action Required"
    action_text = f"Reduce residential HVAC load by {reduction_rate_percent}% across coordination zones"
    expected = f"Expected peak reduction: {reduction_rate_percent * 0.9:.1f}% | Grid stabilization: HIGH confidence"
elif current_status == 'WARNING':
    action_color = "#F39C12"
    action_icon_str = "🟡"
    action_title = "WARNING — Prepare for Intervention"
    action_text = f"Pre-stage HVAC coordination. Initiate {reduction_rate_percent}% reduction if conditions escalate"
    expected = "Monitoring window: Next 2 to 4 hours | Pre-emptive coordination recommended"
else:
    action_color = "#2ECC71"
    action_icon_str = "🟢"
    action_title = "STABLE — No Action Required"
    action_text = "Grid operating within safe parameters. Continue monitoring."
    expected = "Grid Saver uses dual-confirmation logic — both Sense and Predict must independently confirm vulnerability before any intervention is triggered."

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
    vulnerability_level = 'CRITICAL' if current_score >= 70 else 'WARNING' if current_score >= 40 else 'STABLE'
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
            <li>24hr Risk Projection: <strong style='color: #9B59B6;'>{current_prob:.2f}</strong>
                (threshold: {DECISION_THRESHOLD})</li>
        </ul>
        <p style='color: #CCC; margin: 10px 0 5px 0;'>
            <strong>Recommended Action:</strong>
            <span style='color: white;'>{action_text}</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# SECTION 6 — SMART RECOMMENDATIONS (Operational Intelligence)
# ============================================================
st.markdown("## 🧠 Smart Recommendations")

# Dispatch priority score
dispatch_score = compute_dispatch_priority(current_row)
priority_label = "Immediate action required" if dispatch_score > 70 else "Monitor closely"
priority_color = "#E74C3C" if dispatch_score > 70 else "#F39C12" if dispatch_score > 40 else "#2ECC71"

st.markdown(f"""
<div style='background:#161B22; border:1px solid {priority_color};
     padding:12px 16px; border-radius:8px; margin-bottom:12px;'>
    <span style='color:{priority_color}; font-weight:bold; font-size:1rem;'>
        ⚡ Dispatch Priority Score: {dispatch_score}/100 — {priority_label}
    </span>
    <p style='color:#888; margin:5px 0 0 0; font-size:0.75rem;'>
        Formula: (vulnerability_score × 0.5) + (risk_probability × 30) + min(carbon/10, 20)
    </p>
</div>
""", unsafe_allow_html=True)

# Priority actions based on status
if current_status == 'CRITICAL':
    st.markdown(f"🔴 **Priority Action:** Execute {reduction_rate_percent}% HVAC reduction across {homes:,} homes immediately.")
    if current_row.get('spa_action_triggered', False):
        st.markdown("✅ SPA dual-confirmation achieved — full automated dispatch approved.")
    else:
        st.markdown("⚠️ Partial confirmation — deploy controlled reduction while monitoring.")
    st.markdown("📍 Target high-demand feeders and peak residential clusters.")
    st.markdown("📢 Notify grid operators and demand response aggregators immediately.")

elif current_status == 'WARNING':
    st.markdown("🟡 **Priority Action:** Pre-stage demand response resources for potential activation.")
    if current_prob >= 0.4:
        st.markdown("📈 Temporal risk signal elevated — prepare for activation within next peak window.")
    else:
        st.markdown("📊 Risk is moderate — maintain readiness and continue monitoring.")
    st.markdown("🔌 Shift flexible loads (EV charging, water heating) to off-peak hours.")
    st.markdown("📱 Issue advisory notifications to consumers in high-load zones.")

else:
    st.markdown("🟢 **Status:** No immediate intervention required.")
    if current_prob >= 0.4:
        st.markdown("📈 Temporal model indicates potential future risk — monitor upcoming peak window.")
    st.markdown("💡 Encourage off-peak consumption behaviour to maintain grid efficiency.")

# Impact summary
st.markdown(f"""
<div style='background:#161B22; border-left:4px solid #2ECC71;
     padding:10px 14px; border-radius:6px; margin-top:12px;'>
    <span style='color:#2ECC71; font-size:0.9rem;'>📊 <strong>Impact Summary</strong></span><br>
    <span style='color:#CCC; font-size:0.85rem;'>
        Per SPA event (1-hour window): {per_event_mwh:.2f} MWh | ${per_event_cost:,.0f} savings | {per_event_co2:.3f} tons CO₂ avoided<br>
        <strong>Current period:</strong> {SPA_EVENTS_VIEW} SPA events | {TOTAL_MWH_REMOVED:.2f} MWh total reduction
    </span>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "<p style='color:#555; font-size:0.78rem; margin-top:8px;'>"
    "📌 Cost estimate based on $100/MWh conservative grid baseline. "
    "CO₂ derived from current carbon intensity. "
    "Feeder-level targeting and real-time pricing integration available in production."
    "</p>",
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# SECTION 7 — GRID SAVER LOAD REDUCTION SIMULATION
# BEFORE vs AFTER with SPA markers, shaded zones, drop lines
# ============================================================
st.markdown("## ⚡ Grid Saver Load Reduction Simulation")

if reduction_rate_percent != 4:
    st.warning(
        f"⚠️ Note: Grid Saver is validated at 4% HVAC reduction (0.0920 kW per home). "
        f"Results at {reduction_rate_percent}% are proportionally scaled estimates, "
        f"not independently validated values."
    )

# Trigger counts - use correct event counting for display
if live_mode:
    display_sense = int(df_view['sense_triggered'].sum())
    display_predict = int(df_view['predict_triggered'].sum())
    display_spa = SPA_EVENTS_VIEW
    trigger_sub = "hours (last 24h)"
    spa_sub = "events (last 24h)"
else:
    display_sense = NOTEBOOK_SENSE_TRIGGERS
    display_predict = NOTEBOOK_PREDICT_TRIGGERS
    display_spa = FULL_YEAR_SPA_EVENTS
    trigger_sub = "hours (full year validated)"
    spa_sub = f"events (full year — {FULL_YEAR_SPA_EVENTS} validated)"

col_sim1, col_sim2, col_sim3 = st.columns(3)
sim_cards = [
    (col_sim1, f"{display_sense:,}", trigger_sub, "Sense Triggers", "#4A9EFF"),
    (col_sim2, f"{display_predict:,}", trigger_sub, "Predict Triggers", "#9B59B6"),
    (col_sim3, f"{display_spa:,}", spa_sub, "SPA Actions (Dual-Confirmed)", "#F39C12"),
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

# Total Energy Removed headline
st.markdown(f"""
<div style='background:#161B22; border-left:5px solid #2ECC71;
     padding:15px; border-radius:8px; margin-bottom:15px;'>
    <h3 style='color:#2ECC71; margin:0;'>⚡ Total Energy Removed (MWh)</h3>
    <p style='color:white; font-size:1.4rem; margin:5px 0;'>
        {TOTAL_MWH_REMOVED:,.2f} MWh — sum of hourly reductions across SPA-triggered events
    </p>
    <p style='color:#888; margin:0; font-size:0.85rem;'>
        Across {SPA_EVENTS_VIEW} SPA-triggered events in this period |
        Full-year validated: {FULL_YEAR_SPA_EVENTS} events
    </p>
</div>
""", unsafe_allow_html=True)

# Find peak for comparison
if 'synthetic_demand_mw' in df_view.columns:
    demand_col = 'synthetic_demand_mw'
    adjusted_col = 'adjusted_demand_mw'
else:
    demand_col = 'simulated_demand_mw'
    adjusted_col = 'adjusted_demand_mw'

peak_idx = df_view[demand_col].idxmax()
peak_time = df_view.loc[peak_idx, DATETIME_COL]
original_peak = df_view.loc[peak_idx, demand_col]
after_peak = df_view.loc[peak_idx, adjusted_col]
peak_reduction_mw = original_peak - after_peak
pct_reduction = (peak_reduction_mw / original_peak) * 100 if original_peak > 0 else 0

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

st.markdown(
    f"<p style='color:#555; font-size:0.8rem; margin-top:6px;'>"
    f"📌 Peak insight: Original {original_peak:,.0f} MW reduced to {after_peak:,.0f} MW "
    f"at peak demand timestamp. Load shed: {peak_reduction_mw:,.2f} MW ({pct_reduction:.2f}%). "
    f"Validated at 4% HVAC reduction (0.0920 kW per home). "
    f"Reduction occurs only during dual-confirmed SPA events — not continuously."
    f"</p>",
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# BEFORE PLOT (with SPA markers)
# ============================================================
st.markdown("#### 📉 Before Grid Saver — Original Demand with SPA Markers")

fig_before = go.Figure()
fig_before.add_trace(go.Scatter(
    x=df_view[DATETIME_COL],
    y=df_view[demand_col],
    mode='lines',
    name='Original Demand (Proxy)',
    line=dict(color='#E74C3C', width=1.5),
))

# SPA markers on BEFORE chart
if apply_intervention_enabled:
    spa_triggered = df_view[df_view['spa_action_triggered']]
    if not spa_triggered.empty:
        fig_before.add_trace(go.Scatter(
            x=spa_triggered[DATETIME_COL],
            y=spa_triggered[demand_col],
            mode='markers',
            name='SPA Action Triggered',
            marker=dict(color='#F39C12', size=8, symbol='circle', line=dict(width=1, color='white')),
        ))

# Mark peak timestamp
fig_before.add_trace(go.Scatter(
    x=[peak_time, peak_time],
    y=[df_view[demand_col].min(), df_view[demand_col].max()],
    mode='lines',
    name='Peak Demand',
    line=dict(color='#FFFFFF', width=1.5, dash='dash'),
))

fig_before.update_layout(
    paper_bgcolor='#161B22', plot_bgcolor='#161B22',
    font=dict(color='white'),
    xaxis=dict(gridcolor='#30363D', color='#888'),
    yaxis=dict(gridcolor='#30363D', color='#888', title='Demand Proxy (MW)'),
    legend=dict(bgcolor='#1A1A2E', bordercolor='#333'),
    height=350, margin=dict(t=50, b=20),
)
st.plotly_chart(fig_before, use_container_width=True)


# ============================================================
# AFTER PLOT (with shaded SPA zones + drop lines)
# ============================================================
st.markdown("#### 📈 After Grid Saver — Demand with Intervention Applied")

fig_after = go.Figure()

# Original demand (faded reference)
fig_after.add_trace(go.Scatter(
    x=df_view[DATETIME_COL],
    y=df_view[demand_col],
    mode='lines',
    name='Original Demand (Proxy)',
    line=dict(color='#E74C3C', width=1, dash='dot'),
    opacity=0.35,
))

# Adjusted demand after intervention
line_color = '#2ECC71' if apply_intervention_enabled else '#888'
fig_after.add_trace(go.Scatter(
    x=df_view[DATETIME_COL],
    y=df_view[adjusted_col],
    mode='lines',
    name='After Grid Saver Intervention' if apply_intervention_enabled else 'Intervention Disabled',
    line=dict(color=line_color, width=1.5),
    fill='tonexty' if apply_intervention_enabled else None,
    fillcolor='rgba(46, 204, 113, 0.08)' if apply_intervention_enabled else None,
))

# Shaded SPA zones and drop lines (only if intervention enabled)
if apply_intervention_enabled:
    spa_triggered = df_view[df_view['spa_action_triggered']].copy()
    
    # Shaded zones at ALL SPA-triggered hours
    for _, row in spa_triggered.iterrows():
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
                y=[row[demand_col], row[adjusted_col]],
                mode='lines',
                line=dict(color='#F39C12', width=2),
                showlegend=False,
            ))

# Mark peak timestamp
fig_after.add_trace(go.Scatter(
    x=[peak_time, peak_time],
    y=[df_view[demand_col].min(), df_view[demand_col].max()],
    mode='lines',
    name='Peak Demand',
    line=dict(color='#FFFFFF', width=1.5, dash='dash'),
    showlegend=False,
))

fig_after.update_layout(
    paper_bgcolor='#161B22', plot_bgcolor='#161B22',
    font=dict(color='white'),
    xaxis=dict(gridcolor='#30363D', color='#888'),
    yaxis=dict(gridcolor='#30363D', color='#888', title='Demand Proxy (MW)'),
    legend=dict(bgcolor='#1A1A2E', bordercolor='#333'),
    height=350, margin=dict(t=50, b=20),
)
st.plotly_chart(fig_after, use_container_width=True)

st.markdown("""
<div class='info-box'>
📌 <strong>Chart Guide:</strong><br>
• <span style='color:#F39C12'>🟠 SPA markers (BEFORE chart):</span> Hours where dual-confirmation triggered intervention<br>
• <span style='color:#FFA500'>🟧 Shaded zones (AFTER chart):</span> SPA-triggered windows (1-hour each)<br>
• <span style='color:#F39C12'>🟡 Orange drop lines:</span> Top 20 highest-impact interventions showing before→after reduction<br>
• <span style='color:#FFFFFF'>⚪ Dashed white line:</span> Peak demand timestamp<br>
• <span style='color:#888'>Note:</span> Drop lines and shaded zones only appear when intervention is enabled.
</div>
""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# PEAK ZOOM WINDOW (in expander)
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
            x=zoom_df[DATETIME_COL],
            y=zoom_df[demand_col],
            mode='lines',
            name='Original Demand (Proxy)',
            line=dict(color='#E74C3C', width=2),
        ))
        fig_zoom.add_trace(go.Scatter(
            x=zoom_df[DATETIME_COL],
            y=zoom_df[adjusted_col],
            mode='lines',
            name='After Grid Saver',
            line=dict(color='#2ECC71', width=2),
            fill='tonexty' if apply_intervention_enabled else None,
            fillcolor='rgba(46, 204, 113, 0.15)',
        ))
        
        if apply_intervention_enabled:
            for _, row in zoom_df[zoom_df['spa_action_triggered']].iterrows():
                fig_zoom.add_trace(go.Scatter(
                    x=[row[DATETIME_COL], row[DATETIME_COL]],
                    y=[row[demand_col], row[adjusted_col]],
                    mode='lines',
                    line=dict(color='#F39C12', width=2.5),
                    showlegend=False,
                ))
                fig_zoom.add_vrect(
                    x0=row[DATETIME_COL],
                    x1=row[DATETIME_COL] + pd.Timedelta(hours=1),
                    fillcolor='rgba(255, 165, 0, 0.2)',
                    line_width=0,
                )
        
        fig_zoom.add_trace(go.Scatter(
            x=[peak_time, peak_time],
            y=[zoom_df[demand_col].min(), zoom_df[demand_col].max()],
            mode='lines',
            name='Peak',
            line=dict(color='#FFFFFF', width=1.5, dash='dash'),
            showlegend=False,
        ))
        
        fig_zoom.update_layout(
            paper_bgcolor='#161B22', plot_bgcolor='#161B22',
            font=dict(color='white'),
            title=dict(
                text=f'Peak Window — Full Detail Around {peak_time.strftime("%Y-%m-%d %H:%M")}',
                font=dict(color='white', size=13)
            ),
            xaxis=dict(gridcolor='#30363D', color='#888'),
            yaxis=dict(gridcolor='#30363D', color='#888', title='Demand Proxy (MW)'),
            legend=dict(bgcolor='#1A1A2E', bordercolor='#333'),
            height=350, margin=dict(t=50, b=20),
        )
        st.plotly_chart(fig_zoom, use_container_width=True)
        
        st.markdown("""
        <div class='info-box'>
        🔍 <strong>Peak Window Calculation:</strong> ±6 hours from peak demand = 13 hours total<br>
        • <strong style='color:#E74C3C'>Red line:</strong> Original demand profile<br>
        • <strong style='color:#2ECC71'>Green line:</strong> Demand after Grid Saver intervention<br>
        • <strong style='color:#F39C12'>Orange drop lines:</strong> Individual SPA interventions with before→after reduction<br>
        • <strong style='color:#FFA500'>Shaded orange zones:</strong> SPA-triggered intervention windows
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# SECTION 8 — IMPACT AT SCALE
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

reserve_note = "Exceeds reserve margin" if scaled_reduction_mw > 200 else "Building toward reserve margin"
rm_color = "#2ECC71" if scaled_reduction_mw > 200 else "#F39C12"

col_s1, col_s2, col_s3, col_s4 = st.columns(4)
scale_cards = [
    (col_s1, f"{homes:,}", "Homes", "Homes Coordinated", "#4A9EFF"),
    (col_s2, f"{scaled_reduction_mw:,.1f} MW", "removed", "Projected Grid Reduction (per event)", "#2ECC71"),
    (col_s3, grid_impact, "", "Impact Level", impact_color),
    (col_s4, reserve_note, "", "Reserve Margin", rm_color),
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
    (col_a, "👁️", "SENSE", "#1B4F8C", "#4A9EFF",
     "Detect grid vulnerability signals continuously",
     "Carbon intensity monitoring",
     "Electricity Maps US-TEX-ERCO",
     "8,760 hourly records"),
    (col_b, "🧠", "PREDICT", "#1A6B2E", "#2ECC71",
     "Forecast vulnerability windows",
     "XGBoost model | 91.3% Recall (validated)",
     "PJM 32,896 hourly records",
     "24hr Risk Projection — temporal pattern only"),
    (col_c, "⚡", "ACT", "#7B1A1A", "#E74C3C",
     "Coordinate residential HVAC load reduction",
     "3-5% targeted precision",
     "Pecan Street 868,096 records",
     "SPA dual-confirmation safety"),
]

for col, icon, title, bg, color, l1, l2, l3, l4 in arch:
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
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# SECTION 10 — REPORTS AND INSIGHTS (in expander)
# ============================================================
with st.expander("📄 Reports and Insights (Download CSV)"):
    if live_mode:
        st.warning("Reports are disabled in Live Mode. Switch to Analysis Mode to generate reports.")
    else:
        st.markdown("*Select a time period to view grid performance analysis and download a report.*")
        
        col_r1, col_r2, col_r3 = st.columns(3)
        
        with col_r1:
            report_type = st.selectbox("Report Type", ["Yearly", "Monthly", "Weekly"], key="report_type")
        
        with col_r2:
            report_year = st.selectbox("Year", sorted(df['year'].unique(), reverse=True), key="report_year")
        
        with col_r3:
            if report_type == "Monthly":
                available_months = sorted(df[df['year'] == report_year]['month'].unique())
                month_display = [MONTH_NAMES[m] for m in available_months]
                selected_month_name = st.selectbox("Month", month_display, key="report_month")
                report_month_num = [k for k, v in MONTH_NAMES.items() if v == selected_month_name][0]
            elif report_type == "Weekly":
                available_weeks = sorted(df[df['year'] == report_year]['week'].unique())
                report_week = st.selectbox("Week", available_weeks, key="report_week")
            else:
                report_month_num = None
                report_week = None
        
        # Filter data
        try:
            if report_type == "Yearly":
                df_report = df[df['year'] == report_year].copy()
                period_label = f"{report_year}"
            elif report_type == "Monthly":
                df_report = df[(df['year'] == report_year) & (df['month'] == report_month_num)].copy()
                period_label = f"{MONTH_NAMES[report_month_num]} {report_year}"
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
        
        # SPA event count (rising edge detection) for report period
        spa_events_report = count_spa_events(df_report['spa_action_triggered'])
        total_mwh_report = df_report['reduction_mw'].sum() if 'reduction_mw' in df_report.columns else 0
        
        st.markdown(f"**Showing: {period_label}** — {len(df_report):,} hours analysed")
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Avg Vulnerability", f"{avg_vulnerability:.1f}")
        col_m2.metric("Peak Vulnerability", f"{peak_vulnerability:.1f}")
        col_m3.metric("SPA Events", f"{spa_events_report:,}")
        
        # Styled line chart for report (matching main dashboard style)
        fig_report = go.Figure()
        
        for status in ['STABLE', 'WARNING', 'CRITICAL']:
            mask = df_report['grid_status'] == status
            if mask.any():
                fig_report.add_trace(go.Scatter(
                    x=df_report[mask][DATETIME_COL],
                    y=df_report[mask]['vulnerability_score'],
                    mode='markers+lines',
                    name=status,
                    marker=dict(color=color_map[status], size=2, opacity=0.7),
                    line=dict(color=color_map[status], width=0.8),
                    connectgaps=True,
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
        📊 <strong>Period Summary:</strong> Average vulnerability {avg_vulnerability:.1f}/100. 
        {spa_events_report} SPA interventions triggered, saving {total_mwh_report:.2f} MWh.
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
        Built with Colab + GitHub + Streamlit
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
