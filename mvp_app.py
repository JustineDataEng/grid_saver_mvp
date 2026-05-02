import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Grid Saver | Adaptive Grid Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

CARBON_COL         = 'Carbon intensity gCO\u2082eq/kWh (direct)'
CFE_COL            = 'Carbon-free energy percentage (CFE%)'
DECISION_THRESHOLD = 0.4
REDUCTION_RATE     = 0.04
NUM_HOMES          = 25
KW_PER_HOME        = 0.0920

MONTH_NAMES = {1:'January',2:'February',3:'March',4:'April',5:'May',6:'June',
               7:'July',8:'August',9:'September',10:'October',11:'November',12:'December'}
DAY_NAMES   = {0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'}
month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

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

def engineer_features(df_input):
    df_fe = df_input.copy()
    df_fe['hour']        = df_fe['datetime'].dt.hour
    df_fe['day_of_week'] = df_fe['datetime'].dt.dayofweek
    df_fe['month']       = df_fe['datetime'].dt.month
    df_fe['day_of_year'] = df_fe['datetime'].dt.dayofyear
    df_fe['is_weekend']  = (df_fe['day_of_week'] >= 5).astype(int)
    df_fe['is_summer']   = df_fe['month'].isin([6,7,8]).astype(int)
    df_fe['is_winter']   = df_fe['month'].isin([12,1,2]).astype(int)
    df_fe['hour_sin']  = np.sin(2*np.pi*df_fe['hour']/24)
    df_fe['hour_cos']  = np.cos(2*np.pi*df_fe['hour']/24)
    df_fe['month_sin'] = np.sin(2*np.pi*df_fe['month']/12)
    df_fe['month_cos'] = np.cos(2*np.pi*df_fe['month']/12)
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
    'hour','day_of_week','month','day_of_year','is_weekend','is_summer','is_winter',
    'hour_sin','hour_cos','month_sin','month_cos',
    'demand_lag_1h','demand_lag_2h','demand_lag_24h','demand_lag_48h','demand_lag_168h',
    'demand_rolling_6h_mean','demand_rolling_24h_mean','demand_rolling_24h_max',
    'demand_rolling_24h_std','demand_delta_1h','demand_delta_24h'
]

def sense_layer(df_input):
    df_s = df_input.copy()
    carbon_max   = df_s[CARBON_COL].max()
    carbon_min   = df_s[CARBON_COL].min()
    cfe_max      = df_s[CFE_COL].max()
    carbon_denom = (carbon_max-carbon_min) if (carbon_max-carbon_min)!=0 else 1
    cfe_denom    = cfe_max if cfe_max!=0 else 1
    df_s['vulnerability_score'] = (
        ((df_s[CARBON_COL]-carbon_min)/carbon_denom*70) +
        ((1-df_s[CFE_COL]/cfe_denom)*30)
    ).round(1)
    VT = df_s['vulnerability_score'].quantile(0.85)
    df_s['vulnerability_event'] = df_s['vulnerability_score'] >= VT
    def classify(s):
        return 'CRITICAL' if s>=70 else 'WARNING' if s>=40 else 'STABLE'
    df_s['grid_status'] = df_s['vulnerability_score'].apply(classify)
    return df_s, VT

def predict_layer(df_input, model):
    """
    Independent Predict Layer — PJM time-based demand patterns only.
    No Sense Layer signals used. Layers meet only at SPA decision point.
    NOTE: demand_mw is a synthetic temporal signal (PJM baseline + seasonal/hour scaling).
    In production, replaced with a live SCADA feed.
    """
    df_out = df_input.copy()
    pjm_avg = 35000
    time_features = pd.DataFrame({
        'datetime':  df_input['Datetime (UTC)'],
        'demand_mw': pjm_avg + (
            np.where(df_input['Datetime (UTC)'].dt.month.isin([6,7,8]), 5000,
            np.where(df_input['Datetime (UTC)'].dt.month.isin([12,1,2]), 3000, 0))
            + np.where(df_input['Datetime (UTC)'].dt.hour.between(15,20), 2000,
              np.where(df_input['Datetime (UTC)'].dt.hour.between(6,9), 1000, -500))
        )
    })
    df_eng = engineer_features(time_features)
    if df_eng.empty or df_eng[FEATURE_COLS].isnull().any().any():
        df_out['vuln_probability']  = 0.0
        df_out['predict_triggered'] = False
        return df_out
    proba = model.predict_proba(df_eng[FEATURE_COLS])[:,1]
    df_eng['vuln_proba'] = proba
    df_eng['hour']  = df_eng['datetime'].dt.hour
    df_eng['month'] = df_eng['datetime'].dt.month
    pbhm = df_eng.groupby(['hour','month'])['vuln_proba'].mean().reset_index()
    pbhm['predict_triggered'] = pbhm['vuln_proba'] >= DECISION_THRESHOLD
    df_out = df_out.merge(pbhm[['hour','month','vuln_proba','predict_triggered']], on=['hour','month'], how='left')
    df_out = df_out.rename(columns={'vuln_proba':'vuln_probability'})
    df_out['vuln_probability']  = df_out['vuln_probability'].ffill().fillna(0)
    df_out['predict_triggered'] = df_out['predict_triggered'].ffill().fillna(False)
    return df_out

def act_layer(df_input, reduction_rate, homes):
    df_a = df_input.copy()
    df_a['sense_triggered']      = df_a['vulnerability_event']
    df_a['spa_action_triggered'] = df_a['sense_triggered'] & df_a['predict_triggered']
    skph = KW_PER_HOME * (reduction_rate/4)
    df_a['grid_saver_reduction_kw'] = np.where(df_a['spa_action_triggered'], homes*skph, 0)
    return df_a, homes*skph/1000

def compute_impact_metrics(row, homes, reduction_rate):
    """
    Per-event impact using homes slider for projection.
    Validated base: 25 homes × 0.0920 kW = 0.0023 MW = 0.0023 MWh per event.
    Cost baseline: $100/MWh. CO2 from actual carbon intensity.
    """
    scaled_mw        = homes * KW_PER_HOME / 1000   # pure homes × 0.0920
    mwh_saved        = scaled_mw * 1
    cost_savings     = mwh_saved * 100
    co2_avoided_tons = (row[CARBON_COL] * mwh_saved) / 1000
    return mwh_saved, cost_savings, co2_avoided_tons

def compute_dispatch_priority(row):
    score  = row['vulnerability_score'] * 0.5
    score += row.get('vuln_probability', 0) * 30
    score += min(row[CARBON_COL]/10, 20)
    return round(score, 1)

def get_risk_drivers(row, vt, df_full):
    drivers = []
    score   = row['vulnerability_score']
    carbon  = row[CARBON_COL]
    cfe     = row[CFE_COL]
    if score>=70: drivers.append("🔴 Grid is in CRITICAL state — vulnerability score at 70 or above")
    elif score>=40: drivers.append("🟡 Grid is in WARNING state — vulnerability score between 40 and 69")
    else: drivers.append("🟢 Grid is STABLE — vulnerability score below 40")
    cr  = df_full[CARBON_COL].max()-df_full[CARBON_COL].min()
    cpct = (carbon-df_full[CARBON_COL].min())/(cr if cr!=0 else 1)
    cfe_max = df_full[CFE_COL].max()
    cpct2   = cfe/cfe_max if cfe_max!=0 else 0
    if cpct>0.7: drivers.append("🔴 Carbon intensity in upper 30% of observed range — grid running heavily on fossil fuels")
    elif cpct>0.4: drivers.append("🟡 Carbon intensity above mid-range — fossil fuel contribution elevated")
    else: drivers.append("🟢 Carbon intensity in lower range — cleaner generation mix")
    if cpct2<0.3: drivers.append("🔴 Carbon-free energy in lower 30% of observed range — clean supply buffer low")
    elif cpct2<0.6: drivers.append("🟡 Carbon-free energy below mid-range — moderate clean supply")
    else: drivers.append("🟢 Carbon-free energy is strong — healthy clean supply buffer")
    return drivers

# Run pipeline
df, VULNERABILITY_THRESHOLD = sense_layer(df)
df['hour']        = df['Datetime (UTC)'].dt.hour
df['month']       = df['Datetime (UTC)'].dt.month
df['month_name']  = df['Datetime (UTC)'].dt.strftime('%b')
df['date']        = df['Datetime (UTC)'].dt.date
df['day_of_week'] = df['Datetime (UTC)'].dt.dayofweek
df['year']        = df['Datetime (UTC)'].dt.year
df['week']        = df['Datetime (UTC)'].dt.isocalendar().week.astype(int)
df = predict_layer(df, model)
df_full, _ = act_layer(df, REDUCTION_RATE, NUM_HOMES)

NOTEBOOK_SENSE_TRIGGERS   = 1316
NOTEBOOK_PREDICT_TRIGGERS = 1659
NOTEBOOK_SPA_ACTIONS      = 154

# Sidebar
st.sidebar.image("https://img.icons8.com/fluency/96/lightning-bolt.png", width=60)
st.sidebar.title("Grid Saver")
st.sidebar.markdown("**Adaptive Grid Intelligence Platform**")
st.sidebar.divider()

live_mode = st.sidebar.toggle("Recent Window View (Last 24 Hours)", value=False)
if live_mode:
    st.sidebar.markdown("<p style='color:#2ECC71;font-size:0.8rem;'>Showing the most recent 24 hours</p>", unsafe_allow_html=True)

st.sidebar.divider()
months_present       = [m for m in month_order if m in df['month_name'].unique()]
selected_month       = st.sidebar.selectbox("Select Month", ['All Year']+months_present)
reduction_rate_input = st.sidebar.slider("HVAC Reduction Rate (%) — Validated target: 3 to 5%", 1, 10, 4, 1)
homes                = st.sidebar.slider("Homes Coordinated", 1000, 1000000, 100000, 1000)
apply_intervention   = st.sidebar.toggle("Apply Grid Saver Intervention", value=True)

st.sidebar.divider()
with st.sidebar.expander("Dataset Information"):
    st.write("""
    **Sense Layer:** Electricity Maps US-TEX-ERCO 2025 — 8,760 hourly records.
    **Predict Layer:** PJM Interconnection 1998–2002 — 32,896 hourly records. XGBoost. 91.6% Recall.
    **Act Layer:** Pecan Street Inc. Austin TX 2018 — 25 households. 868,096 records.
    Note: Processed output from prototype notebooks. Raw datasets not stored here.
    Predict Layer uses synthetic temporal baseline (PJM mean ± seasonal/hour scaling).
    In production, replaced with live SCADA feed.
    """)
st.sidebar.divider()
st.sidebar.markdown("**Stack:** Colab + GitHub + Streamlit")
st.sidebar.divider()
st.sidebar.markdown("*Justine Adzormado*")

# Filter
if live_mode:
    df_view = df[df['Datetime (UTC)'] >= df['Datetime (UTC)'].max()-pd.Timedelta(hours=24)].copy()
elif selected_month != 'All Year':
    df_view = df[df['month_name']==selected_month].copy()
else:
    df_view = df.copy()

if df_view.empty:
    st.warning("No data available for selected filters.")
    st.stop()

df_view, total_mw_saved = act_layer(df_view, reduction_rate_input, homes)

# Header
mode_label = "RECENT WINDOW VIEW" if live_mode else "ANALYSIS MODE"
mode_color = "#2ECC71" if live_mode else "#4A9EFF"
st.markdown(f"""
<div style='background:linear-gradient(135deg,#1B4F8C,#0D1117);padding:30px;border-radius:12px;
     margin-bottom:20px;border:1px solid #30363D;'>
  <div style='display:flex;justify-content:space-between;align-items:center;'>
    <div>
      <h1 style='color:white;margin:0;font-size:2.2rem;'>⚡ Grid Saver</h1>
      <p style='color:#4A9EFF;margin:5px 0 0 0;font-size:1.1rem;'>Adaptive Grid Intelligence Platform</p>
      <p style='color:#888;margin:5px 0 0 0;font-size:0.9rem;'>Texas ERCOT 2025 | SPA Logic | Dual-Confirmation Architecture</p>
    </div>
    <div style='background:{mode_color}22;border:2px solid {mode_color};padding:10px 20px;border-radius:8px;text-align:center;'>
      <p style='color:{mode_color};font-weight:bold;margin:0;font-size:1rem;'>{"🕐 " if live_mode else "📊 "}{mode_label}</p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Section 1 — Grid Status
st.markdown("## ⚡ Grid Status")
current_row    = df_view.iloc[-1]
current_score  = current_row['vulnerability_score']
current_status = current_row['grid_status']
current_carbon = current_row[CARBON_COL]
current_cfe    = current_row[CFE_COL]
current_prob   = current_row['vuln_probability']
vulnerable_pct = df_view['vulnerability_event'].mean()*100

status_color = {'STABLE':'#2ECC71','WARNING':'#F39C12','CRITICAL':'#E74C3C'}
status_icon  = {'STABLE':'🟢','WARNING':'🟡','CRITICAL':'🔴'}

col1,col2,col3,col4,col5,col6 = st.columns(6)
for col,val,sub,label,color in [
    (col1, status_icon[current_status], current_status,  "Grid Status",                             status_color[current_status]),
    (col2, f"{current_score:.0f}",      "/100",           "Vulnerability Score",                     "white"),
    (col3, f"{current_carbon:.0f}",     "gCO\u2082/kWh", "Carbon Intensity",                        "#E74C3C"),
    (col4, f"{current_cfe:.1f}%",       "clean energy",   "Carbon-Free Energy",                      "#2ECC71"),
    (col5, f"{vulnerable_pct:.1f}%",    "of period",      "Vulnerability Rate",                      "#4A9EFF"),
    (col6, f"{current_prob:.2f}",       "probability",    "24hr Risk Projection (Temporal Pattern)", "#9B59B6"),
]:
    with col:
        st.markdown(f"""
        <div class='metric-card'>
            <h2 style='color:{color};font-size:1.6rem;margin:0;'>{val}</h2>
            <p style='color:#666;margin:2px 0;font-size:0.75rem;'>{sub}</p>
            <p style='color:#888;margin:0;font-size:0.75rem;'>{label}</p>
        </div>""", unsafe_allow_html=True)

st.markdown("<p style='color:#555;font-size:0.78rem;margin-top:4px;'>24hr Risk Projection: XGBoost pattern-recognition of historical grid stress cycles. Not a rolling live forecast — temporal vulnerability window based on learned PJM demand patterns (1998–2002).</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Section 2 — Risk Drivers
st.markdown("## Risk Drivers")
drivers = get_risk_drivers(current_row, VULNERABILITY_THRESHOLD, df)
col_d1, col_d2 = st.columns(2)
for i, driver in enumerate(drivers):
    (col_d1 if i%2==0 else col_d2).markdown(f"**{driver}**")
st.markdown("<br>", unsafe_allow_html=True)

# Section 3 — Vulnerability Chart
st.markdown("## Grid Demand and Vulnerability Windows")
color_map  = {'STABLE':'#2ECC71','WARNING':'#F39C12','CRITICAL':'#E74C3C'}
fig_demand = go.Figure()
for status in ['STABLE','WARNING','CRITICAL']:
    mask = df_view['grid_status']==status
    if mask.any():
        fig_demand.add_trace(go.Scatter(
            x=df_view[mask]['Datetime (UTC)'], y=df_view[mask]['vulnerability_score'],
            mode='markers+lines', name=status,
            marker=dict(color=color_map[status],size=3,opacity=0.8),
            line=dict(color=color_map[status],width=0.8), connectgaps=True,
        ))
fig_demand.add_hline(y=VULNERABILITY_THRESHOLD, line_dash='dash', line_color='#FF4444',
    annotation_text=f'Threshold ({VULNERABILITY_THRESHOLD:.0f})', annotation_font_color='#FF4444')
fig_demand.update_layout(
    paper_bgcolor='#161B22', plot_bgcolor='#161B22', font=dict(color='white'),
    title=dict(text='Last 24h Grid Vulnerability Score' if live_mode else 'Grid Vulnerability Score — ERCOT Texas 2025', font=dict(color='white',size=14)),
    xaxis=dict(gridcolor='#30363D',color='#888'),
    yaxis=dict(gridcolor='#30363D',color='#888',title='Vulnerability Score (0 to 100)'),
    legend=dict(bgcolor='#1A1A2E',bordercolor='#333'), height=350, margin=dict(t=50,b=30),
)
st.plotly_chart(fig_demand, use_container_width=True)

# Section 4 — Peak Timeline
st.markdown("## Peak Vulnerability Timeline")
col_left, col_right = st.columns(2)

with col_left:
    hv = df_view.groupby('hour')['vulnerability_score'].mean().round(1)
    fig_hour = go.Figure(go.Bar(
        x=[f'{h:02d}:00' for h in hv.index], y=hv.values,
        marker_color=['#E74C3C' if s>=70 else '#F39C12' if s>=40 else '#2ECC71' for s in hv.values],
    ))
    fig_hour.add_hline(y=VULNERABILITY_THRESHOLD, line_dash='dash', line_color='#FF4444',
        annotation_text=f'Threshold ({VULNERABILITY_THRESHOLD:.0f})', annotation_font_color='#FF4444')
    fig_hour.update_layout(paper_bgcolor='#161B22', plot_bgcolor='#161B22', font=dict(color='white'),
        title=dict(text='Avg Grid Vulnerability by Hour of Day', font=dict(color='white',size=13)),
        xaxis=dict(gridcolor='#30363D',color='#888',title='Hour (UTC)'),
        yaxis=dict(gridcolor='#30363D',color='#888',title='Vulnerability Score'),
        height=300, margin=dict(t=50,b=30))
    st.plotly_chart(fig_hour, use_container_width=True)

with col_right:
    if live_mode:
        st.info("Monthly trend requires full-year data. Switch to Analysis Mode.")
    elif selected_month != 'All Year':
        dv = df_view.groupby('date')['vulnerability_score'].mean().round(1).sort_index()
        fig_daily = go.Figure(go.Bar(x=[str(d) for d in dv.index], y=dv.values,
            marker_color=['#E74C3C' if s>=70 else '#F39C12' if s>=40 else '#2ECC71' for s in dv.values]))
        fig_daily.update_layout(paper_bgcolor='#161B22', plot_bgcolor='#161B22', font=dict(color='white'),
            title=dict(text=f'Avg Daily Vulnerability — {selected_month}', font=dict(color='white',size=13)),
            xaxis=dict(gridcolor='#30363D',color='#888',title='Date',tickangle=45),
            yaxis=dict(gridcolor='#30363D',color='#888',title='Vulnerability Score'),
            height=300, margin=dict(t=60,b=60))
        st.plotly_chart(fig_daily, use_container_width=True)
    else:
        mv = df_view.groupby('month_name')['vulnerability_score'].mean().round(1)
        mv = mv.reindex([m for m in month_order if m in mv.index])
        fig_month = go.Figure(go.Bar(x=mv.index, y=mv.values,
            marker_color=['#E74C3C' if s>=70 else '#F39C12' if s>=40 else '#2ECC71' for s in mv.values]))
        fig_month.update_layout(paper_bgcolor='#161B22', plot_bgcolor='#161B22', font=dict(color='white'),
            title=dict(text='Avg Grid Vulnerability by Month', font=dict(color='white',size=13)),
            xaxis=dict(gridcolor='#30363D',color='#888',title='Month'),
            yaxis=dict(gridcolor='#30363D',color='#888',title='Vulnerability Score'),
            height=300, margin=dict(t=50,b=30))
        st.plotly_chart(fig_month, use_container_width=True)

# Section 5 — Recommended Action
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("## Recommended Grid Action")
if current_status == 'CRITICAL':
    action_color,action_icon_str = "#E74C3C","🔴"
    action_title = "CRITICAL — Immediate Action Required"
    action_text  = f"Reduce residential HVAC load by {reduction_rate_input}% across coordination zones"
    expected     = f"Expected peak reduction: {reduction_rate_input*0.9:.1f}% | Grid stabilization: HIGH confidence"
elif current_status == 'WARNING':
    action_color,action_icon_str = "#F39C12","🟡"
    action_title = "WARNING — Prepare for Intervention"
    action_text  = f"Pre-stage HVAC coordination. Initiate {reduction_rate_input}% reduction if conditions escalate"
    expected     = "Monitoring window: Next 2 to 4 hours | Pre-emptive coordination recommended"
else:
    action_color,action_icon_str = "#2ECC71","🟢"
    action_title = "STABLE — No Action Required"
    action_text  = "Grid operating within safe parameters. Continue monitoring."
    expected     = "Grid Saver uses dual-confirmation logic — both Sense and Predict must independently confirm vulnerability before any intervention is triggered."

st.markdown(f"""
<div style='background:#161B22;border-left:5px solid {action_color};padding:20px;border-radius:8px;margin:10px 0;'>
    <h3 style='color:{action_color};margin:0;'>{action_icon_str} {action_title}</h3>
    <p style='color:white;margin:10px 0 5px 0;font-size:1rem;'><strong>Recommended Action:</strong> {action_text}</p>
    <p style='color:#888;margin:0;font-size:0.9rem;'>{expected}</p>
</div>""", unsafe_allow_html=True)

if st.button("🧠 Explain Grid Decision"):
    vlevel = 'CRITICAL' if current_score>=70 else 'WARNING' if current_score>=40 else 'STABLE'
    st.markdown(f"""
    <div style='background:#161B22;border:1px solid #1B4F8C;padding:25px;border-radius:10px;margin-top:15px;'>
        <h3 style='color:#4A9EFF;margin:0 0 15px 0;'>AI Decision Explanation</h3>
        <p style='color:#CCC;margin:5px 0;'>Grid Saver classified the system as <strong style='color:{action_color};'>{current_status}</strong> based on:</p>
        <ul style='color:#CCC;margin:10px 0;'>
            <li>Vulnerability Score: <strong style='color:white;'>{current_score:.1f}/100</strong> (threshold: {VULNERABILITY_THRESHOLD:.0f})</li>
            <li>Carbon Intensity: <strong style='color:#FF6B6B;'>{current_carbon:.0f} gCO\u2082eq/kWh</strong></li>
            <li>Carbon-Free Energy: <strong style='color:#2ECC71;'>{current_cfe:.1f}%</strong></li>
            <li>24hr Risk Projection: <strong style='color:#9B59B6;'>{current_prob:.2f}</strong> (threshold: {DECISION_THRESHOLD})</li>
            <li>Vulnerability Level: <strong style='color:{action_color};'>{vlevel}</strong></li>
        </ul>
        <p style='color:#CCC;margin:10px 0 5px 0;'><strong>Recommended Action:</strong> <span style='color:white;'>{action_text}</span></p>
        <p style='color:#888;margin:0;font-size:0.9rem;'>{expected}</p>
        <p style='color:#555;margin:15px 0 0 0;font-size:0.8rem;'>A {reduction_rate_input}% HVAC reduction across {homes:,} homes removes approximately {homes*KW_PER_HOME*(reduction_rate_input/4):,.1f} kW per SPA event (1-hour window) from peak demand.</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Smart Recommendations
st.markdown("## Smart Recommendations")
dispatch_score                       = compute_dispatch_priority(current_row)
mwh_saved, cost_savings, co2_avoided = compute_impact_metrics(current_row, homes, reduction_rate_input)
recommendations = []
spa_current    = current_row.get('spa_action_triggered', False)
priority_label = "Immediate action required" if dispatch_score>70 else "Monitor closely"
recommendations.append((f"Dispatch Priority Score: {dispatch_score}/100 — {priority_label}", "priority"))

if current_status == 'CRITICAL':
    recommendations.append((f"Execute {reduction_rate_input}% HVAC reduction across {homes:,} homes immediately.", "action"))
    recommendations.append(("SPA dual-confirmation achieved — full automated dispatch approved." if spa_current else "Partial confirmation — deploy controlled reduction while monitoring.", "action"))
    recommendations.append(("Target high-demand feeders and peak residential clusters.", "detail"))
    recommendations.append(("Notify grid operators and demand response aggregators immediately.", "detail"))
elif current_status == 'WARNING':
    recommendations.append(("Pre-stage demand response resources for potential activation.", "action"))
    recommendations.append(("Temporal risk signal elevated — prepare for activation within next peak window." if current_prob>=0.4 else "Risk is moderate — maintain readiness and continue monitoring.", "action" if current_prob>=0.4 else "detail"))
    recommendations.append(("Shift flexible loads (EV charging, water heating) to off-peak hours.", "detail"))
    recommendations.append(("Issue advisory notifications to consumers in high-load zones.", "detail"))
else:
    recommendations.append(("No immediate intervention required.", "action"))
    if current_prob>=0.4:
        recommendations.append(("Temporal model indicates potential future risk — monitor upcoming peak window.", "detail"))
    recommendations.append(("Encourage off-peak consumption behaviour to maintain grid efficiency.", "detail"))
    recommendations.append(("Maintain baseline grid monitoring operations.", "detail"))

recommendations.append((f"Estimated impact per SPA event (1-hour window): {mwh_saved:.2f} MWh reduced | ${cost_savings:,.0f} cost savings | {co2_avoided:.3f} tons CO\u2082 avoided", "impact"))

for i, (rec, rec_type) in enumerate(recommendations):
    if rec_type == "priority":
        pc = "#E74C3C" if dispatch_score>70 else "#F39C12" if dispatch_score>40 else "#2ECC71"
        st.markdown(f"<div style='background:#161B22;border:1px solid {pc};padding:12px 16px;border-radius:8px;margin-bottom:8px;'><span style='color:{pc};font-weight:bold;font-size:1rem;'>⚡ {rec}</span></div>", unsafe_allow_html=True)
    elif rec_type == "action":
        st.markdown(f"🔴 **Priority Action:** {rec}" if i==1 else f"🔸 **{rec}**")
    elif rec_type == "impact":
        st.markdown(f"<div style='background:#161B22;border-left:4px solid #2ECC71;padding:10px 14px;border-radius:6px;margin-top:10px;'><span style='color:#2ECC71;font-size:0.9rem;'>📊 {rec}</span></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"• {rec}")

st.markdown("<p style='color:#555;font-size:0.78rem;margin-top:8px;'>Cost estimate based on $100/MWh conservative grid baseline. CO\u2082 derived from current carbon intensity. Feeder-level targeting and real-time pricing integration available in production.</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# SECTION 6 — GRID SAVER LOAD REDUCTION SIMULATION
# Purpose: prove the architecture works on real Pecan Street data
# 25 homes × 0.0920 kW = 2.3 kW real reduction
# Scaled for visual clarity on ERCOT grid scale (~75,000 MW)
# ============================================================
st.markdown("## Grid Saver Load Reduction Simulation")

# SPA trigger counts — locked to notebook validated figures in analysis mode
if live_mode:
    display_sense   = int(df_view['sense_triggered'].sum()) if 'sense_triggered' in df_view.columns else 0
    display_predict = int(df_view['predict_triggered'].sum())
    # Count events (0→1 transitions), not hours
    spa_series      = df_view['spa_action_triggered'].astype(int) if 'spa_action_triggered' in df_view.columns else pd.Series([0])
    display_spa     = int((spa_series.diff() == 1).sum())
    trigger_sub,spa_sub = "hours (last 24h)","events (last 24h)"
else:
    display_sense,display_predict,display_spa = NOTEBOOK_SENSE_TRIGGERS,NOTEBOOK_PREDICT_TRIGGERS,NOTEBOOK_SPA_ACTIONS
    trigger_sub,spa_sub = "hours (full year validated)","events (full year validated)"

col_sim1,col_sim2,col_sim3 = st.columns(3)
for col,val,sub,label,color in [
    (col_sim1,f"{display_sense:,}",  trigger_sub,"Sense Triggers",              "#4A9EFF"),
    (col_sim2,f"{display_predict:,}",trigger_sub,"Predict Triggers",            "#9B59B6"),
    (col_sim3,f"{display_spa:,}",    spa_sub,    "SPA Actions (Dual-Confirmed)","#F39C12"),
]:
    with col:
        st.markdown(f"<div class='metric-card'><h2 style='color:{color};font-size:1.6rem;margin:0;'>{val}</h2><p style='color:#666;margin:2px 0;font-size:0.75rem;'>{sub}</p><p style='color:#888;margin:0;font-size:0.75rem;'>{label}</p></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# SIMULATION LOGIC
# Real values from Phase 3 Pecan Street validation:
#   25 homes × 0.0920 kW = 2.3 kW = 0.0023 MW real reduction
#   2.2% peak reduction validated across 25 Austin TX households
#
# Grid reference: ERCOT peak ~75,000 MW
# Scaling: apply 2.2% reduction to ERCOT reference for visibility
#   75,000 MW × 2.2% = 1,650 MW scaled reduction
# This is a visual representation — labeled clearly below
# ============================================================
ERCOT_PEAK_REF   = 75000   # MW — ERCOT grid scale reference
VALIDATED_PCT    = 0.022   # 2.2% — Phase 3 Pecan Street validated reduction
REAL_REDUCTION_KW = NUM_HOMES * KW_PER_HOME  # 25 × 0.0920 = 2.3 kW
REAL_REDUCTION_MW = REAL_REDUCTION_KW / 1000  # 0.0023 MW

# Scaled reduction for visualization on grid scale
VISUAL_REDUCTION_MW = ERCOT_PEAK_REF * VALIDATED_PCT  # 1,650 MW

# Build simulation signal from ERCOT timestamps
# Demand curve uses temporal pattern — peaks at 15:00-20:00 (same as Predict Layer)
sim_hours_arr = df_view['hour'].values
sim_demand = ERCOT_PEAK_REF * (
    0.80 + 0.10 * np.sin((sim_hours_arr / 24) * np.pi) +
    0.10 * np.where((sim_hours_arr >= 15) & (sim_hours_arr <= 20), 1, 0)
)

# Apply scaled reduction ONLY during SPA-triggered hours
spa_mask = df_view['spa_action_triggered'].values
sim_reduction = np.where(spa_mask, VISUAL_REDUCTION_MW, 0)
sim_adjusted  = sim_demand - sim_reduction

# Peak metrics at the highest-demand SPA-triggered hour
spa_indices = np.where(spa_mask)[0]
if len(spa_indices) > 0:
    peak_sim_idx      = spa_indices[np.argmax(sim_demand[spa_indices])]
    sim_original_peak = sim_demand[peak_sim_idx]
    sim_after_peak    = sim_adjusted[peak_sim_idx]
    sim_peak_shed     = sim_original_peak - sim_after_peak
    sim_pct           = (sim_peak_shed / sim_original_peak) * 100
else:
    sim_original_peak = sim_demand.max()
    sim_after_peak    = sim_original_peak
    sim_peak_shed     = 0.0
    sim_pct           = 0.0

# Peak time for vertical marker
peak_sim_time = df_view.iloc[int(np.argmax(sim_demand))]['Datetime (UTC)']

# Peak cards
col_p1,col_p2,col_p3,col_p4 = st.columns(4)
for col,val,label,color in [
    (col_p1,f"{sim_original_peak:,.0f} MW",  "Original Peak (ERCOT ref)",   "white"),
    (col_p2,f"{sim_after_peak:,.0f} MW",     "After Grid Saver",            "#2ECC71"),
    (col_p3,f"{sim_pct:.1f}%",               "Peak Demand Reduction",       "#4A9EFF"),
    (col_p4,f"{sim_peak_shed:,.0f} MW",      "Peak Load Shed (Scaled)",     "#F39C12"),
]:
    with col:
        st.markdown(f"<div class='metric-card'><h2 style='color:{color};font-size:1.4rem;margin:0;'>{val}</h2><p style='color:#888;margin:4px 0 0 0;font-size:0.78rem;'>{label}</p></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Before plot
fig_before = go.Figure()
fig_before.add_trace(go.Scatter(
    x=df_view['Datetime (UTC)'], y=sim_demand,
    mode='lines', name='Grid Demand (ERCOT scale)',
    line=dict(color='#E74C3C',width=1.5)))
fig_before.add_trace(go.Scatter(
    x=df_view[df_view['spa_action_triggered']]['Datetime (UTC)'],
    y=sim_demand[spa_mask],
    mode='markers', name='SPA Action Triggered',
    marker=dict(color='#F39C12',size=6,symbol='circle')))
fig_before.add_trace(go.Scatter(
    x=[peak_sim_time, peak_sim_time],
    y=[sim_demand.min(), sim_demand.max()],
    mode='lines', name='Peak Demand',
    line=dict(color='#FFFFFF',width=1.5,dash='dash'), showlegend=True))
fig_before.update_layout(
    paper_bgcolor='#161B22', plot_bgcolor='#161B22', font=dict(color='white'),
    title=dict(text='Before Grid Saver — Grid Demand at ERCOT Scale', font=dict(color='white',size=13)),
    xaxis=dict(gridcolor='#30363D',color='#888'),
    yaxis=dict(gridcolor='#30363D',color='#888',title='Demand (MW)'),
    legend=dict(bgcolor='#1A1A2E',bordercolor='#333'), height=320, margin=dict(t=50,b=20))
st.plotly_chart(fig_before, use_container_width=True)

# After plot
fig_after = go.Figure()
fig_after.add_trace(go.Scatter(
    x=df_view['Datetime (UTC)'], y=sim_demand,
    mode='lines', name='Original Demand',
    line=dict(color='#E74C3C',width=1,dash='dot'), opacity=0.35))
fig_after.add_trace(go.Scatter(
    x=df_view['Datetime (UTC)'], y=sim_adjusted,
    mode='lines', name='After Grid Saver Intervention',
    line=dict(color='#2ECC71',width=1.5),
    fill='tonexty', fillcolor='rgba(46,204,113,0.08)'))

# Shaded SPA zones
spa_triggered_df = df_view[df_view['spa_action_triggered']].copy()
for _, row in spa_triggered_df.iterrows():
    try:
        fig_after.add_vrect(
            x0=row['Datetime (UTC)'],
            x1=row['Datetime (UTC)']+pd.Timedelta(hours=1),
            fillcolor='rgba(255,165,0,0.15)', line_width=0)
    except Exception:
        pass

# Drop lines — top 20 highest demand SPA hours
if len(spa_indices) > 0:
    top20_idx = spa_indices[np.argsort(sim_demand[spa_indices])[-20:]]
    for idx in top20_idx:
        t = df_view.iloc[idx]['Datetime (UTC)']
        fig_after.add_trace(go.Scatter(
            x=[t, t], y=[sim_demand[idx], sim_adjusted[idx]],
            mode='lines', line=dict(color='#F39C12',width=2), showlegend=False))

fig_after.add_trace(go.Scatter(
    x=[peak_sim_time, peak_sim_time],
    y=[sim_demand.min(), sim_demand.max()],
    mode='lines', name='Peak Demand',
    line=dict(color='#FFFFFF',width=1.5,dash='dash'), showlegend=False))

fig_after.update_layout(
    paper_bgcolor='#161B22', plot_bgcolor='#161B22', font=dict(color='white'),
    title=dict(text='After Grid Saver — Demand with Intervention Applied', font=dict(color='white',size=13)),
    xaxis=dict(gridcolor='#30363D',color='#888'),
    yaxis=dict(gridcolor='#30363D',color='#888',title='Demand (MW)'),
    legend=dict(bgcolor='#1A1A2E',bordercolor='#333'), height=320, margin=dict(t=50,b=20))
st.plotly_chart(fig_after, use_container_width=True)

st.markdown("""
<p style='color:#888;font-size:0.82rem;margin-top:4px;'>
<strong>Simulation note:</strong> HVAC load reduction is scaled for visualization clarity on grid scale.
Real-world validation: 25 Pecan Street households (Austin TX 2018) — 2.2% peak demand reduction per SPA event
(0.0920 kW per home, 2.3 kW total across 25 homes).
Reduction applied only during dual-confirmed SPA events (Sense + Predict both triggered).
Shaded zones mark all SPA-triggered windows. Drop lines show top 20 highest-demand intervention points.
</p>""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Section 7 — Impact at Scale
st.markdown("## Impact at Scale")
st.markdown("*Adjust the Homes Coordinated slider in the sidebar to see how Grid Saver scales.*")
# Pure math — validated 0.0920 kW per home only, no reduction_rate_input dependency
scaled_reduction_mw = homes * KW_PER_HOME / 1000
if homes<50000: grid_impact,impact_color = "Neighbourhood Scale","#F39C12"
elif homes<250000: grid_impact,impact_color = "District Scale","#4A9EFF"
elif homes<600000: grid_impact,impact_color = "Regional Scale","#9B59B6"
else: grid_impact,impact_color = "National Scale","#2ECC71"
reserve_note = "Exceeds reserve margin" if scaled_reduction_mw>200 else "Building toward reserve margin"
rm_color     = "#2ECC71" if scaled_reduction_mw>200 else "#F39C12"

col_s1,col_s2,col_s3,col_s4 = st.columns(4)
for col,val,sub,label,color in [
    (col_s1,f"{homes:,}","Homes","Homes Coordinated","#4A9EFF"),
    (col_s2,f"{scaled_reduction_mw:,.1f} MW","removed","Grid Reduction","#2ECC71"),
    (col_s3,grid_impact,"","Impact Level",impact_color),
    (col_s4,reserve_note,"","Reserve Margin",rm_color),
]:
    with col:
        st.markdown(f"<div class='metric-card'><h2 style='color:{color};font-size:1.3rem;margin:0;'>{val}</h2><p style='color:#666;margin:2px 0;font-size:0.75rem;'>{sub}</p><p style='color:#888;margin:0;font-size:0.75rem;'>{label}</p></div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Section 8 — Architecture
st.markdown("## System Architecture")
col_a,col_b,col_c = st.columns(3)
for col,icon,title,bg,color,l1,l2,l3,l4 in [
    (col_a,"\U0001f441\ufe0f","SENSE","#1B4F8C","#4A9EFF","Detect grid vulnerability signals continuously","Carbon intensity monitoring","Electricity Maps US-TEX-ERCO","8,760 hourly records"),
    (col_b,"\U0001f9e0","PREDICT","#1A6B2E","#2ECC71","Forecast vulnerability windows","XGBoost model | 91.6% Recall","PJM 32,896 hourly records","24hr Risk Projection — XGBoost pattern-recognition of historical grid cycles"),
    (col_c,"⚡","ACT","#7B1A1A","#E74C3C","Coordinate residential HVAC load reduction","3 to 5% targeted precision","Pecan Street 868,096 records","Human-override safety protocol"),
]:
    with col:
        st.markdown(f"<div style='background:{bg}22;border:1px solid {color};padding:20px;border-radius:10px;text-align:center;border-top:4px solid {color};'><h2 style='color:{color};font-size:2rem;margin:0;'>{icon}</h2><h3 style='color:{color};margin:10px 0 5px 0;'>{title}</h3><p style='color:#888;font-size:0.85rem;margin:0;'>{l1}<br>{l2}<br>{l3}<br>{l4}</p></div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Section 9 — Reports and Insights
st.markdown("## Reports and Insights")
if live_mode:
    st.info("Reports and Insights shows historical analysis. Switch to Analysis Mode to access time-filtered reports.")
else:
    st.markdown("*Select a time period to view grid performance analysis and download a report.*")

col_r1,col_r2,col_r3 = st.columns(3)
with col_r1: report_type = st.selectbox("Report Type",["Yearly","Monthly","Weekly"],key="report_type")
with col_r2: report_year = st.selectbox("Year",sorted(df['year'].unique(),reverse=True),key="report_year")
with col_r3:
    if report_type=="Monthly":
        avail_m = sorted(df[df['year']==report_year]['month'].unique())
        sel_mn  = st.selectbox("Month",[MONTH_NAMES[m] for m in avail_m],key="report_month")
        report_month = [k for k,v in MONTH_NAMES.items() if v==sel_mn][0]
    elif report_type=="Weekly":
        avail_w = sorted(df[df['year']==report_year]['week'].unique())
        wlabels = {}
        for w in avail_w:
            wd = df[(df['year']==report_year)&(df['week']==w)]
            wlabels[w] = f"Week {w} ({DAY_NAMES[int(wd['day_of_week'].iloc[0])]} to {DAY_NAMES[int(wd['day_of_week'].iloc[-1])]})" if not wd.empty else f"Week {w}"
        sel_wl      = st.selectbox("Select Week",list(wlabels.values()),key="report_week")
        report_week = [k for k,v in wlabels.items() if v==sel_wl][0]
    else:
        st.markdown("")

if report_type=="Yearly":
    df_report,period_label = df[df['year']==report_year].copy(), f"{report_year}"
elif report_type=="Monthly":
    df_report = df[(df['year']==report_year)&(df['month']==report_month)].copy()
    period_label = f"{MONTH_NAMES[report_month]} {report_year}"
else:
    df_report = df[(df['year']==report_year)&(df['week']==report_week)].copy()
    if not df_report.empty:
        period_label = f"Week {report_week}, {report_year} ({DAY_NAMES[int(df_report['day_of_week'].iloc[0])]} to {DAY_NAMES[int(df_report['day_of_week'].iloc[-1])]})"
    else:
        period_label = f"Week {report_week}, {report_year}"

if df_report.empty:
    st.warning("No data available for the selected period.")
else:
    avg_v  = df_report['vulnerability_score'].mean()
    peak_v = df_report['vulnerability_score'].max()
    low_v  = df_report['vulnerability_score'].min()
    high_r = int((df_report['vulnerability_score']>=70).sum())
    warn_r = int(((df_report['vulnerability_score']>=40)&(df_report['vulnerability_score']<70)).sum())
    stab_r = int((df_report['vulnerability_score']<40).sum())
    st.markdown(f"**Showing: {period_label}** — {len(df_report):,} hours analysed")
    st.markdown("<br>", unsafe_allow_html=True)

    col_m1,col_m2,col_m3,col_m4,col_m5,col_m6 = st.columns(6)
    for col,val,sub,label,color in [
        (col_m1,f"{avg_v:.1f}", "/100","Average Vulnerability","white"),
        (col_m2,f"{peak_v:.1f}","/100","Peak Vulnerability",   "#E74C3C"),
        (col_m3,f"{low_v:.1f}", "/100","Lowest Score",         "#2ECC71"),
        (col_m4,f"{high_r:,}",  "hours","Critical Events",     "#E74C3C"),
        (col_m5,f"{warn_r:,}",  "hours","Warning Events",      "#F39C12"),
        (col_m6,f"{stab_r:,}",  "hours","Stable Hours",        "#2ECC71"),
    ]:
        with col:
            st.markdown(f"<div class='metric-card'><h2 style='color:{color};font-size:1.4rem;margin:0;'>{val}</h2><p style='color:#666;margin:2px 0;font-size:0.75rem;'>{sub}</p><p style='color:#888;margin:0;font-size:0.75rem;'>{label}</p></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_trend,col_dist = st.columns([2,1])

    with col_trend:
        if report_type=="Weekly":
            df_report['day_name'] = df_report['day_of_week'].map(DAY_NAMES)
            dvr = df_report.groupby('day_name')['vulnerability_score'].mean().round(1)
            dvr = dvr.reindex([DAY_NAMES[i] for i in range(7) if DAY_NAMES[i] in dvr.index])
            fig_trend = go.Figure(go.Bar(x=dvr.index, y=dvr.values,
                marker_color=['#E74C3C' if s>=70 else '#F39C12' if s>=40 else '#2ECC71' for s in dvr.values]))
        else:
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(x=df_report['Datetime (UTC)'], y=df_report['vulnerability_score'],
                mode='lines', name='Vulnerability Score', line=dict(color='#4A9EFF',width=1.2),
                fill='tozeroy', fillcolor='rgba(74,158,255,0.1)'))
        fig_trend.add_hline(y=VULNERABILITY_THRESHOLD, line_dash='dash', line_color='#E74C3C',
            annotation_text=f'Threshold ({VULNERABILITY_THRESHOLD:.0f})', annotation_font_color='#E74C3C')
        fig_trend.update_layout(paper_bgcolor='#161B22', plot_bgcolor='#161B22', font=dict(color='white'),
            title=dict(text=f'Vulnerability Trend — {period_label}', font=dict(color='white',size=13)),
            xaxis=dict(gridcolor='#30363D',color='#888'),
            yaxis=dict(gridcolor='#30363D',color='#888',title='Vulnerability Score (0 to 100)',range=[0,100]),
            height=300, margin=dict(t=50,b=30))
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_dist:
        sc = df_report['grid_status'].value_counts()
        fig_dist = go.Figure(go.Pie(labels=sc.index, values=sc.values,
            marker_colors=[{'STABLE':'#2ECC71','WARNING':'#F39C12','CRITICAL':'#E74C3C'}.get(s,'#888') for s in sc.index],
            hole=0.4, textfont=dict(color='white',size=12)))
        fig_dist.update_layout(paper_bgcolor='#161B22', font=dict(color='white'),
            title=dict(text='Grid Status Distribution', font=dict(color='white',size=13)),
            legend=dict(bgcolor='#1A1A2E',bordercolor='#333'), height=300, margin=dict(t=50,b=10))
        st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("### Insight Summary")
    insight = "Grid conditions were consistently critical" if avg_v>=70 else "Grid showed moderate vulnerability with recurring warning signals" if avg_v>=40 else "Grid conditions remained largely stable"
    insight += f" during {period_label}."
    if high_r>0: insight += f" {high_r:,} critical event{'s were' if high_r>1 else ' was'} detected, concentrated in peak vulnerability hours."
    if warn_r>0: insight += f" An additional {warn_r:,} warning period{'s' if warn_r>1 else ''} indicated elevated but manageable grid vulnerability."
    st.info(insight)

    st.markdown("<br>", unsafe_allow_html=True)
    csv_data = df_report[['Datetime (UTC)',CARBON_COL,CFE_COL,'vulnerability_score','vulnerability_event','grid_status','hour','month','month_name']].to_csv(index=False)
    st.download_button(label="Download Report (CSV)", data=csv_data,
        file_name=f"Grid Saver {report_type} {period_label} Report.csv", mime="text/csv")

# Footer
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style='background:#161B22;padding:15px;border-radius:8px;border:1px solid #30363D;text-align:center;margin-top:20px;'>
    <p style='color:#888;margin:0;font-size:0.85rem;'>Grid Saver | Adaptive Grid Intelligence Platform | Justine Adzormado | Built with Colab + GitHub + Streamlit</p>
    <p style='color:#555;margin:5px 0 0 0;font-size:0.75rem;'>Sense: Electricity Maps US-TEX-ERCO 2025 (Academic Access) | Predict: PJM XGBoost 91.6% Recall | 24hr Risk Projection | Act: Pecan Street Austin TX 2018 | 25 real households | Full SPA pipeline validated across Phase 1, 2 and 3</p>
</div>
""", unsafe_allow_html=True)
