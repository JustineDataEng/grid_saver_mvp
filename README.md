# Grid Saver — Adaptive Grid Intelligence Platform

**Turning passive electricity consumption into active grid stability.**

Grid Saver is a three-layer AI system that detects grid vulnerability, forecasts stress events up to 24 hours ahead, and coordinates demand reduction to reduce the risk and impact of outages.

---

## 🚨 The Problem

Electricity grids are increasingly vulnerable to demand spikes, generation variability, and infrastructure stress.  
Most systems react only after instability occurs — when it is already too late.

---

## 💡 The Solution

Grid Saver provides early warning and coordinated response through a unified intelligence pipeline:

- Detects real-time grid vulnerability signals  
- Predicts high-risk stress periods up to 24 hours ahead  
- Coordinates demand reduction across residential loads  

---

## 🧠 Architecture

### 1. Sense Layer
- Real-time vulnerability detection  
- Uses carbon intensity and generation mix signals  
- Outputs a dynamic grid vulnerability score  

### 2. Predict Layer
- XGBoost forecasting model  
- 91.6% Recall | 0.977 ROC-AUC  
- Predicts grid stress events up to 24 hours ahead  

### 3. Act Layer
- Simulates coordinated HVAC demand reduction  
- Validated 2.2% peak load reduction per event  
- Projects ~92 MW reduction at 1 million homes  

---

## 📊 Validation

Grid Saver was built and validated using real-world datasets:

- ERCOT (Electricity Maps) — grid signals  
- PJM Interconnection — load forecasting  
- Pecan Street — residential energy consumption  

---

## ⚙️ Tech Stack

- Python  
- XGBoost  
- Streamlit  
- Google Colab  

---

## 🖥️ MVP Features

- Real-time grid status (Stable / Warning / Critical)  
- 24-hour stress risk prediction  
- Risk Drivers — plain language explanation of what is causing grid stress  
- Simulated demand reduction and impact at scale  
- Impact at Scale — projects Grid Saver's effect from 25 homes to 1 million  
- Reports and Insights — time-filtered analysis with trend chart, insight summary and CSV export  

---

## 📊 Reports and Insights

The Reports and Insights module allows users to:

- Filter grid performance data by weekly, monthly or yearly period  
- View key metrics including average vulnerability score, peak stress, critical events and stable hours  
- Explore a vulnerability trend chart and grid status distribution  
- Read an AI-generated insight summary describing grid conditions for the selected period  
- Download a filtered report as a CSV file  

---

## ⚠️ Scope and Limitations

Grid Saver does not detect or repair physical faults (e.g., fires, equipment failure).  
It identifies **risk conditions** that make such failures more likely, enabling earlier intervention.

---

## 🚀 Future Development

- Real-time grid API integration  
- Smart device control (thermostats, IoT)  
- Utility partnerships  
- SaaS deployment for energy operators  

---

## 👤 Author

**Justine Adzormado**
