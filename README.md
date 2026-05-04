# ⚡ Grid Saver: Adaptive Grid Intelligence Platform

Grid Saver is a decision-support system that detects grid stress, predicts vulnerability, and executes coordinated demand-side response using a dual-confirmation control logic.
**Live Demo:** [gridsavermvp.streamlit.app](https://gridsavermvp.streamlit.app)
*(App may take a few seconds to load if inactive)*



## Problem

Ghana's power grid has experienced recurring instability driven by peak demand stress and fuel dependency. These challenges are not unique to Ghana; they reflect a global grid management problem.

Modern grids face increasing pressure from:

- High carbon intensity during peak demand periods
- Variability in renewable energy supply
- Limited real-time coordination of residential loads

Most systems either react too late, or rely on single-signal triggers that produce high false positives.



## Why Now

- Grid instability events are increasing in both emerging and advanced power systems
- Renewable variability is making demand prediction harder
- Demand-side coordination remains underutilised at scale

Grid Saver addresses all three simultaneously.



## Solution

Grid Saver introduces a dual-confirmation intelligence system (SPA) that only acts when two independent signals agree:

- Real-time grid stress (Sense Layer)
- Forward-looking risk pattern (Predict Layer)

This prevents false activations while ensuring action is taken at the right moment, at the correct operational window.



## System Architecture

Grid Saver operates through three independent layers that meet only at the decision point. No layer feeds its output into another. They confirm independently before any action is triggered.

### 👁️ Sense Layer (Detection)

Detects current grid vulnerability from real-time signals.

- Input: Carbon intensity and carbon-free energy percentage (CFE%)
- Output: Dynamic vulnerability score (0 to 100), grid status (Stable, Warning, Critical)
- Source: Electricity Maps US-TEX-ERCO 2025 (8,760 hourly records)
- Formula: 70% carbon intensity signal + 30% CFE signal, normalised to observed range
- Threshold: Top 15% most vulnerable hours flagged as vulnerability events

### 🧠 Predict Layer (Forecasting)

Generates an independent temporal risk projection using historical demand patterns.

- Input: Time-based features only (hour, day of week, month, seasonality, lag patterns)
- Output: 24hr Risk Projection (probability 0 to 1), triggered flag at threshold 0.40
- Training data: PJM Interconnection demand data (1998 to 2002), used in the MVP notebook
- Deployment: Trained model (gridsaver_model.pkl) loaded in the app and applied to ERCOT timestamps
- Transparency: For MVP deployment, a temporally constructed synthetic demand proxy is used to activate the trained model, as real-time demand data is not available. In production this is replaced with a live SCADA demand feed.
- Performance: 91.3% Recall, 0.978 ROC-AUC (evaluated on PJM validation dataset)
- Independence: Receives no input from the Sense Layer. Uses ERCOT timestamps only to derive temporal patterns.

### ⚡ Act Layer (Coordination)

Coordinates residential HVAC demand reduction when both Sense and Predict independently confirm risk.

- Input: SPA dual-confirmation signal (Sense AND Predict must both trigger)
- Output: Coordinated HVAC load reduction across residential homes
- Observed in dataset: 2.2% peak demand reduction across 25 real Austin TX households
- Scales to approximately 92 MW removed at 1 million homes coordinated
- Source: Pecan Street Inc. Austin TX 2018 (868,096 records, 25 real households)



## SPA Logic (Core Innovation)

Grid Saver does NOT act on a single trigger.

**SPA Trigger = Sense AND Predict**

- Sense detects current grid stress
- Predict forecasts the upcoming risk window

Only when both independently agree is intervention executed. Otherwise monitoring continues. This reduces false activations and improves system stability.



## Why Independence Matters

The Sense Layer reads real-time grid conditions. The Predict Layer identifies historical risk patterns based on time of day, season, and demand cycle.

Grid Saver only acts when both agree. If the Sense Layer detects a carbon spike but the temporal pattern model does not confirm risk, no action is triggered. If the Predict Layer flags a high-risk hour but the Sense Layer shows stable conditions, no action is triggered. Both must confirm.

This dual-confirmation logic reduces false positives and ensures interventions occur only when truly needed.



## Prototype vs MVP

### Prototype (Multi-Source Validation)

The prototype demonstrates that the SPA logic works across multiple independent real-world datasets:

- Sense: Electricity Maps US-TEX-ERCO 2025 (ERCOT, Texas)
- Predict: PJM Interconnection 1998 to 2002 (US Northeast and Midwest)
- Act: Pecan Street Inc. Austin TX 2018

Each layer was validated independently to prove the concept works across different grids, regions and time periods. This is the laboratory proof.

### MVP (Unified Production Architecture)

The MVP demonstrates how Grid Saver functions as a single cohesive system using one regional data stream, the way it would operate in a real deployment environment such as GRIDCo in Ghana or ERCOT in Texas.

In production, grid intelligence systems typically receive one high-speed data stream (SCADA). Grid Saver maintains independence by processing different dimensions of that stream. Sense reads real-time grid conditions. Predict reads historical temporal patterns from the same timestamps. Act responds only when both confirm.

The logic is identical. The data flow is production-grade.



## Simulation Model

**Three-Layer Baseline**

- Theoretical baseline: 70,320 MW (explanation and HVAC decomposition only, never plotted)
- Model peak envelope: 66,804 MW (95% of theoretical, documented reference, not enforced)
- Observed peak: runtime value derived from dataset, never overridden

**HVAC Reduction Model**

- HVAC share: 25% of grid load
- Default reduction rate: 4%
- System impact per SPA event: 703.2 MW

**Rebound Effect**

- 60% of reduced load returns in the next hour (thermal snapback)
- 85% compliance rate (behavioral assumption from literature, not validated against Pecan Street dataset directly)



## Validated Impact (Annual Simulation)

| Metric | Value |
|---|---|
| Sense triggers | 1,316 hours |
| Predict triggers | 1,659 hours |
| SPA events (dual-confirmed) | 154 hours |
| Gross annual reduction | 108,293 MWh |
| Thermal rebound | 64,976 MWh |
| **Net annual savings** | **43,317 MWh (dual-confirmed intervention only)** |



## Validation

Grid Saver was built and validated using real-world datasets:

- Electricity Maps US-TEX-ERCO 2025 (grid carbon and generation signals, Sense Layer and app deployment)
- PJM Interconnection 1998 to 2002 (demand forecasting model training, MVP notebook, produces gridsaver_model.pkl)
- Pecan Street Inc. Austin TX 2018 (residential HVAC load coordination, Act Layer validation)

The XGBoost model is trained on PJM demand data in the MVP notebook. The trained model is then deployed in the app using ERCOT timestamps as input. PJM and ERCOT share the same demand seasonality structure making this transfer valid for demonstration and early-stage validation.

Note: Ghana grid data is not publicly available for research use. Validation was conducted on the most transparent publicly available grid datasets in the world. The architecture and behavioural model transfer directly to any grid environment including GRIDCo and ECG in Ghana.



## Tech Stack

- **Frontend:** Streamlit
- **Data Processing:** Pandas, NumPy
- **ML Model:** XGBoost
- **Visualization:** Plotly
- **Notebook:** Google Colab
- **Deployment:** Streamlit Cloud + GitHub



## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The MVP notebook (gridsaver_mvp.ipynb) must be run first in Google Colab to generate gridsaver_model.pkl and data_sample.csv before deploying the app.



## MVP Features

- Real-time grid status (Stable, Warning, Critical)
- 24hr Risk Projection (XGBoost pattern-recognition of historical grid stress cycles, 24hr lead time)
- Explainable Risk Drivers (translates system signals into operator-understandable causes)
- Dual-confirmation SPA decision logic (action only when both layers agree)
- Grid Saver Load Reduction Simulation (before and after intervention visualisation)
- Impact at Scale (projects Grid Saver effect from neighbourhood to national scale)
- Reports and Insights (time-filtered analysis with trend chart, insight summary and CSV export)



## Limitations

- Uses historical datasets, not real-time SCADA integration
- Predict layer uses a temporally constructed synthetic demand proxy, not a live demand feed
- HVAC compliance (85%) is a literature-based behavioral assumption, not validated against Pecan Street dataset directly
- Scaling beyond 3 to 5% HVAC reduction is extrapolated and not independently validated
- Grid Saver does not detect or repair physical faults such as fires or equipment failure. It identifies vulnerability conditions that make such failures more likely, enabling earlier coordinated response.



## Future Development

- Live grid API integration (SCADA, Electricity Maps live feed)
- Smart device control via thermostat APIs (Ecobee, Google Nest)
- Dynamic pricing signal integration
- Utility and grid operator partnerships
- Model retraining pipeline with drift detection and rolling recall audits
- SaaS deployment for energy operators
- Expansion to West African grids (GRIDCo, ECG Ghana)



## Repository Structure

```
grid_saver_mvp/
├── app.py                    Streamlit MVP application
├── gridsaver_model.pkl       Trained XGBoost model
├── data_sample.csv           Processed ERCOT dataset (derived output)
├── gridsaver_mvp.ipynb       MVP pipeline notebook (Sense + Predict + Act)
├── requirements.txt          Python dependencies
└── README.md                 This file
```

Prototype notebooks are included for reference to show the multi-source validation that underpins the MVP architecture.



## Author

**Justine Adzormado**
Occupational HSE Professional | Data Scientist | Founder, Grid Saver
Women Techsters Fellow (Tech4Dev, Class of 2026)

Built from real-world grid challenges observed in Ghana, designed for global deployment.

> ⚠️ Production readiness requires live SCADA integration and regulatory approval.
Current version operates on historical data with validated models.
