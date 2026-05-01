# Grid Saver — Adaptive Grid Intelligence Platform

**Turning passive electricity consumption into active grid stability.**

**Live Demo:** [gridsavermvp.streamlit.app](https://gridsavermvp.streamlit.app)
*(App may take a few seconds to load if inactive)*

Grid Saver is a three-layer AI system that detects grid vulnerability, generates independent temporal risk projections, and coordinates residential demand reduction to prevent outages before they occur.



## The Problem

Electricity grids are increasingly vulnerable to demand spikes, generation variability, and infrastructure stress. Most systems react only after instability occurs — when it is already too late. Few existing systems predict residential vulnerability windows at scale without requiring dedicated hardware installation.



## The Solution

Grid Saver provides early warning and coordinated response through a unified intelligence pipeline:

- Detects real-time grid vulnerability signals from carbon intensity and generation mix
- Generates independent temporal risk projections up to 24 hours ahead using learned demand patterns
- Coordinates demand reduction across residential HVAC loads when both layers independently confirm risk



## Architecture — The SPA Pipeline

Grid Saver operates through three independent layers that meet only at the decision point. No layer feeds its output into another. They confirm independently before any action is triggered.

### Sense Layer — The Eyes

Detects current grid vulnerability from real-time signals.

- Input: Carbon intensity and carbon-free energy percentage (CFE%)
- Output: Dynamic vulnerability score (0 to 100), grid status (Stable, Warning, Critical)
- Source: Electricity Maps US-TEX-ERCO 2025 — 8,760 hourly records
- Formula: 70% carbon intensity signal + 30% CFE signal, normalised to observed range
- Threshold: Top 15% most vulnerable hours flagged as vulnerability events

### Predict Layer — The Brain

Generates an independent temporal risk projection using historical demand patterns.

- Input: Time-based features only — hour, day of week, month, seasonality, lag patterns
- Output: 24hr Risk Projection (Temporal Pattern) — risk probability (0 to 1), triggered flag (threshold: 0.40)
- Training data: PJM Interconnection demand data (1998 to 2002) — used in the MVP notebook to train the model
- Deployment: Trained model (gridsaver_model.pkl) is loaded in the app and applied to ERCOT timestamps
- Transparency: For MVP deployment, a temporally constructed synthetic demand baseline is used to activate the trained model, as real-time demand data is not available. In production this is replaced with a live SCADA demand feed.
- Performance: 91.6% Recall, 0.977 ROC-AUC (evaluated on PJM validation dataset)
- Projection type: Pattern-recognition of historical grid stress cycles — identifies vulnerability windows with 24hr lead time
- Independence: Receives no input from the Sense Layer. Uses ERCOT timestamps only to derive temporal patterns.

### Act Layer — The Response

Coordinates residential HVAC demand reduction when both Sense and Predict independently confirm risk.

- Input: SPA dual-confirmation signal (Sense AND Predict must both trigger)
- Output: Coordinated HVAC load reduction across residential homes
- Validated: 2.2% peak demand reduction across 25 real Austin TX households
- Projection: Approximately 92 MW removed at 1 million homes coordinated
- Source: Pecan Street Inc. Austin TX 2018 — 868,096 records, 25 real households



## Why Independence Matters

The Sense Layer reads what the grid is doing right now. The Predict Layer reads what has historically happened at this hour, this season, this day type. These are genuinely different signals even from the same grid.

Grid Saver only acts when both independently agree. If the Sense Layer detects a carbon spike but the temporal pattern model does not confirm risk, no action is triggered. If the Predict Layer flags a high-risk hour but the Sense Layer shows stable conditions, no action is triggered. Both must confirm.

This is the core innovation. It is not pattern matching. It is dual-confirmation intelligence.



## Prototype vs MVP

### Prototype (Multi-Source Validation)

The prototype demonstrates that the SPA logic works across multiple independent real-world datasets:

- Sense: Electricity Maps US-TEX-ERCO 2025 (ERCOT, Texas)
- Predict: PJM Interconnection 1998 to 2002 (US Northeast and Midwest)
- Act: Pecan Street Inc. Austin TX 2018

Each layer was validated independently to prove the concept works across different grids, regions and time periods. This is the laboratory proof.

### MVP (Unified Production Architecture)

The MVP demonstrates how Grid Saver functions as a single cohesive system using one regional data stream — the way it would operate in a real deployment environment such as GRIDCo in Ghana or ERCOT in Texas.

In production, grid intelligence systems typically receive one high-speed data stream (SCADA). Grid Saver maintains independence by processing different dimensions of that stream. Sense reads real-time grid conditions. Predict reads historical temporal patterns from the same timestamps. Act responds only when both confirm.

The logic is identical. The data flow is production-grade.



## Validation

Grid Saver was built and validated using real-world datasets:

- Electricity Maps US-TEX-ERCO 2025 — grid carbon and generation signals (Sense Layer and app deployment)
- PJM Interconnection 1998 to 2002 — demand forecasting model training (MVP notebook, produces gridsaver_model.pkl)
- Pecan Street Inc. Austin TX 2018 — residential HVAC load coordination (Act Layer validation)

The XGBoost model is trained on PJM demand data in the MVP notebook. The trained model is then deployed in the app using ERCOT timestamps as input. PJM and ERCOT share the same demand seasonality structure making this transfer valid for demonstration and early-stage validation.

Note: Ghana grid data is not publicly available for research use. Validation was conducted on the most transparent publicly available grid datasets in the world. The architecture and behavioural model transfer directly to any grid environment including GRIDCo and ECG in Ghana.



## Tech Stack

- Python
- XGBoost
- Streamlit
- Google Colab
- GitHub (Private)



## MVP Features

- Real-time grid status (Stable, Warning, Critical)
- 24hr Risk Projection (Temporal Pattern) — XGBoost pattern-recognition of historical grid stress cycles identifying vulnerability windows with 24hr lead time
- Risk Drivers — explains what is causing the current vulnerability score using relative signal positioning
- Dual-confirmation SPA decision logic — action only when both layers agree
- Grid Saver Load Reduction Simulation — before and after intervention visualisation
- Impact at Scale — projects Grid Saver effect from neighbourhood to national scale
- Reports and Insights — time-filtered analysis with trend chart, insight summary and CSV export



## Scope and Limitations

Grid Saver does not detect or repair physical faults such as fires or equipment failure. It identifies vulnerability conditions that make such failures more likely, enabling earlier coordinated response.

The Predict Layer generates a 24hr Risk Projection using temporal pattern-recognition, not a rolling live forecast. It identifies vulnerability windows based on learned historical grid stress cycles. PJM and ERCOT share the same demand seasonality structure — summer peaks, winter peaks, weekday and weekend cycles — making this transfer valid for demonstration and early-stage validation. Production deployment would retrain on local grid demand data.

Scaling beyond the validated 3 to 5% HVAC reduction range is extrapolated and not independently validated at this stage.



## Future Development

- Real-time grid API integration (SCADA, Electricity Maps live feed)
- Smart device control via thermostat APIs (Ecobee, Google Nest)
- Utility and grid operator partnerships
- Model retraining pipeline with drift detection and rolling recall audits
- SaaS deployment for energy operators
- Expansion to West African grids (GRIDCo, ECG Ghana)



## Repository Structure

```
grid_saver_mvp/
├── app.py                    Streamlit MVP application
├── gridsaver_model.pkl       Trained XGBoost model (Phase 2)
├── data_sample.csv           Processed ERCOT dataset (derived output)
├── gridsaver_mvp.ipynb       MVP pipeline notebook (Sense + Predict + Act)
├── requirements.txt          Python dependencies
└── README.md                 This file
```

Prototype notebooks are included for reference to show the multi-source validation that underpins the MVP architecture.

The MVP notebook (gridsaver_mvp.ipynb) trains the XGBoost model on PJM demand data and saves it as gridsaver_model.pkl. The app loads this trained model and applies it to ERCOT timestamps. The notebook must be run in Google Colab to generate gridsaver_model.pkl and data_sample.csv before deploying the app.



## Author

**Justine Adzormado**
Occupational HSE Professional | Data Scientist | Founder, Grid Saver
Women Techsters Fellow (Tech4Dev, Class of 2026)
