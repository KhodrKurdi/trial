# AUBMC Physician Behavioral Performance Dashboard

A data-driven analytics system for monitoring physician behavioral performance at an academic medical center. Built as an MSBA Capstone Project at the Olayan School of Business, American University of Beirut.

---

## Overview

This project processes annual peer behavioral survey data across three survey groups and three fiscal cycles (2023–2025), identifying underperforming physicians through a multi-method outlier detection framework, NLP-based sentiment analysis, and an interactive Streamlit dashboard with an AI querying assistant.

**Key outputs — 2025 cycle:**
- 345 physicians evaluated across 3 survey groups
- 24,243 peer evaluation forms processed
- 16 Priority physicians identified (risk score 3–4)
- 45 Monitor physicians identified (risk score 1–2)
- 284 Clear physicians (risk score 0)

---

## Project Structure

```
├── streamlit_anon.py          # Main dashboard application (anonymized version)
├── GITHUB_URLS                # Data source configuration (inside streamlit_anon.py)
├── README.md                  # This file
│
├── Data Files (hosted on GitHub / configured via GITHUB_URLS)
│   ├── aubmc_23/24/25         # AUBMC General behavioral survey CSVs
│   ├── ed_23/24/25            # Emergency Department survey CSVs
│   ├── patho_23/24/25         # Pathology & Lab survey CSVs
│   ├── Datasource_cycle_2023/2024/2025.csv   # Physician lookup files
│   └── Physicians_Indicators_Anonymized.csv  # Clinical indicators
```

---

## Features

### Multi-Method Outlier Detection
Three independent statistical methods applied to 2025 behavioral scores:

| Method | Threshold | Strength |
|---|---|---|
| IQR Lower Fence | Score < Q1 − 1.5 × IQR | Non-parametric, robust to skewed data |
| Z-Score | Z ≤ −2 (ddof=0) | Captures strongest individual deviations |
| Bottom 10% (P10) | Score ≤ percentile(0.10) | Guarantees consistent monitoring |

### Sentiment Analysis
- VADER with a custom 40+ term medical lexicon
- Negative-component override rule: if `neg ≥ 0.08` → classified as NEGATIVE regardless of compound score
- Flagging rule: any single meaningful comment with compound ≤ −0.05 triggers a flag
- Non-informative comment filter (N/A, D/A, no-contact responses excluded)

### Composite Risk Score
Four independent binary flags summed into a 0–4 score:
- **0 = Clear** — no flags triggered
- **1–2 = Monitor** — partial evidence, enhanced follow-up required
- **3–4 = Priority** — convergent evidence, immediate review within 30 days

### Dashboard Tabs
1. **Summary** — KPIs, risk distribution, department comparison
2. **Risk Register** — physician deep-dive, peer comments, VADER scores
3. **Score Analysis** — IQR scatter, histograms, outlier method comparison
4. **Sentiment** — comment breakdown, yearly sentiment trend
5. **Trends** — score trajectories, percentile ranking, question-level sparklines
6. **Departments** — clinical indicators, visit volumes, complaint counts
7. **Ask MC** — AI natural language assistant powered by Claude Haiku

### Ask MC — AI Assistant
- Built on the Anthropic API (Claude Haiku)
- Full physician dataset serialized as structured text context (~12,000–15,000 tokens)
- Multi-turn conversational memory within session
- Anonymization-aware — never reveals real physician names or institutional identifiers

---

## Setup & Deployment

### Requirements
```
streamlit
pandas
numpy
matplotlib
vaderSentiment
requests
```

Install with:
```bash
pip install streamlit pandas numpy matplotlib vaderSentiment requests
```

### Configuration
All data file paths are configured in the `GITHUB_URLS` dictionary at the top of `streamlit_anon.py`:

```python
GITHUB_URLS = {
    "aubmc_23":    "https://raw.githubusercontent.com/YOUR_REPO/...aubmc_2023.csv",
    "aubmc_24":    "https://raw.githubusercontent.com/YOUR_REPO/...aubmc_2024.csv",
    "aubmc_25":    "https://raw.githubusercontent.com/YOUR_REPO/...aubmc_2025.csv",
    "ed_23":       "https://raw.githubusercontent.com/YOUR_REPO/...ed_2023.csv",
    "ed_24":       "https://raw.githubusercontent.com/YOUR_REPO/...ed_2024.csv",
    "ed_25":       "https://raw.githubusercontent.com/YOUR_REPO/...ed_2025.csv",
    "patho_23":    "https://raw.githubusercontent.com/YOUR_REPO/...patho_2023.csv",
    "patho_24":    "https://raw.githubusercontent.com/YOUR_REPO/...patho_2024.csv",
    "patho_25":    "https://raw.githubusercontent.com/YOUR_REPO/...patho_2025.csv",
    "lookup_2023": "https://raw.githubusercontent.com/YOUR_REPO/...Datasource_2023.csv",
    "lookup_2024": "https://raw.githubusercontent.com/YOUR_REPO/...Datasource_2024.csv",
    "lookup_2025": "https://raw.githubusercontent.com/YOUR_REPO/...Datasource_2025.csv",
    "indicators":  "https://raw.githubusercontent.com/YOUR_REPO/...Physicians_Indicators.csv",
}
```

### Anthropic API Key
Add your key to Streamlit secrets:
```toml
# .streamlit/secrets.toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

### Run Locally
```bash
streamlit run streamlit_anon.py
```

### Deploy to Streamlit Cloud
1. Push `streamlit_anon.py` to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect the repository and set the main file as `streamlit_anon.py`
4. Add `ANTHROPIC_API_KEY` in the Streamlit Cloud secrets manager

---

## Adding the 2026 Cycle

When the 2026 survey cycle is complete, update the dashboard in three steps:

**Step 1 — Export and upload the new CSV files**
Export the three BLUE Explorance survey CSVs for 2026 (AUBMC General, ED, Pathology) and the new Datasource lookup file. Upload them to your GitHub repository alongside the existing files.

**Step 2 — Update GITHUB_URLS**
In `streamlit_anon.py`, add the new file paths and shift the year window:

```python
# Add 2026 files
"aubmc_26":    "https://raw.githubusercontent.com/YOUR_REPO/...aubmc_2026.csv",
"ed_26":       "https://raw.githubusercontent.com/YOUR_REPO/...ed_2026.csv",
"patho_26":    "https://raw.githubusercontent.com/YOUR_REPO/...patho_2026.csv",
"lookup_2026": "https://raw.githubusercontent.com/YOUR_REPO/...Datasource_2026.csv",

# Update the load_dept calls to use the new year
aubmc_raw, aubmc_phys, aubmc_sent = load_dept(["aubmc_24","aubmc_25","aubmc_26"], "PROJECT-1")
ed_raw,    ed_phys,    ed_sent    = load_dept(["ed_24","ed_25","ed_26"],           "PROJECT-2")
patho_raw, patho_phys, patho_sent = load_dept(["patho_24","patho_25","patho_26"], "PROJECT-3")
```

**Step 3 — Update the indicators file**
Replace or append the `Physicians_Indicators_Anonymized.csv` with the updated file containing 2026 clinic visit, wait time, and complaint data.

That is it. No code changes to the pipeline, scoring, or dashboard are needed — the system handles the new year automatically.

---

## Anonymization

The dashboard runs in fully anonymized mode:
- Physician IDs displayed as `PHYS_xxxx` (anonymized in source files)
- Survey groups displayed as `PROJECT-1`, `PROJECT-2`, `PROJECT-3`
- Departments and divisions displayed using anonymized codes
- Real names never loaded or stored

---

## Future Work

### 1. Predictive Analytics — Forecasting Outlier Risk
The three-year dataset (2023–2025) provides a longitudinal foundation for predictive modeling. A logistic regression or gradient boosting model could be trained to predict which Monitor physicians are most likely to escalate to Priority in the following cycle, based on their score trajectory, rate of decline, sentiment trend, and department-level indicators. This would shift the system from reactive identification to proactive risk forecasting — flagging physicians before they deteriorate rather than after.

Potential features for the model:
- Year-over-year score change (slope)
- Number of consecutive declining cycles
- Percentile rank trajectory
- Sentiment compound score trend
- Department-level complaint rate

### 2. Transformer-Based Sentiment Analysis
When a sufficient labeled comment dataset becomes available (~500–1,000 labeled examples), replace VADER with a fine-tuned transformer model such as ClinicalBERT or a multilingual Arabic-English model. This would handle sarcasm, complex negation, and mixed-sentiment comments that VADER systematically misses. VADER would be retained as a baseline for comparison and validation.

### 3. Real-Time Integration with BLUE Explorance
Replace the annual CSV export workflow with a direct API integration with the BLUE Explorance platform. This would enable continuous data refresh and mid-cycle monitoring rather than annual snapshots — surfacing behavioral deterioration as it happens rather than at year-end.

### 4. Role-Based Access Control
Implement department-level authentication so department heads see only their own physicians, division chiefs see only their division, and the CMO sees the full population. Currently the dashboard is accessible to anyone with the deployment link.

### 5. Action Tracking Module
Add a layer on top of the Risk Register where quality officers can log actions taken for each Priority or Monitor physician — coaching sessions, FPPE initiations, feedback delivered. This closes the loop between identification and intervention and creates an auditable trail for JCI inspections.

---

## References

- Hutto, C. J., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. *ICWSM*. https://doi.org/10.1609/icwsm.v8i1.14550
- Barik, K., & Misra, S. (2024). Analysis of customer reviews with an improved VADER lexicon classifier. *Journal of Big Data*, 11, 10. https://doi.org/10.1186/s40537-023-00861-x
- Mramba, L. K., et al. (2024). Detecting potential outliers in longitudinal data with time-dependent covariates. *European Journal of Clinical Nutrition*, 78(4), 344–350. https://doi.org/10.1038/s41430-023-01393-6
- Pavlou, M., et al. (2023). Outlier identification and monitoring of institutional or clinician performance. *BMC Health Services Research*, 23, 33. https://doi.org/10.1186/s12913-022-08995-z
- Paddock, S. M. (2014). Statistical benchmarks for health care provider performance assessment. *Health Services Research*, 49(3), 1056–1073. https://doi.org/10.1111/1475-6773.12149
- Kidd, V. D., & Liu, J. H. (2022). Development of a visual dashboard to assess provider productivity. *BMC Health Services Research*, 22, 882. https://doi.org/10.1186/s12913-022-08216-7
- Patel, S., et al. (2022). Using participatory design to develop a provider-level performance dashboard. *The Joint Commission Journal on Quality and Patient Safety*, 48(3), 165–172. https://doi.org/10.1016/j.jcjq.2021.10.003

---

## Author

**Khodr Kurdi**
MSBA — Olayan School of Business, American University of Beirut
Capstone Project — Spring 2026
