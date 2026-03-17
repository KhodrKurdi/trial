import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import io
import warnings
import json
warnings.filterwarnings("ignore")


# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AUBMC Physician Performance Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fb; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        border-left: 4px solid #2563eb;
        margin-bottom: 8px;
    }
    .metric-card.warning { border-left-color: #f59e0b; }
    .metric-card.danger  { border-left-color: #ef4444; }
    .metric-card.success { border-left-color: #10b981; }
    .metric-card.neutral { border-left-color: #6366f1; }
    .metric-label { font-size: 12px; color: #6b7280; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { font-size: 32px; font-weight: 700; color: #111827; line-height: 1.2; }
    .metric-sub   { font-size: 12px; color: #9ca3af; margin-top: 2px; }
    .section-header {
        font-size: 18px; font-weight: 700; color: #1f2937;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 8px; margin-bottom: 16px;
    }
    .pill-red    { background:#fef2f2; color:#dc2626; padding:3px 10px; border-radius:999px; font-size:12px; font-weight:600; }
    .pill-yellow { background:#fffbeb; color:#d97706; padding:3px 10px; border-radius:999px; font-size:12px; font-weight:600; }
    .pill-green  { background:#f0fdf4; color:#16a34a; padding:3px 10px; border-radius:999px; font-size:12px; font-weight:600; }
    .pill-grey   { background:#f3f4f6; color:#6b7280; padding:3px 10px; border-radius:999px; font-size:12px; font-weight:600; }
    div[data-testid="stSidebarNav"] { display: none; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: white; border-radius: 8px 8px 0 0;
        font-weight: 600; color: #374151;
    }
    .stTabs [aria-selected="true"] { background: #2563eb !important; color: white !important; }
    .comment-card {
        background: white; border-radius: 10px; padding: 14px 18px;
        margin-bottom: 10px; border-left: 3px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .comment-card.neg { border-left-color: #ef4444; }
    .comment-card.pos { border-left-color: #10b981; }
    .comment-card.neu { border-left-color: #d1d5db; }
    /* Hide sidebar and its toggle button entirely */
    [data-testid="collapsedControl"] { display: none !important; }
    section[data-testid="stSidebar"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ─── HELPER FUNCTIONS ────────────────────────────────────────────────────────
RATING_SCALE = {
    "Always": 4, "Most of the time": 3,
    "Sometimes": 2, "Hardly ever": 1, "Never": 0, "nan": np.nan
}

def clean_question_col(col):
    if not isinstance(col, str): return col
    c = col.strip()
    c = (c.replace("â€™","'").replace("â€¦","...").replace("\r"," ").replace("\n"," "))
    c = re.sub(r"\s+", " ", c).strip()
    if c == "Subject ID":            return "physician_id"
    if c == "Raters Group":          return "raters_group"
    if c == "Completed by":          return "completed_by"
    if c == "Rater Name":            return "rater_name"
    if c.startswith("Fillout Date"): return "fillout_date"
    if c.startswith("Q2_Comments"):  return "comments"
    if not c.startswith("Q1_"):      return c
    m = re.search(r"_(.*?)_First Scale", c)
    core = m.group(1) if m else c
    core = core.lower()
    core = re.sub(r"[^a-z0-9]+", "_", core).strip("_")
    return f"q_{core}"

def clean_headers(df):
    df = df.copy()
    df.columns = [clean_question_col(c) for c in df.columns]
    return df

def map_ratings(df):
    df = df.copy()
    q_cols = [c for c in df.columns if c.startswith("q_")]
    for c in q_cols:
        s = df[c].astype(str).str.strip()
        mapped  = s.map(RATING_SCALE)
        numeric = pd.to_numeric(s, errors="coerce")
        df[c] = numeric.where(numeric.notna(), mapped)
    return df

def compute_score(df):
    q_cols = [c for c in df.columns if c.startswith("q_")]
    df["overall_score"] = df[q_cols].mean(axis=1, skipna=True)
    return df

def add_year(df):
    if "fillout_date" in df.columns:
        df["year"] = pd.to_datetime(df["fillout_date"], errors="coerce").dt.year
    elif "year" not in df.columns:
        # Fallback: try to detect year column by name variants
        year_candidates = [c for c in df.columns if "year" in c.lower() or "date" in c.lower()]
        if year_candidates:
            df["year"] = pd.to_datetime(df[year_candidates[0]], errors="coerce").dt.year
        else:
            df["year"] = np.nan  # year unknown — will be excluded from 2025 filter
    return df

def aggregate_physician(df):
    # Exclude self-evaluations — not peer assessments
    if "raters_group" in df.columns:
        df = df[df["raters_group"] != "Faculty Self-Evaluation"].copy()

    q_cols = [c for c in df.columns if c.startswith("q_")]

    # Flat mean: stack all question responses across all forms per physician,
    # drop NaN, then take a single mean — avoids mean-of-means distortion
    # where forms with fewer answered questions get equal weight to full forms
    def flat_mean(grp):
        vals = grp[q_cols].values.flatten()
        vals = vals[~pd.isnull(vals)]
        return float(vals.mean()) if len(vals) > 0 else np.nan

    grouped = df.groupby("physician_id")
    scores  = grouped.apply(flat_mean).reset_index()
    scores.columns = ["physician_id", "avg_behavior_score"]
    n_forms = grouped.size().reset_index(name="n_forms")
    return scores.merge(n_forms, on="physician_id")

def add_outlier_flags(phys_df):
    df = phys_df.copy()
    scores   = df["avg_behavior_score"]
    pop_mean = scores.mean()
    pop_std  = scores.std(ddof=0) if len(scores) > 1 else 0

    df["z_score"]         = (scores - pop_mean) / pop_std if pop_std > 0 else 0.0
    df["low_z_outlier"]   = df["z_score"] <= -2

    Q1, Q3                = scores.quantile(0.25), scores.quantile(0.75)
    IQR                   = Q3 - Q1
    df["low_iqr_outlier"] = scores < (Q1 - 1.5 * IQR) if IQR > 0 else False
    df["low_bottom10"]    = scores <= scores.quantile(0.10)
    return df, pop_mean, pop_std

vader = SentimentIntensityAnalyzer()

def score_vader(text, threshold=-0.05):
    try:
        s = vader.polarity_scores(str(text))
        c = s["compound"]
        # Notebook uses standard VADER thresholds fixed at ±0.05
        label = "POSITIVE" if c >= 0.05 else ("NEGATIVE" if c <= -0.05 else "NEUTRAL")
        return {"compound": c, "sentiment": label}
    except:
        return {"compound": 0.0, "sentiment": "NEUTRAL"}

def run_sentiment(df, threshold=-0.05):
    # Match notebook: score all non-empty, non-self-eval comments — no skip list
    df_s = df[
        (df.get("raters_group", pd.Series(dtype=str)) != "Faculty Self-Evaluation") &
        (df["comments"].notna()) &
        (df["comments"].astype(str).str.strip() != "")
    ].copy()
    df_s["comments"] = df_s["comments"].astype(str).str.strip()
    results = df_s["comments"].apply(lambda t: score_vader(t, threshold))
    df_s = pd.concat([df_s, pd.DataFrame(results.tolist(), index=df_s.index)], axis=1)
    return df_s

def sentiment_summary(df_sent, min_comments=5, threshold=-0.05):
    # Check if any comment has compound < 0 (catches comments VADER labels NEUTRAL but are still negative-leaning)
    has_any_negative = (
        df_sent[df_sent["compound"] < 0]
        .groupby("physician_id")
        .size()
        .reset_index(name="n_neg_compound")
    )
    has_any_negative["has_negative"] = True

    s = (
        df_sent.assign(is_neg=(df_sent["sentiment"]=="NEGATIVE"))
        .groupby("physician_id", as_index=False)
        .agg(total_comments=("is_neg","count"),
             negative_comments=("is_neg","sum"),
             avg_compound=("compound","mean"))
    )
    s["negative_ratio"] = s["negative_comments"] / s["total_comments"]

    # Merge in the any-negative flag
    s = s.merge(has_any_negative[["physician_id","has_negative"]], on="physician_id", how="left")
    s["has_negative"] = s["has_negative"].fillna(False)

    # Flag if physician has ANY comment with compound < 0
    s["negative_outlier"] = s["has_negative"]
    s = s.drop(columns=["has_negative"])
    return s

def merge_sentiment(phys_df, sent_s):
    out = phys_df.merge(
        sent_s[["physician_id","total_comments","negative_ratio","avg_compound","negative_outlier"]],
        on="physician_id", how="left"
    )
    out["negative_outlier"] = out["negative_outlier"].fillna(False)
    return out

def add_risk(phys_df):
    df = phys_df.copy()
    # 4 independent flags — each worth 1 point
    f1 = df["low_iqr_outlier"].fillna(False).astype(bool).astype(int)  if "low_iqr_outlier"  in df.columns else pd.Series(0, index=df.index)
    f2 = df["low_z_outlier"].fillna(False).astype(bool).astype(int)    if "low_z_outlier"    in df.columns else pd.Series(0, index=df.index)
    f3 = df["low_bottom10"].fillna(False).astype(bool).astype(int)     if "low_bottom10"     in df.columns else pd.Series(0, index=df.index)
    f4 = df["negative_outlier"].fillna(False).astype(bool).astype(int) if "negative_outlier" in df.columns else pd.Series(0, index=df.index)
    df["risk_score"] = f1 + f2 + f3 + f4
    # 3-4 = Priority, 1-2 = Monitor, 0 = Clear
    df["final_flag"] = df["risk_score"] >= 3
    return df

def risk_pill(score):
    if score >= 3:   return '<span class="pill-red">⚠ Priority</span>'
    if score >= 1:   return '<span class="pill-yellow">👁 Monitor</span>'
    return '<span class="pill-green">✓ Clear</span>'

def process_dept(df_raw, dept_name, threshold=-0.05, min_f=1):
    df = clean_headers(df_raw)
    df = map_ratings(df)
    df = compute_score(df)
    df = add_year(df)
    # Aggregate and flag on 2025 only — consistent with notebook methodology
    df_2025 = df[df["year"] == 2025] if ("year" in df.columns and not df[df["year"] == 2025].empty) else df
    phys = aggregate_physician(df_2025)
    # Apply min_forms BEFORE outlier detection so thresholds are computed
    # on the same population that will appear in results (matches notebook)
    phys = phys[phys["n_forms"] >= min_f].copy().reset_index(drop=True)
    phys, mean, std = add_outlier_flags(phys)
    sent_raw = run_sentiment(df, threshold) if "comments" in df.columns else pd.DataFrame()
    if not sent_raw.empty:
        # Compute sentiment on 2025 comments where available, else fall back to all years.
        # This ensures physicians with comments only in 2023/2024 are still evaluated.
        if "year" in sent_raw.columns and not sent_raw[sent_raw["year"] == 2025].empty:
            sent_2025 = sent_raw[sent_raw["year"] == 2025]
        else:
            sent_2025 = sent_raw
        sent_s_2025 = sentiment_summary(sent_2025)

        # For physicians with no 2025 comments, supplement from all-years sentiment
        sent_s_all  = sentiment_summary(sent_raw)
        missing     = ~sent_s_all["physician_id"].isin(sent_s_2025["physician_id"])
        sent_s      = pd.concat([sent_s_2025, sent_s_all[missing]], ignore_index=True)

        phys = merge_sentiment(phys, sent_s)
    else:
        phys["total_comments"]   = 0
        phys["negative_ratio"]   = np.nan
        phys["avg_compound"]     = np.nan
        phys["negative_outlier"] = False
    phys = add_risk(phys)
    phys["department"] = dept_name
    return df, phys, sent_raw

# ─── FIXED SETTINGS ──────────────────────────────────────────────────────────
min_forms   = 1
sent_thresh = -0.01

# ─── GITHUB DATA SOURCES ─────────────────────────────────────────────────────
# Replace each value below with your raw GitHub URL
# Raw URL format: https://raw.githubusercontent.com/<user>/<repo>/main/<path>.csv

GITHUB_URLS = {
    # ── Behaviour survey CSVs (3 departments × 3 years) ──────────────────────
    "aubmc_23": "AUBMC, Behavior survey responses, 2023.csv",
    "aubmc_24": "AUBMC, Behavior survey responses, 2024.csv",
    "aubmc_25": "AUBMC, Behavior raw data 2025.csv",
    "ed_23":    "ED, Behavior survey responses, 2023.csv",
    "ed_24":    "ED, Behavior survey responses, 2024.csv",
    "ed_25":    "ED, Behavior raw data 2025.csv",
    "patho_23": "Patho & Lab, Behavior survey responses, 2023.csv",
    "patho_24": "Patho,lab behavior survey responses, 2024.csv",
    "patho_25": "Patho,Lab, Behavior raw data 2025.csv",

    # ── Physicians Indicators CSV (Tab 6 — Departments & Divisions) ───────────
    "indicators": "Physicians indicators.csv",
}

# ─── DATA LOADING ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_from_github(urls, min_f, threshold, _version="v5.1"):
    def fetch(url):
        if not url or url.startswith("REPLACE"):
            return None
        try:
            return pd.read_csv(url)
        except Exception as e:
            st.warning(f"Could not load {url}: {e}")
            return None

    def load_dept(keys, name):
        frames = [fetch(urls[k]) for k in keys]
        frames = [f for f in frames if f is not None]
        if not frames: return None, None, None
        raw = pd.concat(frames, ignore_index=True)
        return process_dept(raw, name, threshold, min_f=min_f)

    aubmc_raw, aubmc_phys, aubmc_sent = load_dept(["aubmc_23","aubmc_24","aubmc_25"], "AUBMC")
    ed_raw,    ed_phys,    ed_sent    = load_dept(["ed_23","ed_24","ed_25"],           "ED")
    patho_raw, patho_phys, patho_sent = load_dept(["patho_23","patho_24","patho_25"], "Pathology")

    return {
        "AUBMC":     (aubmc_raw, aubmc_phys, aubmc_sent),
        "ED":        (ed_raw,    ed_phys,    ed_sent),
        "Pathology": (patho_raw, patho_phys, patho_sent),
    }

# ── PROCESS DATA ─────────────────────────────────────────────────────────────
with st.spinner("Loading data from GitHub and running VADER sentiment analysis..."):
    data = load_from_github(
        GITHUB_URLS,
        min_forms,
        sent_thresh,
        _version="v5.1"
    )

# Build combined physician table from available departments
all_phys_frames = []
for name, (raw, phys, sent) in data.items():
    if phys is not None and len(phys) > 0:
        all_phys_frames.append(phys)

if not all_phys_frames:
    st.error("No data could be processed. Please check your uploaded files.")
    st.stop()

all_phys = pd.concat(all_phys_frames, ignore_index=True)
available_depts = [n for n, (r,p,s) in data.items() if p is not None and len(p) > 0]

# ─── MAIN HEADER ─────────────────────────────────────────────────────────────
st.markdown("# 🏥 AUBMC Physician Performance Dashboard")
st.markdown(f"**Departments active:** {'  ·  '.join(available_depts)}  &nbsp;&nbsp; **Physicians:** {len(all_phys):,}  &nbsp;&nbsp; **Years:** 2023–2025")
st.markdown("---")

# ─── TABS ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Executive Summary",
    "🎯 Flagged Physicians",
    "📊 Project View",
    "💬 Sentiment Explorer",
    "📈 Trends (2023–2025)",
    "🏢 Departments & Divisions"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EXECUTIVE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">🔑 Key Performance Indicators</div>', unsafe_allow_html=True)

    total      = len(all_phys)
    priority   = (all_phys["risk_score"] >= 3).sum()
    monitor    = (all_phys["risk_score"].between(1, 2)).sum()
    clear      = (all_phys["risk_score"] == 0).sum()
    avg_score  = all_phys["avg_behavior_score"].mean()
    pct_neg    = (all_phys["negative_outlier"] == True).sum()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(f"""<div class="metric-card neutral">
            <div class="metric-label">Total Physicians</div>
            <div class="metric-value">{total}</div>
            <div class="metric-sub">across {len(available_depts)} dept(s)</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card danger">
            <div class="metric-label">Priority Review</div>
            <div class="metric-value">{priority}</div>
            <div class="metric-sub">{priority/total*100:.1f}% of physicians</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card warning">
            <div class="metric-label">Monitor</div>
            <div class="metric-value">{monitor}</div>
            <div class="metric-sub">{monitor/total*100:.1f}% of physicians</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card success">
            <div class="metric-label">Clear</div>
            <div class="metric-value">{clear}</div>
            <div class="metric-sub">{clear/total*100:.1f}% of physicians</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        color_class = "danger" if avg_score < 2.5 else ("warning" if avg_score < 3.0 else "success")
        st.markdown(f"""<div class="metric-card {color_class}">
            <div class="metric-label">Overall Avg Score</div>
            <div class="metric-value">{avg_score:.2f}</div>
            <div class="metric-sub">scale: 0 – 4</div>
        </div>""", unsafe_allow_html=True)
    with c6:
        st.markdown(f"""<div class="metric-card {'danger' if pct_neg > 0 else 'success'}">
            <div class="metric-label">Neg. Sentiment Flags</div>
            <div class="metric-value">{pct_neg}</div>
            <div class="metric-sub">{pct_neg/total*100:.1f}% of physicians</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Risk Score — big and prominent ───────────────────────────────────────
    st.markdown('<div class="section-header">🎯 Risk Score Breakdown</div>', unsafe_allow_html=True)
    risk_vals = [
        int((all_phys["risk_score"] == 0).sum()),
        int(all_phys["risk_score"].between(1, 2).sum()),
        int((all_phys["risk_score"] >= 3).sum()),
    ]
    total_phys_r = sum(risk_vals)
    fig_risk, ax_risk = plt.subplots(figsize=(10, 4.5))
    risk_labels_r = ["Clear (0)", "Monitor (1–2)", "Priority (3–4)"]
    risk_colors_r = ["#10b981", "#f59e0b", "#ef4444"]
    bars_r = ax_risk.bar(risk_labels_r, risk_vals, color=risk_colors_r,
                         edgecolor="white", linewidth=2, width=0.45)
    for bar, val in zip(bars_r, risk_vals):
        pct = val / total_phys_r * 100 if total_phys_r > 0 else 0
        ax_risk.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     str(val), ha="center", va="bottom", fontweight="900", fontsize=24, color="#1f2937")
        if val > 0:
            ax_risk.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                         f"{pct:.1f}%", ha="center", va="center",
                         fontweight="700", fontsize=15, color="white")
    ax_risk.set_ylabel("Number of Physicians", fontsize=12)
    ax_risk.set_title("Composite Risk Distribution — All Physicians", fontsize=14, fontweight="bold", pad=14)
    ax_risk.tick_params(axis="x", labelsize=14)
    ax_risk.grid(axis="y", alpha=0.3, linestyle="--")
    ax_risk.set_facecolor("#fafafa"); fig_risk.patch.set_facecolor("white")
    ax_risk.spines["top"].set_visible(False); ax_risk.spines["right"].set_visible(False)
    plt.tight_layout(); st.pyplot(fig_risk, use_container_width=True); plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Department Risk Comparison ────────────────────────────────────────────
    st.markdown('<div class="section-header">🏥 Department Risk Comparison</div>', unsafe_allow_html=True)

    dept_risk_rows = []
    for dept in available_depts:
        _, phys, _ = data[dept]
        if phys is None: continue
        total_d   = len(phys)
        priority_d = int((phys["risk_score"] >= 3).sum())
        monitor_d  = int((phys["risk_score"].between(1,2)).sum())
        clear_d    = int((phys["risk_score"] == 0).sum())
        dept_risk_rows.append({
            "dept": dept, "Priority": priority_d,
            "Monitor": monitor_d, "Clear": clear_d, "total": total_d,
            "avg": round(phys["avg_behavior_score"].mean(), 2),
        })
    dept_risk_df = pd.DataFrame(dept_risk_rows)

    x      = np.arange(len(dept_risk_df))
    width  = 0.25
    fig_dr, ax_dr = plt.subplots(figsize=(10, 4.5))
    b1 = ax_dr.bar(x - width, dept_risk_df["Priority"], width, color="#ef4444", alpha=0.88, label="Priority", edgecolor="white")
    b2 = ax_dr.bar(x,          dept_risk_df["Monitor"],  width, color="#f59e0b", alpha=0.88, label="Monitor",  edgecolor="white")
    b3 = ax_dr.bar(x + width,  dept_risk_df["Clear"],    width, color="#10b981", alpha=0.88, label="Clear",    edgecolor="white")
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax_dr.text(bar.get_x() + bar.get_width()/2, h + 0.2,
                           str(int(h)), ha="center", va="bottom", fontsize=10, fontweight="700")
    # Avg score annotation per dept
    for i, row in dept_risk_df.iterrows():
        ax_dr.text(i, dept_risk_df[["Priority","Monitor","Clear"]].iloc[i].max() + 2.5,
                   f"Avg: {row['avg']}", ha="center", fontsize=9,
                   color="#6b7280", fontweight="600")
    ax_dr.set_xticks(x)
    ax_dr.set_xticklabels(dept_risk_df["dept"], fontsize=12)
    ax_dr.set_ylabel("Number of Physicians", fontsize=10)
    ax_dr.set_title("Priority · Monitor · Clear by Department", fontsize=13, fontweight="bold", pad=12)
    ax_dr.legend(fontsize=10, loc="upper right")
    ax_dr.grid(axis="y", alpha=0.3, linestyle="--")
    ax_dr.set_facecolor("#fafafa"); fig_dr.patch.set_facecolor("white")
    ax_dr.spines["top"].set_visible(False); ax_dr.spines["right"].set_visible(False)
    plt.tight_layout(); st.pyplot(fig_dr, use_container_width=True); plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Top Flagged Physicians + Sentiment Snapshot ───────────────────────────
    col_left, col_right = st.columns([1.6, 1])

    with col_left:
        st.markdown('<div class="section-header">⚠️ Top Physicians Needing Attention</div>', unsafe_allow_html=True)
        top_flagged = (
            all_phys[all_phys["risk_score"] >= 1]
            .sort_values(["risk_score", "avg_behavior_score"], ascending=[False, True])
            .head(10)
        )
        if top_flagged.empty:
            st.success("No physicians flagged across all departments.")
        else:
            for _, fp in top_flagged.iterrows():
                rs = int(fp["risk_score"])
                bg     = "#fef2f2" if rs >= 3 else "#fffbeb"
                border = "#ef4444" if rs >= 3 else "#f59e0b"
                label  = "⚠ Priority" if rs >= 3 else "👁 Monitor"
                flags  = []
                if fp.get("low_iqr_outlier", False): flags.append("IQR")
                if fp.get("low_z_outlier",   False): flags.append("Z")
                if fp.get("low_bottom10",    False): flags.append("P10")
                if fp.get("negative_outlier",False): flags.append("Sent.")
                neg_r  = f"{fp['negative_ratio']:.0%}" if pd.notna(fp.get("negative_ratio")) else "—"
                st.markdown(f"""
                <div style="background:{bg}; border-left:4px solid {border}; border-radius:8px;
                            padding:10px 16px; margin-bottom:8px; box-shadow:0 1px 3px rgba(0,0,0,0.05)">
                    <div style="display:flex; justify-content:space-between; align-items:center">
                        <span style="font-size:14px; font-weight:700; color:#111827">{fp["physician_id"]}</span>
                        <span style="font-size:12px; font-weight:700; color:{border}">{label} &nbsp;|&nbsp; Score {fp["avg_behavior_score"]:.2f}</span>
                    </div>
                    <div style="font-size:11px; color:#6b7280; margin-top:4px">
                        {fp.get("department","—")} &nbsp;·&nbsp; {int(fp["n_forms"])} evals
                        &nbsp;·&nbsp; Flags: <b>{"  ".join(flags) if flags else "—"}</b>
                        &nbsp;·&nbsp; Neg. comments: <b>{neg_r}</b>
                    </div>
                </div>""", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-header">💬 Sentiment Snapshot</div>', unsafe_allow_html=True)

        # Gather all sentiment data
        all_sent_frames = []
        for dn in available_depts:
            _, _, sr = data[dn]
            if sr is not None and not sr.empty:
                all_sent_frames.append(sr)

        if all_sent_frames:
            sent_all = pd.concat(all_sent_frames, ignore_index=True)
            total_c  = len(sent_all)
            neg_c    = (sent_all["sentiment"] == "NEGATIVE").sum()
            pos_c    = (sent_all["sentiment"] == "POSITIVE").sum()
            neu_c    = (sent_all["sentiment"] == "NEUTRAL").sum()
            neg_pct  = neg_c / total_c * 100 if total_c > 0 else 0
            pos_pct  = pos_c / total_c * 100 if total_c > 0 else 0
            neu_pct  = neu_c / total_c * 100 if total_c > 0 else 0

            st.markdown(f"""
            <div style="background:white; border-radius:12px; padding:20px 22px;
                        box-shadow:0 1px 4px rgba(0,0,0,0.08); margin-bottom:12px">
                <div style="font-size:12px; color:#6b7280; font-weight:600; text-transform:uppercase; letter-spacing:.05em">Total Comments</div>
                <div style="font-size:32px; font-weight:700; color:#111827">{total_c:,}</div>
                <div style="font-size:12px; color:#9ca3af; margin-top:2px">across all departments</div>
            </div>""", unsafe_allow_html=True)

            # Sentiment bar
            fig_sent, ax_sent = plt.subplots(figsize=(4, 1.8))
            ax_sent.barh([""], [neg_pct], color="#ef4444", alpha=0.88, label=f"Negative {neg_pct:.1f}%")
            ax_sent.barh([""], [neu_pct], left=[neg_pct], color="#9ca3af", alpha=0.75, label=f"Neutral {neu_pct:.1f}%")
            ax_sent.barh([""], [pos_pct], left=[neg_pct+neu_pct], color="#10b981", alpha=0.88, label=f"Positive {pos_pct:.1f}%")
            for val, left, col in [(neg_pct,0,"#ef4444"),(neu_pct,neg_pct,"#6b7280"),(pos_pct,neg_pct+neu_pct,"#10b981")]:
                if val > 4:
                    ax_sent.text(left + val/2, 0, f"{val:.1f}%", ha="center", va="center",
                                 fontsize=9, fontweight="700", color="white")
            ax_sent.set_xlim(0, 100)
            ax_sent.set_title("Comment Sentiment Split", fontsize=10, fontweight="bold")
            ax_sent.axis("off")
            ax_sent.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.55), ncol=3)
            fig_sent.patch.set_facecolor("white")
            plt.tight_layout(); st.pyplot(fig_sent, use_container_width=True); plt.close()

            # Neg sentiment outlier physicians
            neg_flag_n = int(all_phys["negative_outlier"].sum()) if "negative_outlier" in all_phys.columns else 0
            st.markdown(f"""
            <div style="background:#fef2f2; border-left:4px solid #ef4444; border-radius:8px;
                        padding:12px 16px; margin-top:8px">
                <div style="font-size:12px; color:#6b7280; font-weight:600">Negative Sentiment Outliers</div>
                <div style="font-size:28px; font-weight:700; color:#ef4444">{neg_flag_n}</div>
                <div style="font-size:11px; color:#9ca3af">physicians above IQR upper fence on negative ratio</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.info("No comment data available.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Department Summary Table ──────────────────────────────────────────────
    st.markdown('<div class="section-header">Department Summary</div>', unsafe_allow_html=True)
    summary_rows = []
    for dept in available_depts:
        _, phys, _ = data[dept]
        if phys is None: continue
        row = {
            "Department":      dept,
            "Physicians":      len(phys),
            "Avg Score":       f"{phys['avg_behavior_score'].mean():.2f}",
            "Priority (3-4)":  int((phys['risk_score']>=3).sum()),
            "Monitor (1-2)":   int((phys['risk_score'].between(1,2)).sum()),
            "Clear (0)":       int((phys['risk_score']==0).sum()),
            "Sentiment Flags": int(phys['negative_outlier'].sum()) if 'negative_outlier' in phys.columns else 0,
        }
        summary_rows.append(row)
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FLAGGED PHYSICIANS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">🎯 Physician Risk Register</div>', unsafe_allow_html=True)

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        dept_filter = st.selectbox("Department", ["All"] + available_depts, key="flag_dept")
    with col_f2:
        risk_filter = st.selectbox("Risk Level", ["All","Priority (3-4)","Monitor (1-2)","Clear (0)"], key="flag_risk")
    with col_f3:
        sort_by = st.selectbox("Sort by", ["Risk Score ↓", "Avg Score ↑", "Neg. Ratio ↓"], key="flag_sort")

    df_view = all_phys.copy()
    if dept_filter != "All":
        df_view = df_view[df_view["department"] == dept_filter]
    if risk_filter == "Priority (3-4)":
        df_view = df_view[df_view["risk_score"] >= 3]
    elif risk_filter == "Monitor (1-2)":
        df_view = df_view[df_view["risk_score"].between(1, 2)]
    elif risk_filter == "Clear (0)":
        df_view = df_view[df_view["risk_score"] == 0]

    if sort_by == "Risk Score ↓":
        df_view = df_view.sort_values(["risk_score","avg_behavior_score"], ascending=[False,True])
    elif sort_by == "Avg Score ↑":
        df_view = df_view.sort_values("avg_behavior_score")
    else:
        df_view = df_view.sort_values("negative_ratio", ascending=False)

    # Table with risk pills
    display_cols = {
        "physician_id":       "Physician ID",
        "department":         "Department",
        "avg_behavior_score": "Avg Score",
        "n_forms":            "Evaluations",
        "z_score":            "Z-Score",
        "low_iqr_outlier":    "IQR Flag",
        "low_z_outlier":      "Z-Flag",
        "low_bottom10":       "Bottom 10%",
        "negative_outlier":   "Neg. Sentiment",
        "risk_score":         "Risk Score",
    }
    show_df = df_view[[c for c in display_cols if c in df_view.columns]].copy()
    show_df.columns = [display_cols[c] for c in show_df.columns]
    if "Avg Score" in show_df.columns:
        show_df["Avg Score"] = show_df["Avg Score"].round(3)
    if "Z-Score" in show_df.columns:
        show_df["Z-Score"] = show_df["Z-Score"].round(2)
    if "Neg. Ratio" in show_df.columns:
        show_df["Neg. Ratio"] = show_df["Neg. Ratio"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
    if "Avg Compound" in show_df.columns:
        show_df["Avg Compound"] = show_df["Avg Compound"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")

    st.dataframe(
        show_df.reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Risk Score": st.column_config.ProgressColumn(
                "Risk Score", min_value=0, max_value=4, format="%d"
            ),
            "Avg Score": st.column_config.ProgressColumn(
                "Avg Score", min_value=0, max_value=4, format="%.3f"
            ),
        }
    )

    # CSV export
    csv_out = df_view.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Export filtered table as CSV", csv_out,
                       "flagged_physicians.csv", "text/csv")

    # ── Physician deep-dive ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">🔍 Individual Physician Deep-Dive</div>', unsafe_allow_html=True)

    dd1, dd2, dd3 = st.columns(3)
    with dd1:
        dd_dept = st.selectbox("Department", available_depts, key="deep_dept")
    with dd2:
        # Build year list from raw data for selected dept
        raw_dd, phys_dd_all, sent_dd = data[dd_dept]
        if raw_dd is not None and "year" in raw_dd.columns:
            yr_opts = ["All Years"] + sorted(raw_dd["year"].dropna().unique().astype(int).tolist(), reverse=True)
        else:
            yr_opts = ["All Years"]
        dd_year = st.selectbox("Year", yr_opts, key="deep_year")
    with dd3:
        # Filter physicians by dept (and year if selected)
        if raw_dd is not None:
            if dd_year == "All Years":
                dd_raw_filt = raw_dd
            else:
                dd_raw_filt = raw_dd[raw_dd["year"] == int(dd_year)]
            phys_in_yr = sorted(dd_raw_filt["physician_id"].dropna().unique().tolist())
        else:
            phys_in_yr = []
        if phys_in_yr:
            selected_id = st.selectbox("Physician ID", phys_in_yr, key="deep_id")
        else:
            selected_id = None
            st.info("No physicians for this selection.")

    if selected_id and raw_dd is not None:
        # Get aggregated row — use year-filtered data if year selected
        if dd_year == "All Years":
            phys_src = phys_dd_all
        else:
            yr_raw = raw_dd[raw_dd["year"] == int(dd_year)]
            if not yr_raw.empty:
                phys_src = aggregate_physician(yr_raw)
                phys_src, _, _ = add_outlier_flags(phys_src)
                # Re-merge 2025 sentiment for this year's risk computation
                if sent_dd is not None and not sent_dd.empty and int(dd_year) == 2025:
                    sent_yr = sentiment_summary(sent_dd)
                    phys_src = merge_sentiment(phys_src, sent_yr)
                else:
                    phys_src["negative_outlier"] = False
                phys_src = add_risk(phys_src)
                phys_src["department"] = dd_dept
            else:
                phys_src = phys_dd_all

        row_mask = phys_src["physician_id"] == selected_id if phys_src is not None else pd.Series(False)
        if phys_src is None or not row_mask.any():
            st.warning(f"No data found for {selected_id}.")
        else:
            row = phys_src[row_mask].iloc[0]
            year_label = f" — {dd_year}" if dd_year != "All Years" else " — All Years"

            dc1, dc2, dc3, dc4 = st.columns(4)
            with dc1: st.metric("Department", dd_dept)
            with dc2: st.metric(f"Avg Score{year_label}", f"{row['avg_behavior_score']:.3f} / 4.0")
            with dc3: st.metric("Evaluations", int(row["n_forms"]))
            with dc4: st.metric("Risk Score (0–4)", f"{int(row['risk_score'])} / 4")

            dc5, dc6, dc7, dc8 = st.columns(4)
            with dc5: st.metric("Z-Score", f"{row.get('z_score', 0):.2f}")
            with dc6:
                iqr_dd = "🔴 YES" if row.get("low_iqr_outlier", False) else "🟢 No"
                st.metric("IQR Outlier", iqr_dd)
            with dc7:
                neg_r = row.get("negative_ratio", np.nan)
                st.metric("Neg. Comment Ratio", f"{neg_r:.1%}" if pd.notna(neg_r) else "—")
            with dc8:
                neg_s = "🔴 YES" if row.get("negative_outlier", False) else "🟢 No"
                st.metric("Neg. Sentiment", neg_s)

            # Comments — filter by year if selected
            if sent_dd is not None and not sent_dd.empty and "physician_id" in sent_dd.columns:
                phys_comments = sent_dd[sent_dd["physician_id"] == selected_id].copy()
                if dd_year != "All Years" and "year" in phys_comments.columns:
                    phys_comments = phys_comments[phys_comments["year"] == int(dd_year)]
                if not phys_comments.empty:
                    yr_suffix = f", {dd_year}" if dd_year != "All Years" else ""
                    st.markdown(f"**Peer Comments** ({len(phys_comments)} total{yr_suffix}):")
                    for _, crow in phys_comments.sort_values("compound").iterrows():
                        css_class = "neg" if crow["sentiment"]=="NEGATIVE" else ("pos" if crow["sentiment"]=="POSITIVE" else "neu")
                        emoji     = "🔴" if crow["sentiment"]=="NEGATIVE" else ("🟢" if crow["sentiment"]=="POSITIVE" else "⚪")
                        year_str  = str(int(crow["year"])) if "year" in crow and pd.notna(crow.get("year")) else "—"
                        rater     = crow.get("raters_group","—")
                        st.markdown(f"""
                        <div class="comment-card {css_class}">
                            <div style="font-size:11px; color:#9ca3af; margin-bottom:6px">
                                {emoji} <b>{crow["sentiment"]}</b> &nbsp;·&nbsp; Score: <b>{crow["compound"]:.3f}</b>
                                &nbsp;·&nbsp; Year: {year_str} &nbsp;·&nbsp; {rater}
                            </div>
                            <div style="font-size:14px; color:#374151">{crow["comments"]}</div>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.info("No comments available for this selection.")
    elif not selected_id:
        st.info("No physicians match the current filter.")



# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DEPARTMENT VIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">📊 Department-Level Analysis</div>', unsafe_allow_html=True)

    dept_sel = st.selectbox("Select Department", available_depts, key="dept_view")
    _, phys_d, _ = data[dept_sel]

    if phys_d is None or phys_d.empty:
        st.warning("No data available for this department.")
    else:
        d1, d2, d3, d4 = st.columns(4)
        with d1: st.metric("Physicians", len(phys_d))
        with d2: st.metric("Dept. Mean Score", f"{phys_d['avg_behavior_score'].mean():.3f}")
        with d3: st.metric("IQR Outliers", int(phys_d["low_iqr_outlier"].sum()) if "low_iqr_outlier" in phys_d.columns else 0)
        with d4: st.metric("Priority Flags", int((phys_d["risk_score"]>=3).sum()))

        col_l, col_r = st.columns(2)

        # IQR scatter plot
        with col_l:
            st.markdown("**IQR Outlier View — Score Distribution**")
            fig, ax = plt.subplots(figsize=(6, 4.5))
            scores_d  = phys_d["avg_behavior_score"]
            Q1d, Q3d  = scores_d.quantile(0.25), scores_d.quantile(0.75)
            iqr_fence = Q1d - 1.5 * (Q3d - Q1d)
            normal    = phys_d[~phys_d["low_iqr_outlier"]] if "low_iqr_outlier" in phys_d.columns else phys_d
            outliers  = phys_d[phys_d["low_iqr_outlier"]]  if "low_iqr_outlier" in phys_d.columns else phys_d.iloc[0:0]
            ax.scatter(normal.index,  normal["avg_behavior_score"],
                       alpha=0.6, color="#3b82f6", s=55, label="Within range", zorder=3)
            ax.scatter(outliers.index, outliers["avg_behavior_score"],
                       color="#ef4444", s=100, zorder=5, label=f"IQR Outliers (n={len(outliers)})")
            ax.axhline(iqr_fence, color="#ef4444", linewidth=2, linestyle="--",
                       label=f"IQR Lower Fence ({iqr_fence:.2f})")
            ax.axhline(scores_d.mean(), color="#1d4ed8", linewidth=1.5, linestyle=":",
                       label=f"Mean ({scores_d.mean():.2f})")
            ax.set_xlabel("Physician Index", fontsize=10)
            ax.set_ylabel("Avg Behaviour Score", fontsize=10)
            ax.set_title(f"{dept_sel} — IQR Score Outliers", fontsize=11, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3, linestyle="--")
            ax.set_facecolor("#fafafa")
            fig.patch.set_facecolor("white")
            st.pyplot(fig, use_container_width=True)
            plt.close()

        # Within-dept colleague comparison histogram
        with col_r:
            st.markdown("**Score Distribution — Colleague Comparison**")
            fig2, ax2 = plt.subplots(figsize=(6, 4.5))
            scores = phys_d["avg_behavior_score"]
            n, bins, patches = ax2.hist(scores, bins=20, edgecolor="white",
                                         linewidth=0.8, color="#3b82f6", alpha=0.75)

            # Colour IQR outliers red in the histogram
            Q1h, Q3h   = scores.quantile(0.25), scores.quantile(0.75)
            iqr_thresh = Q1h - 1.5 * (Q3h - Q1h)
            for patch, left_edge in zip(patches, bins[:-1]):
                if left_edge < iqr_thresh:
                    patch.set_facecolor("#ef4444")
                    patch.set_alpha(0.8)

            ax2.axvline(scores.mean(), color="#1d4ed8", linewidth=2,
                        linestyle="-", label=f"Mean ({scores.mean():.2f})")
            ax2.axvline(scores.quantile(0.10), color="#f59e0b", linewidth=1.5,
                        linestyle=":", label=f"10th pct ({scores.quantile(.1):.2f})")

            red_patch   = mpatches.Patch(color="#ef4444", alpha=0.8, label="Below IQR fence")
            blue_patch  = mpatches.Patch(color="#3b82f6", alpha=0.75, label="Within range")
            ax2.legend(handles=[red_patch, blue_patch] +
                       [plt.Line2D([0],[0],color="#1d4ed8",linewidth=2,label=f"Mean ({scores.mean():.2f})"),
                        plt.Line2D([0],[0],color="#f59e0b",linewidth=1.5,linestyle=":",label=f"10th pct")],
                       fontsize=8)

            ax2.set_xlabel("Avg Behaviour Score (0–4)", fontsize=10)
            ax2.set_ylabel("Number of Physicians", fontsize=10)
            ax2.set_title(f"{dept_sel} — Colleague Comparison", fontsize=11, fontweight="bold")
            ax2.grid(axis="y", alpha=0.3, linestyle="--")
            ax2.set_facecolor("#fafafa")
            fig2.patch.set_facecolor("white")
            st.pyplot(fig2, use_container_width=True)
            plt.close()

        # Outlier method comparison table
        st.markdown("**Outlier Method Comparison**")
        method_df = pd.DataFrame({
                "Method":       ["IQR Lower Fence", "Z-Score (≤−2)", "Bottom 10%", "Neg. Sentiment"],
                "Flag Column":  ["low_iqr_outlier", "low_z_outlier", "low_bottom10", "negative_outlier"],
            })
        method_df["Physicians Flagged"] = method_df["Flag Column"].apply(
            lambda c: int(phys_d[c].sum()) if c in phys_d.columns else 0
        )
        method_df["% of Department"] = (
            method_df["Physicians Flagged"] / len(phys_d) * 100
        ).round(1).astype(str) + "%"
        st.dataframe(method_df[["Method","Physicians Flagged","% of Department"]],
                     use_container_width=True, hide_index=True)

        # Within-dept ranking table
        st.markdown("**Physician Ranking within Department**")
        rank_cols = ["physician_id","avg_behavior_score","n_forms","z_score",
                     "low_iqr_outlier","low_z_outlier","low_bottom10",
                     "negative_outlier","risk_score"]
        rank_cols = [c for c in rank_cols if c in phys_d.columns]
        rank_df = phys_d[rank_cols].copy()
        rank_df = rank_df.sort_values("avg_behavior_score")
        rank_df["Percentile"] = (rank_df["avg_behavior_score"].rank(pct=True)*100).round(1).astype(str) + "%"
        rank_df["avg_behavior_score"] = rank_df["avg_behavior_score"].round(3)
        if "z_score" in rank_df.columns:
            rank_df["z_score"] = rank_df["z_score"].round(2)
        col_rename = {
            "physician_id":       "Physician ID",
            "avg_behavior_score": "Avg Score",
            "n_forms":            "Evaluations",
            "z_score":            "Z-Score",
            "low_iqr_outlier":    "IQR Flag",
            "low_z_outlier":      "Z-Flag",
            "low_bottom10":       "Bottom 10%",
            "negative_outlier":   "Neg. Sentiment",
            "risk_score":         "Risk Score",
        }
        rank_df = rank_df.rename(columns={k:v for k,v in col_rename.items() if k in rank_df.columns})
        st.dataframe(rank_df.reset_index(drop=True), use_container_width=True, hide_index=True,
                     column_config={"Risk Score": st.column_config.ProgressColumn(min_value=0, max_value=4, format="%d"),
                                    "Avg Score":  st.column_config.ProgressColumn(min_value=0, max_value=4, format="%.3f")})


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SENTIMENT EXPLORER (placeholder to close the with tab3 block)
# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SENTIMENT EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">💬 Sentiment Analysis</div>', unsafe_allow_html=True)

    # Gather sentiment data across all departments
    sent_frames = []
    for dn in available_depts:
        _, _, sr = data[dn]
        if sr is not None and not sr.empty:
            sr2 = sr.copy()
            sr2["dept"] = dn
            sent_frames.append(sr2)

    if not sent_frames:
        st.info("No comment data available. Upload behaviour survey CSVs to enable sentiment analysis.")
    else:
        all_sent = pd.concat(sent_frames, ignore_index=True)

        # ── KPI row ───────────────────────────────────────────────────────────
        sc1, sc2, sc3, sc4 = st.columns(4)
        total_c = len(all_sent)
        neg_c   = (all_sent["sentiment"] == "NEGATIVE").sum()
        pos_c   = (all_sent["sentiment"] == "POSITIVE").sum()
        neu_c   = (all_sent["sentiment"] == "NEUTRAL").sum()
        with sc1: st.metric("Total Comments", f"{total_c:,}")
        with sc2: st.metric("🔴 Negative", f"{neg_c:,} ({neg_c/total_c*100:.1f}%)")
        with sc3: st.metric("🟢 Positive", f"{pos_c:,} ({pos_c/total_c*100:.1f}%)")
        with sc4: st.metric("⚪ Neutral",  f"{neu_c:,} ({neu_c/total_c*100:.1f}%)")

        st.markdown("---")

        # ── Chart 1: Sentiment breakdown by department (stacked bar) ─────────
        st.markdown('<div class="section-header">📊 Sentiment Breakdown by Department</div>', unsafe_allow_html=True)

        dept_sent = (
            all_sent.groupby(["dept","sentiment"], as_index=False)
            .size()
            .rename(columns={"size":"count"})
        )
        dept_totals = dept_sent.groupby("dept")["count"].transform("sum")
        dept_sent["pct"] = dept_sent["count"] / dept_totals * 100

        depts_order  = dept_sent.groupby("dept")["count"].sum().sort_values(ascending=False).index.tolist()
        neg_pct  = dept_sent[dept_sent["sentiment"]=="NEGATIVE"].set_index("dept")["pct"].reindex(depts_order).fillna(0)
        pos_pct  = dept_sent[dept_sent["sentiment"]=="POSITIVE"].set_index("dept")["pct"].reindex(depts_order).fillna(0)
        neu_pct  = dept_sent[dept_sent["sentiment"]=="NEUTRAL"].set_index("dept")["pct"].reindex(depts_order).fillna(0)
        neg_cnt  = dept_sent[dept_sent["sentiment"]=="NEGATIVE"].set_index("dept")["count"].reindex(depts_order).fillna(0)
        pos_cnt  = dept_sent[dept_sent["sentiment"]=="POSITIVE"].set_index("dept")["count"].reindex(depts_order).fillna(0)

        fig_sb, ax_sb = plt.subplots(figsize=(10, max(4, len(depts_order)*0.45)))
        y = range(len(depts_order))
        b1 = ax_sb.barh(list(y), neg_pct.values, color="#ef4444", alpha=0.85, label="Negative")
        b2 = ax_sb.barh(list(y), neu_pct.values, left=neg_pct.values, color="#9ca3af", alpha=0.75, label="Neutral")
        b3 = ax_sb.barh(list(y), pos_pct.values, left=(neg_pct+neu_pct).values, color="#10b981", alpha=0.85, label="Positive")

        # Annotate negative % on each bar
        for i, (np_, nc) in enumerate(zip(neg_pct.values, neg_cnt.values)):
            if np_ > 3:
                ax_sb.text(np_/2, i, f"{np_:.1f}%", va="center", ha="center",
                           fontsize=8, fontweight="700", color="white")

        ax_sb.set_yticks(list(y))
        ax_sb.set_yticklabels(depts_order, fontsize=9)
        ax_sb.set_xlabel("% of Comments", fontsize=10)
        ax_sb.set_title("Sentiment Breakdown by Department (% of comments)", fontsize=12, fontweight="bold")
        ax_sb.axvline(100, color="#e5e7eb", linewidth=0.8)
        ax_sb.legend(fontsize=9, loc="lower right")
        ax_sb.set_xlim(0, 100)
        ax_sb.grid(axis="x", alpha=0.25, linestyle="--")
        ax_sb.set_facecolor("#fafafa")
        fig_sb.patch.set_facecolor("white")
        plt.tight_layout()
        st.pyplot(fig_sb, use_container_width=True)
        plt.close()

        st.markdown("---")

        # ── Chart 2: Yearly sentiment trend (2023-2025) ───────────────────────
        st.markdown('<div class="section-header">📈 Yearly Sentiment Trend (2023–2025)</div>', unsafe_allow_html=True)

        trend_dept_sent = st.selectbox("Filter by Department", ["All Departments"] + available_depts, key="sent_trend_dept")
        df_trend_sent = all_sent if trend_dept_sent == "All Departments" else all_sent[all_sent["dept"] == trend_dept_sent]

        if "year" not in df_trend_sent.columns or df_trend_sent["year"].isna().all():
            st.warning("Year data not available in sentiment data.")
        else:
            yr_sent = (
                df_trend_sent.groupby(["year","sentiment"], as_index=False)
                .size()
                .rename(columns={"size":"count"})
            )
            yr_totals = yr_sent.groupby("year")["count"].transform("sum")
            yr_sent["pct"] = yr_sent["count"] / yr_totals * 100
            years_s = sorted(df_trend_sent["year"].dropna().unique().astype(int))

            neg_yr = yr_sent[yr_sent["sentiment"]=="NEGATIVE"].set_index("year")["pct"].reindex(years_s).fillna(0)
            pos_yr = yr_sent[yr_sent["sentiment"]=="POSITIVE"].set_index("year")["pct"].reindex(years_s).fillna(0)
            neu_yr = yr_sent[yr_sent["sentiment"]=="NEUTRAL"].set_index("year")["pct"].reindex(years_s).fillna(0)
            total_yr = yr_sent.groupby("year")["count"].sum().reindex(years_s).fillna(0)

            fig_yr, (ax_yr1, ax_yr2) = plt.subplots(1, 2, figsize=(12, 4.5))

            # Left: stacked bar by year
            w = 0.5
            x = range(len(years_s))
            ax_yr1.bar(x, neg_yr.values, width=w, color="#ef4444", alpha=0.85, label="Negative")
            ax_yr1.bar(x, neu_yr.values, width=w, bottom=neg_yr.values, color="#9ca3af", alpha=0.75, label="Neutral")
            ax_yr1.bar(x, pos_yr.values, width=w, bottom=(neg_yr+neu_yr).values, color="#10b981", alpha=0.85, label="Positive")
            for xi, np_ in zip(x, neg_yr.values):
                ax_yr1.text(xi, np_/2, f"{np_:.1f}%", ha="center", va="center",
                            fontsize=9, fontweight="700", color="white")
            ax_yr1.set_xticks(list(x))
            ax_yr1.set_xticklabels([str(y) for y in years_s], fontsize=10)
            ax_yr1.set_ylabel("% of Comments", fontsize=10)
            ax_yr1.set_title("Sentiment Mix by Year", fontsize=11, fontweight="bold")
            ax_yr1.legend(fontsize=9)
            ax_yr1.set_ylim(0, 100)
            ax_yr1.grid(axis="y", alpha=0.3, linestyle="--")
            ax_yr1.set_facecolor("#fafafa")

            # Right: negative % trend line
            ax_yr2.plot(years_s, neg_yr.values, color="#ef4444", linewidth=2.5,
                        marker="o", markersize=8, label="Negative %")
            ax_yr2.plot(years_s, pos_yr.values, color="#10b981", linewidth=2.5,
                        marker="s", markersize=8, label="Positive %")
            for yr_v, nv, pv in zip(years_s, neg_yr.values, pos_yr.values):
                ax_yr2.annotate(f"{nv:.1f}%", (yr_v, nv), textcoords="offset points",
                                xytext=(0, 10), ha="center", fontsize=9, color="#ef4444", fontweight="700")
                ax_yr2.annotate(f"{pv:.1f}%", (yr_v, pv), textcoords="offset points",
                                xytext=(0,-15), ha="center", fontsize=9, color="#10b981", fontweight="700")
            ax_yr2.set_xticks(years_s)
            ax_yr2.set_xticklabels([str(y) for y in years_s], fontsize=10)
            ax_yr2.set_ylabel("% of Comments", fontsize=10)
            ax_yr2.set_title("Negative vs Positive Trend", fontsize=11, fontweight="bold")
            ax_yr2.legend(fontsize=9)
            ax_yr2.grid(alpha=0.3, linestyle="--")
            ax_yr2.set_facecolor("#fafafa")

            fig_yr.patch.set_facecolor("white")
            plt.tight_layout()
            st.pyplot(fig_yr, use_container_width=True)
            plt.close()

            # Summary table
            st.markdown("**Year-by-Year Summary**")
            yr_table = pd.DataFrame({
                "Year":             [str(y) for y in years_s],
                "Total Comments":   total_yr.astype(int).values,
                "Negative %":       neg_yr.round(1).values,
                "Neutral %":        neu_yr.round(1).values,
                "Positive %":       pos_yr.round(1).values,
            })
            st.dataframe(yr_table, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — TRENDS (2023–2025)
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">📈 Year-on-Year Trends (2023–2025)</div>', unsafe_allow_html=True)

    # ── Filters row ───────────────────────────────────────────────────────────
    tf1, tf2, tf3 = st.columns([1, 1.5, 1])
    with tf1:
        trend_dept = st.selectbox("Department", available_depts, key="trend_dept")
    raw_d, phys_d, _ = data[trend_dept]
    # Exclude self-evaluations from all trend calculations
    if raw_d is not None and "raters_group" in raw_d.columns:
        raw_d = raw_d[raw_d["raters_group"] != "Faculty Self-Evaluation"].copy()

    if raw_d is None or raw_d.empty or "year" not in raw_d.columns or raw_d["year"].isna().all():
        st.warning("Year data not available. Check that your files include a Fillout Date column.")
    else:
        years_avail = sorted(raw_d["year"].dropna().unique().astype(int))

        # Physician selector — "All" shows department-level view
        all_phys_ids = sorted(raw_d["physician_id"].dropna().unique().tolist())
        with tf2:
            view_mode = st.radio("View", ["Department Overall", "Individual Physician"],
                                 horizontal=True, key="trend_mode")
        with tf3:
            if view_mode == "Individual Physician":
                selected_phys = st.selectbox("Physician ID", all_phys_ids, key="trend_phys")
            else:
                selected_phys = None

        st.markdown("---")

        # ── DEPARTMENT OVERALL VIEW ───────────────────────────────────────────
        if view_mode == "Department Overall":

            trend_rows = []
            for yr in years_avail:
                df_yr   = raw_d[raw_d["year"] == yr]
                phys_yr = aggregate_physician(df_yr)
                phys_yr = phys_yr[phys_yr["n_forms"] >= min_forms].copy().reset_index(drop=True)
                phys_yr, _, _ = add_outlier_flags(phys_yr)
                trend_rows.append({
                    "Year":             yr,
                    "Physicians":       len(phys_yr),
                    "Avg Score":        round(phys_yr["avg_behavior_score"].mean(), 3),
                    "IQR Outliers":     int(phys_yr["low_iqr_outlier"].sum()) if "low_iqr_outlier" in phys_yr.columns else 0,
                    "% Flagged":        round(phys_yr["low_iqr_outlier"].mean()*100, 1) if "low_iqr_outlier" in phys_yr.columns else 0,
                    "Median Score":     round(phys_yr["avg_behavior_score"].median(), 3),
                    "Score Std":        round(phys_yr["avg_behavior_score"].std(), 3),
                })
            trend_df = pd.DataFrame(trend_rows)

            if len(trend_df) >= 2:
                delta_score = trend_df["Avg Score"].iloc[-1] - trend_df["Avg Score"].iloc[0]
                delta_flag  = trend_df["% Flagged"].iloc[-1] - trend_df["% Flagged"].iloc[0]
            else:
                delta_score = 0; delta_flag = 0

            tc1, tc2, tc3 = st.columns(3)
            with tc1: st.metric("Score Change (first→last year)", f"{delta_score:+.3f}",
                                 delta=f"{delta_score:+.3f}", delta_color="normal")
            with tc2: st.metric("% Flagged Change", f"{delta_flag:+.1f}%",
                                 delta=f"{delta_flag:+.1f}%", delta_color="inverse")
            with tc3: st.metric("Years Covered", len(years_avail))

            col_t1, col_t2 = st.columns(2)

            with col_t1:
                st.markdown("**Average Score Trend**")
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(trend_df["Year"], trend_df["Avg Score"], "o-",
                        color="#3b82f6", linewidth=2.5, markersize=8, label="Mean")
                ax.fill_between(
                    trend_df["Year"],
                    trend_df["Avg Score"] - trend_df["Score Std"],
                    trend_df["Avg Score"] + trend_df["Score Std"],
                    alpha=0.15, color="#3b82f6", label="±1 SD"
                )
                ax.plot(trend_df["Year"], trend_df["Median Score"], "s--",
                        color="#8b5cf6", linewidth=1.5, markersize=6, label="Median")
                for _, row in trend_df.iterrows():
                    ax.annotate(f"{row['Avg Score']:.2f}",
                                 (row["Year"], row["Avg Score"]),
                                 textcoords="offset points", xytext=(0,10),
                                 ha="center", fontsize=9, fontweight="600")
                ax.set_xticks(years_avail)
                ax.set_xlabel("Year", fontsize=10)
                ax.set_ylabel("Avg Behaviour Score (0–4)", fontsize=10)
                ax.set_title(f"{trend_dept} Score Trend", fontsize=11, fontweight="bold")
                ax.legend(fontsize=9)
                ax.grid(alpha=0.3, linestyle="--")
                ax.set_facecolor("#fafafa")
                fig.patch.set_facecolor("white")
                st.pyplot(fig, use_container_width=True)
                plt.close()

            with col_t2:
                st.markdown("**% Physicians Flagged by IQR**")
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                bar_cols = ["#10b981" if p < 10 else ("#f59e0b" if p < 20 else "#ef4444")
                            for p in trend_df["% Flagged"]]
                bars = ax2.bar(trend_df["Year"], trend_df["% Flagged"],
                               color=bar_cols, edgecolor="white", linewidth=1.5, width=0.5)
                for bar, val in zip(bars, trend_df["% Flagged"]):
                    ax2.text(bar.get_x() + bar.get_width()/2,
                             bar.get_height() + 0.3, f"{val}%",
                             ha="center", va="bottom", fontsize=10, fontweight="700")
                ax2.set_xticks(years_avail)
                ax2.set_xlabel("Year", fontsize=10)
                ax2.set_ylabel("% Physicians Below IQR Fence", fontsize=10)
                ax2.set_title(f"{trend_dept} — IQR Flagged Rate Over Time", fontsize=11, fontweight="bold")
                ax2.grid(axis="y", alpha=0.3, linestyle="--")
                ax2.set_facecolor("#fafafa")
                fig2.patch.set_facecolor("white")
                ax2.spines["top"].set_visible(False)
                ax2.spines["right"].set_visible(False)
                st.pyplot(fig2, use_container_width=True)
                plt.close()

            st.markdown("**Year-over-Year Summary Table**")
            st.dataframe(trend_df, use_container_width=True, hide_index=True)

        # ── INDIVIDUAL PHYSICIAN VIEW ─────────────────────────────────────────
        else:
            phys_raw = raw_d[raw_d["physician_id"] == selected_phys].copy()

            if phys_raw.empty:
                st.warning(f"No data found for physician {selected_phys}.")
            else:
                # Build per-year stats for this physician
                phys_year_rows = []
                for yr in years_avail:
                    yr_data = phys_raw[phys_raw["year"] == yr]
                    if yr_data.empty:
                        continue
                    q_cols = [c for c in yr_data.columns if c.startswith("q_")]
                    # Flat mean across all question responses — consistent with aggregate_physician
                    flat_vals = yr_data[q_cols].values.flatten()
                    flat_vals = flat_vals[~pd.isnull(flat_vals)]
                    flat_mean = float(flat_vals.mean()) if len(flat_vals) > 0 else np.nan
                    phys_year_rows.append({
                        "Year":           yr,
                        "Forms":          len(yr_data),
                        "Avg Score":      round(flat_mean, 3),
                        "Min Score":      round(yr_data["overall_score"].min(), 3),
                        "Max Score":      round(yr_data["overall_score"].max(), 3),
                        "Score Std":      round(yr_data["overall_score"].std(), 3) if len(yr_data) > 1 else 0.0,
                    })
                phys_trend = pd.DataFrame(phys_year_rows)

                if phys_trend.empty:
                    st.warning("No year-level data available for this physician.")
                else:
                    # Compute dept avg per year for benchmarking — flat mean consistent with aggregate_physician
                    dept_avgs = {}
                    for yr in years_avail:
                        yr_df   = raw_d[raw_d["year"] == yr]
                        phys_yr_bench = aggregate_physician(yr_df)
                        phys_yr_bench = phys_yr_bench[phys_yr_bench["n_forms"] >= min_forms]
                        dept_avgs[yr] = phys_yr_bench["avg_behavior_score"].mean() if not phys_yr_bench.empty else np.nan

                    # KPI summary cards
                    years_seen = phys_trend["Year"].tolist()
                    first_score = phys_trend["Avg Score"].iloc[0]
                    last_score  = phys_trend["Avg Score"].iloc[-1]
                    delta_phys  = last_score - first_score
                    avg_forms   = phys_trend["Forms"].mean()
                    overall_avg = phys_trend["Avg Score"].mean()

                    pk1, pk2, pk3, pk4 = st.columns(4)
                    with pk1:
                        st.metric("Years on Record", len(years_seen))
                    with pk2:
                        st.metric("Overall Avg Score", f"{overall_avg:.3f} / 4.0")
                    with pk3:
                        st.metric("Score Change", f"{delta_phys:+.3f}",
                                  delta=f"{delta_phys:+.3f}", delta_color="normal")
                    with pk4:
                        st.metric("Avg Forms / Year", f"{avg_forms:.1f}")

                    st.markdown("---")
                    col_p1, col_p2 = st.columns(2)

                    # Physician score trend vs department average
                    with col_p1:
                        st.markdown(f"**Score Trend — Physician {selected_phys} vs Department Mean**")
                        fig, ax = plt.subplots(figsize=(6, 4))

                        # Department average line
                        dept_yr_list   = [yr for yr in years_avail if yr in dept_avgs]
                        dept_avg_list  = [dept_avgs[yr] for yr in dept_yr_list]
                        ax.plot(dept_yr_list, dept_avg_list, "s--",
                                color="#9ca3af", linewidth=1.5, markersize=5,
                                label=f"{trend_dept} Mean", alpha=0.8)

                        # Physician score line
                        ax.plot(phys_trend["Year"], phys_trend["Avg Score"], "o-",
                                color="#3b82f6", linewidth=2.5, markersize=9,
                                label=f"Physician {selected_phys}", zorder=5)

                        # Min-max shading for spread
                        if "Min Score" in phys_trend.columns and "Max Score" in phys_trend.columns:
                            ax.fill_between(phys_trend["Year"],
                                            phys_trend["Min Score"], phys_trend["Max Score"],
                                            alpha=0.1, color="#3b82f6", label="Score range")

                        # Label each data point
                        for _, row in phys_trend.iterrows():
                            ax.annotate(f"{row['Avg Score']:.2f}",
                                        (row["Year"], row["Avg Score"]),
                                        textcoords="offset points", xytext=(0, 10),
                                        ha="center", fontsize=9, fontweight="700",
                                        color="#1d4ed8")

                        ax.set_xticks(years_avail)
                        ax.set_xlabel("Year", fontsize=10)
                        ax.set_ylabel("Avg Behaviour Score (0–4)", fontsize=10)
                        ax.set_title(f"Physician {selected_phys} — Score Over Time",
                                     fontsize=11, fontweight="bold")
                        ax.legend(fontsize=9)
                        ax.grid(alpha=0.3, linestyle="--")
                        ax.set_facecolor("#fafafa")
                        fig.patch.set_facecolor("white")
                        st.pyplot(fig, use_container_width=True)
                        plt.close()

                    # Number of forms received per year
                    with col_p2:
                        st.markdown("**Evaluations Received Per Year**")
                        fig2, ax2 = plt.subplots(figsize=(6, 4))
                        bar_colors = ["#3b82f6"] * len(phys_trend)
                        bars = ax2.bar(phys_trend["Year"], phys_trend["Forms"],
                                       color=bar_colors, edgecolor="white",
                                       linewidth=1.5, width=0.5, alpha=0.85)
                        for bar, val in zip(bars, phys_trend["Forms"]):
                            ax2.text(bar.get_x() + bar.get_width()/2,
                                     bar.get_height() + 0.1, str(int(val)),
                                     ha="center", va="bottom",
                                     fontsize=11, fontweight="700")
                        ax2.set_xticks(phys_trend["Year"].tolist())
                        ax2.set_xlabel("Year", fontsize=10)
                        ax2.set_ylabel("Number of Evaluations", fontsize=10)
                        ax2.set_title(f"Physician {selected_phys} — Evaluations Per Year",
                                      fontsize=11, fontweight="bold")
                        ax2.grid(axis="y", alpha=0.3, linestyle="--")
                        ax2.set_facecolor("#fafafa")
                        fig2.patch.set_facecolor("white")
                        ax2.spines["top"].set_visible(False)
                        ax2.spines["right"].set_visible(False)
                        st.pyplot(fig2, use_container_width=True)
                        plt.close()

                    # Percentile rank vs department per year
                    st.markdown("**Percentile Rank within Department — Year by Year**")
                    pct_rows = []
                    for yr in phys_trend["Year"].tolist():
                        yr_all   = raw_d[raw_d["year"] == yr]
                        phys_agg = aggregate_physician(yr_all)
                        phys_agg = phys_agg[phys_agg["n_forms"] >= min_forms].copy()
                        if selected_phys in phys_agg["physician_id"].values:
                            phys_agg["pct"] = phys_agg["avg_behavior_score"].rank(pct=True) * 100
                            pct_val = phys_agg.loc[
                                phys_agg["physician_id"] == selected_phys, "pct"
                            ].values[0]
                            dept_avg_yr = phys_agg["avg_behavior_score"].mean()
                            pct_rows.append({
                                "Year": yr,
                                "Percentile Rank": round(pct_val, 1),
                                "Physician Score": round(phys_agg.loc[
                                    phys_agg["physician_id"] == selected_phys,
                                    "avg_behavior_score"].values[0], 3),
                                "Dept Mean Score":  round(dept_avg_yr, 3),
                                "Physicians in Dept": len(phys_agg),
                            })
                    pct_df = pd.DataFrame(pct_rows)

                    if not pct_df.empty:
                        fig3, ax3 = plt.subplots(figsize=(10, 3.5))
                        colours_pct = ["#10b981" if p >= 50 else ("#f59e0b" if p >= 25 else "#ef4444")
                                       for p in pct_df["Percentile Rank"]]
                        bars3 = ax3.barh(pct_df["Year"].astype(str),
                                         pct_df["Percentile Rank"],
                                         color=colours_pct, edgecolor="white",
                                         linewidth=1, height=0.45, alpha=0.85)
                        ax3.axvline(50, color="#6b7280", linestyle="--",
                                    linewidth=1.2, label="50th percentile")
                        ax3.axvline(25, color="#ef4444", linestyle=":",
                                    linewidth=1.2, label="25th percentile (concern)")
                        for bar, val in zip(bars3, pct_df["Percentile Rank"]):
                            ax3.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                                     f"{val:.0f}th",
                                     va="center", fontsize=10, fontweight="700")
                        ax3.set_xlabel("Percentile Rank within Department (100 = best)", fontsize=10)
                        ax3.set_title(f"Physician {selected_phys} — Percentile Rank Over Time",
                                      fontsize=11, fontweight="bold")
                        ax3.set_xlim(0, 110)
                        ax3.legend(fontsize=9)
                        ax3.grid(axis="x", alpha=0.3, linestyle="--")
                        ax3.set_facecolor("#fafafa")
                        fig3.patch.set_facecolor("white")
                        st.pyplot(fig3, use_container_width=True)
                        plt.close()

                        st.markdown("**Year-by-Year Summary for this Physician**")
                        merged_summary = phys_trend.merge(pct_df[["Year","Percentile Rank","Dept Mean Score","Physicians in Dept"]],
                                                          on="Year", how="left")
                        st.dataframe(merged_summary, use_container_width=True, hide_index=True)

                    # ── PEER COMPARISON ───────────────────────────────────────
                    st.markdown("---")
                    st.markdown('<div class="section-header">👥 Peer Comparison — Department Distribution</div>', unsafe_allow_html=True)

                    # Build per-year box plot data + selected physician dot
                    bp1, bp2 = st.columns(2)

                    with bp1:
                        st.markdown("**Score Distribution by Year — Selected Physician vs Peers**")
                        fig_bp, ax_bp = plt.subplots(figsize=(7, 5))

                        box_data  = []
                        positions = []
                        sel_dots  = []

                        for pos, yr in enumerate(years_avail):
                            yr_dept_scores = raw_d[raw_d["year"] == yr]["overall_score"].dropna().tolist()
                            if not yr_dept_scores:
                                continue
                            box_data.append(yr_dept_scores)
                            positions.append(pos)

                            # Selected physician dot for this year
                            sel_yr = raw_d[(raw_d["year"] == yr) & (raw_d["physician_id"] == selected_phys)]["overall_score"].dropna()
                            sel_dots.append((pos, sel_yr.mean() if not sel_yr.empty else np.nan))

                        bp = ax_bp.boxplot(
                            box_data, positions=positions, widths=0.5, patch_artist=True,
                            boxprops=dict(facecolor="#dbeafe", color="#3b82f6", linewidth=1.5),
                            medianprops=dict(color="#1d4ed8", linewidth=2.5),
                            whiskerprops=dict(color="#6b7280", linewidth=1.2),
                            capprops=dict(color="#6b7280", linewidth=1.5),
                            flierprops=dict(marker="o", color="#9ca3af", alpha=0.4, markersize=4),
                        )

                        # Selected physician dot per year
                        for pos, val in sel_dots:
                            if not np.isnan(val):
                                ax_bp.scatter(pos, val, color="#ef4444", s=120, zorder=10,
                                              marker="D", label=f"▶ {selected_phys}" if pos == positions[0] else "")
                                ax_bp.annotate(f"{val:.2f}", (pos, val),
                                               textcoords="offset points", xytext=(10, 0),
                                               fontsize=9, fontweight="700", color="#ef4444")

                        ax_bp.set_xticks(positions)
                        ax_bp.set_xticklabels([str(yr) for yr in years_avail[:len(positions)]], fontsize=11)
                        ax_bp.set_ylabel("Avg Behaviour Score (0–4)", fontsize=10)
                        ax_bp.set_ylim(0, 4.2)
                        ax_bp.set_title(f"{trend_dept} — Score Distribution per Year", fontsize=11, fontweight="bold")
                        ax_bp.legend(fontsize=9, loc="lower right")
                        ax_bp.grid(axis="y", alpha=0.3, linestyle="--")
                        ax_bp.set_facecolor("#fafafa")
                        fig_bp.patch.set_facecolor("white")
                        plt.tight_layout()
                        st.pyplot(fig_bp, use_container_width=True)
                        plt.close()

                    with bp2:
                        st.markdown("**Overall Score Distribution — Selected Physician vs All Peers**")
                        fig_bp2, ax_bp2 = plt.subplots(figsize=(7, 5))

                        all_dept_scores = raw_d["overall_score"].dropna().tolist()
                        sel_overall = raw_d[raw_d["physician_id"] == selected_phys]["overall_score"].dropna()
                        sel_mean = sel_overall.mean() if not sel_overall.empty else np.nan

                        ax_bp2.boxplot(
                            [all_dept_scores], positions=[0], widths=0.4, patch_artist=True,
                            boxprops=dict(facecolor="#dbeafe", color="#3b82f6", linewidth=1.5),
                            medianprops=dict(color="#1d4ed8", linewidth=2.5),
                            whiskerprops=dict(color="#6b7280", linewidth=1.2),
                            capprops=dict(color="#6b7280", linewidth=1.5),
                            flierprops=dict(marker="o", color="#9ca3af", alpha=0.4, markersize=4),
                        )

                        if not np.isnan(sel_mean):
                            ax_bp2.scatter(0, sel_mean, color="#ef4444", s=180, zorder=10,
                                           marker="D", label=f"▶ {selected_phys} ({sel_mean:.3f})")
                            ax_bp2.axhline(sel_mean, color="#ef4444", linestyle=":", linewidth=1.5, alpha=0.6)

                        dept_median = np.median(all_dept_scores)
                        dept_mean   = np.mean(all_dept_scores)
                        ax_bp2.axhline(dept_mean, color="#1d4ed8", linestyle="--", linewidth=1.5,
                                       label=f"Dept mean ({dept_mean:.3f})", alpha=0.8)

                        # Percentile of selected physician
                        if not np.isnan(sel_mean):
                            pct = (np.array(all_dept_scores) < sel_mean).mean() * 100
                            ax_bp2.text(0.3, sel_mean, f"{pct:.0f}th percentile",
                                        fontsize=10, fontweight="700",
                                        color="#ef4444", va="center")

                        ax_bp2.set_xticks([0])
                        ax_bp2.set_xticklabels([trend_dept], fontsize=10)
                        ax_bp2.set_ylabel("Avg Behaviour Score (0–4)", fontsize=10)
                        ax_bp2.set_ylim(0, 4.2)
                        ax_bp2.set_title("Overall Score — Physician vs Department", fontsize=11, fontweight="bold")
                        ax_bp2.legend(fontsize=9, loc="lower right")
                        ax_bp2.grid(axis="y", alpha=0.3, linestyle="--")
                        ax_bp2.set_facecolor("#fafafa")
                        fig_bp2.patch.set_facecolor("white")
                        plt.tight_layout()
                        st.pyplot(fig_bp2, use_container_width=True)
                        plt.close()

                    # ── Peer ranking table (compact) ───────────────────────────
                    st.markdown("**Peer Ranking Table**")
                    st.caption("Sorted by Overall Avg ↓ · Trend = last year minus first year")

                    peer_rows = []
                    for pid in all_phys_ids:
                        pid_data = raw_d[raw_d["physician_id"] == pid]
                        if pid_data.empty:
                            continue
                        row = {"Physician ID": pid}
                        all_scores = []
                        for yr in years_avail:
                            yr_data_pid = pid_data[pid_data["year"] == yr]
                            if yr_data_pid.empty:
                                avg = np.nan
                            else:
                                q_cols_pid = [c for c in yr_data_pid.columns if c.startswith("q_")]
                                flat_pid = yr_data_pid[q_cols_pid].values.flatten()
                                flat_pid = flat_pid[~pd.isnull(flat_pid)]
                                avg = round(float(flat_pid.mean()), 3) if len(flat_pid) > 0 else np.nan
                            row[str(yr)] = avg
                            if not np.isnan(avg):
                                all_scores.append(avg)
                        row["Overall Avg"] = round(np.mean(all_scores), 3) if all_scores else np.nan
                        valid = [row[str(yr)] for yr in years_avail if not np.isnan(row.get(str(yr), np.nan))]
                        trend_val = round(valid[-1] - valid[0], 3) if len(valid) >= 2 else np.nan
                        row["Trend"] = f"▲ {trend_val:+.3f}" if pd.notna(trend_val) and trend_val > 0 else (
                                       f"▼ {trend_val:+.3f}" if pd.notna(trend_val) and trend_val < 0 else "— 0.000")
                        peer_rows.append(row)

                    peer_df = pd.DataFrame(peer_rows).sort_values("Overall Avg", ascending=False).reset_index(drop=True)
                    peer_df.insert(0, "Rank", range(1, len(peer_df)+1))

                    def highlight_selected(row):
                        if row["Physician ID"] == selected_phys:
                            return ["background-color: #dbeafe; font-weight: bold"] * len(row)
                        return [""] * len(row)

                    yr_fmt = {str(yr): lambda x: f"{x:.3f}" if pd.notna(x) else "—" for yr in years_avail}
                    styled = peer_df.style.apply(highlight_selected, axis=1).format(
                        {**yr_fmt, "Overall Avg": lambda x: f"{x:.3f}" if pd.notna(x) else "—"}
                    )
                    st.dataframe(styled, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — DEPARTMENTS & DIVISIONS
# ═══════════════════════════════════════════════════════════════════════════════

DEPT_DIVISION_MAP = {
    "Anesthesia and Pain Medicine": [],
    "Dentofacial Medicine":         ["Orthodontics"],
    "Dermatology":                  [],
    "Diagnostic Radiology":         [],
    "Emergency Medicine":           [],
    "Family Medicine":              [],
    "Internal Medicine": [
        "Cardiology",
        "Endocrinology and Metabolism",
        "Gastroenterology",
        "General Internal Medicine and Geriatrics",
        "Geriatrics",
        "Hematology-Oncology",
        "Infectious Diseases",
        "Nephrology and Hypertension",
        "Pulmonary and Critical Care",
        "Rheumatology",
    ],
    "Neurology":        [],
    "Ob/Gyn": [
        "Obstetrics and Gynecology",
        "Maternal-Fetal Medicine",
        "Reproductive Endocrinology and Infertility",
        "Gynecologic Oncology",
    ],
    "Ophthalmology": [
        "Ophthalmology",
        "Vitreo-Retinal surgery",
        "Corneal/Refractive Surgery",
        "Oculoplastics/Orbital/Lacrimal surgery",
        "Pediatric Ophthalmology & Motility",
    ],
    "Otolaryngology, Head & Neck Surgery": [],
    "Pathology and Lab":  [],
    "Pediatrics": [
        "Pediatrics and Adolescent Medicine",
        "Neonatology",
        "Pediatric Cardiology",
        "Pediatric Critical Care",
        "Pediatric Endocrinology",
        "Pediatric Gastroenterology",
        "Pediatric Hematology-Oncology",
        "Pediatric Infectious Diseases",
        "Pediatric Nephrology",
        "Pediatric Neurology",
        "Pediatric Pulmonology",
    ],
    "Psychiatry":        [],
    "Radiation Oncology":[],
    "Surgery": [
        "General Surgery",
        "Pediatric Surgery Service",
        "Cardiothoracic Surgery",
        "Neurosurgery",
        "Orthopaedic Surgery",
        "Plastic Surgery",
        "Urology",
        "Vascular Surgery",
    ],
}

# Exact mapping: every Division value in the CSV -> parent Department label
DIV_TO_DEPT = {
    # Internal Medicine (13 rows in CSV)
    "Cardiology":                               "Internal Medicine",
    "Endocrinology":                            "Internal Medicine",
    "Endocrinology and Metabolism":             "Internal Medicine",
    "Gastroenterology":                         "Internal Medicine",
    "General Internal Medicine and Geriatrics": "Internal Medicine",
    "Geriatrics":                               "Internal Medicine",
    "Hematology-Oncology":                      "Internal Medicine",
    "Infectious Diseases":                      "Internal Medicine",
    "Internal Medicine":                        "Internal Medicine",
    "Nephrology":                               "Internal Medicine",
    "Nephrology and Hypertension":              "Internal Medicine",
    "Pulmonary and Critical Care":              "Internal Medicine",
    "Rheumatology":                             "Internal Medicine",
    # Surgery (8)
    "General Surgery":                          "Surgery",
    "Pediatric Surgery Service":                "Surgery",
    "Cardiothoracic Surgery":                   "Surgery",
    "Neurosurgery":                             "Surgery",
    "Orthopaedic Surgery":                      "Surgery",
    "Plastic Surgery":                          "Surgery",
    "Urology":                                  "Surgery",
    "Vascular Surgery":                         "Surgery",
    # Ob/Gyn (4)
    "Obstetrics and Gynecology":                "Ob/Gyn",
    "Maternal-Fetal Medicine":                  "Ob/Gyn",
    "Reproductive Endocrinology and Infertility": "Ob/Gyn",
    "Gynecologic Oncology":                     "Ob/Gyn",
    # Ophthalmology (5)
    "Ophthalmology":                            "Ophthalmology",
    "Vitreo-Retinal surgery":                   "Ophthalmology",
    "Corneal/Refractive Surgery":               "Ophthalmology",
    "Oculoplastics/Orbital/Lacrimal surgery":   "Ophthalmology",
    "Pediatric Ophthalmology & Motility":       "Ophthalmology",
    # Pediatrics (11)
    "Pediatrics and Adolescent Medicine":       "Pediatrics",
    "Neonatology":                              "Pediatrics",
    "Pediatric Cardiology":                     "Pediatrics",
    "Pediatric Critical Care":                  "Pediatrics",
    "Pediatric Endocrinology":                  "Pediatrics",
    "Pediatric Gastroenterology":               "Pediatrics",
    "Pediatric Hematology-Oncology":            "Pediatrics",
    "Pediatric Infectious Diseases":            "Pediatrics",
    "Pediatric Nephrology":                     "Pediatrics",
    "Pediatric Neurology":                      "Pediatrics",
    "Pediatric Pulmonology":                    "Pediatrics",
    # Standalone departments (each maps to itself)
    "Anesthesia and Pain Medicine":             "Anesthesia and Pain Medicine",
    "Dentofacial Medicine":                     "Dentofacial Medicine",
    "Orthodontics":                             "Dentofacial Medicine",
    "Dermatology":                              "Dermatology",
    "Diagnostic Radiology":                     "Diagnostic Radiology",
    "Emergency Medicine":                       "Emergency Medicine",
    "Family Medicine":                          "Family Medicine",
    "Neurology":                                "Neurology",
    "Otorhinolaryngology - Head and Neck Surgery": "Otolaryngology, Head & Neck Surgery",
    "Pathology and Lab":                        "Pathology and Lab",
    "Psychiatry":                               "Psychiatry",
    "Radiation Oncology":                       "Radiation Oncology",
}


with tab6:
    st.markdown('<div class="section-header">🏢 Departments & Divisions — Indicators Analysis</div>', unsafe_allow_html=True)

    @st.cache_data(show_spinner=False)
    def load_indicators(url, _version="v5.1"):
        if not url or url.startswith("REPLACE"):
            return None
        try:
            for enc in ["utf-8", "latin-1", "cp1252", "iso-8859-1"]:
                try:
                    df = pd.read_csv(url, encoding=enc)
                    break
                except (UnicodeDecodeError, Exception):
                    continue
            else:
                st.warning("Could not decode indicators file with any known encoding.")
                return None
        except Exception as e:
            st.warning(f"Could not load indicators file: {e}")
            return None
        df.columns = df.columns.str.strip()
        if "Division" in df.columns:
            df["Division_norm"] = df["Division"].str.strip()
        if "Department" in df.columns:
            df["Department"] = df["Department"].str.strip()
        for col in ["ClinicVisits", "ClinicWaitingTime", "PatientComplaints"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    ind_df = load_indicators(GITHUB_URLS.get("indicators", ""), _version="v5.1")

    if ind_df is None:
        st.info("Indicators data not available. Add the indicators URL to GITHUB_URLS['indicators'] in the source file.")
        st.markdown("---")
        st.markdown('<div class="section-header">📋 AUBMC Organisational Structure</div>', unsafe_allow_html=True)
        org_cols = st.columns(2)
        dept_list = list(DEPT_DIVISION_MAP.keys())
        half = len(dept_list) // 2
        for col, depts in zip(org_cols, [dept_list[:half], dept_list[half:]]):
            with col:
                for dept in depts:
                    divs = DEPT_DIVISION_MAP[dept]
                    if divs:
                        with st.expander(f"**{dept}** — {len(divs)} divisions"):
                            for d in divs:
                                st.markdown(f"&nbsp;&nbsp;&nbsp;• {d}")
                    else:
                        st.markdown(f"**{dept}**")
    else:

        # ── Cycle filter ──────────────────────────────────────────────────────
        cycles = ["All"] + sorted(ind_df["FiscalCycle"].dropna().unique().tolist(), reverse=True) \
                 if "FiscalCycle" in ind_df.columns else ["All"]
        sel_cycle = st.selectbox("Fiscal Cycle", cycles, key="ind_cycle")
        df_filt = ind_df if sel_cycle == "All" else ind_df[ind_df["FiscalCycle"] == sel_cycle]

        st.markdown("---")

        # ── KPI cards ─────────────────────────────────────────────────────────
        k1, k2, k3 = st.columns(3)
        total_visits     = int(df_filt["ClinicVisits"].sum())       if "ClinicVisits"      in df_filt.columns else 0
        total_complaints = int(df_filt["PatientComplaints"].sum())  if "PatientComplaints" in df_filt.columns else 0
        avg_wait         = df_filt["ClinicWaitingTime"].mean()      if "ClinicWaitingTime" in df_filt.columns else np.nan

        with k1:
            st.markdown(f'''<div class="metric-card success">
                <div class="metric-label">Total Clinic Visits</div>
                <div class="metric-value">{total_visits:,}</div>
                <div class="metric-sub">all departments</div>
            </div>''', unsafe_allow_html=True)
        with k2:
            wt_class = "success" if pd.isna(avg_wait) or avg_wait < 20 else ("warning" if avg_wait < 40 else "danger")
            wt_val   = f"{avg_wait:.1f} min" if pd.notna(avg_wait) else "—"
            st.markdown(f'''<div class="metric-card {wt_class}">
                <div class="metric-label">Avg Waiting Time</div>
                <div class="metric-value">{wt_val}</div>
                <div class="metric-sub">clinic waiting time</div>
            </div>''', unsafe_allow_html=True)
        with k3:
            cmp_class = "success" if total_complaints == 0 else ("warning" if total_complaints < 20 else "danger")
            st.markdown(f'''<div class="metric-card {cmp_class}">
                <div class="metric-label">Patient Complaints</div>
                <div class="metric-value">{total_complaints:,}</div>
                <div class="metric-sub">total reported</div>
            </div>''', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Department overview ───────────────────────────────────────────────
        st.markdown('<div class="section-header">📊 Department Overview</div>', unsafe_allow_html=True)

        if "Department" in df_filt.columns:
            # Build agg spec only from columns that actually exist in the file
            agg_spec = {}
            id_col = "Aubnetid" if "Aubnetid" in df_filt.columns else (df_filt.columns[0])
            agg_spec["Physicians"] = (id_col, "nunique")
            if "Division_norm"    in df_filt.columns: agg_spec["Divisions"]        = ("Division_norm",    "nunique")
            if "ClinicVisits"     in df_filt.columns: agg_spec["Total_Visits"]     = ("ClinicVisits",     "sum")
            if "ClinicWaitingTime"in df_filt.columns: agg_spec["Avg_Wait"]         = ("ClinicWaitingTime","mean")
            if "PatientComplaints"in df_filt.columns: agg_spec["Total_Complaints"] = ("PatientComplaints","sum")
            if "PatientComplaints"in df_filt.columns: agg_spec["Avg_Complaints"]   = ("PatientComplaints","mean")

            dept_summary = (
                df_filt.groupby("Department", as_index=False)
                .agg(**agg_spec)
                .reset_index(drop=True)
            )
            # Add missing columns as zeros so downstream code doesn't break
            for c in ["Total_Visits", "Total_Complaints", "Divisions", "Avg_Wait", "Avg_Complaints"]:
                if c not in dept_summary.columns:
                    dept_summary[c] = 0
            for c in ["Total_Visits", "Total_Complaints"]:
                dept_summary[c] = dept_summary[c].fillna(0).astype(int)
            dept_summary["Avg_Wait"]       = dept_summary["Avg_Wait"].round(1)
            dept_summary["Avg_Complaints"] = dept_summary["Avg_Complaints"].round(2)
            dept_summary = dept_summary.sort_values("Total_Visits", ascending=False).reset_index(drop=True)

            dv1, dv2 = st.columns(2)
            with dv1:
                st.markdown("**Clinic Visits by Department**")
                fig, ax = plt.subplots(figsize=(7, max(4, len(dept_summary) * 0.42)))
                colours = ["#ef4444" if v == dept_summary["Total_Visits"].max()
                           else "#3b82f6" for v in dept_summary["Total_Visits"]]
                bars = ax.barh(dept_summary["Department"], dept_summary["Total_Visits"],
                               color=colours, edgecolor="white", linewidth=0.8, alpha=0.85)
                mx = dept_summary["Total_Visits"].max()
                for bar, val in zip(bars, dept_summary["Total_Visits"]):
                    ax.text(val + mx * 0.01, bar.get_y() + bar.get_height() / 2,
                            f"{val:,}", va="center", fontsize=8, fontweight="600")
                ax.set_xlabel("Total Clinic Visits", fontsize=10)
                ax.set_title("Clinic Visits by Department", fontsize=11, fontweight="bold")
                ax.grid(axis="x", alpha=0.3, linestyle="--")
                ax.set_facecolor("#fafafa")
                fig.patch.set_facecolor("white")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

            with dv2:
                st.markdown("**Patient Complaints by Department**")
                dept_cmp = dept_summary[dept_summary["Total_Complaints"] > 0].sort_values("Total_Complaints", ascending=False)
                if dept_cmp.empty:
                    st.success("No complaints recorded for this cycle.")
                else:
                    fig2, ax2 = plt.subplots(figsize=(7, max(4, len(dept_cmp) * 0.42)))
                    c2 = ["#ef4444" if v == dept_cmp["Total_Complaints"].max()
                          else "#f59e0b" for v in dept_cmp["Total_Complaints"]]
                    bars2 = ax2.barh(dept_cmp["Department"], dept_cmp["Total_Complaints"],
                                     color=c2, edgecolor="white", linewidth=0.8, alpha=0.85)
                    for bar, val in zip(bars2, dept_cmp["Total_Complaints"]):
                        ax2.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
                                 str(int(val)), va="center", fontsize=8, fontweight="600")
                    ax2.set_xlabel("Total Patient Complaints", fontsize=10)
                    ax2.set_title("Patient Complaints by Department", fontsize=11, fontweight="bold")
                    ax2.grid(axis="x", alpha=0.3, linestyle="--")
                    ax2.set_facecolor("#fafafa")
                    fig2.patch.set_facecolor("white")
                    plt.tight_layout()
                    st.pyplot(fig2, use_container_width=True)
                    plt.close()

            st.markdown("**Department Summary Table**")
            dept_display = dept_summary.copy()
            dept_display.columns = ["Department", "Physicians", "Divisions", "Total Visits",
                                     "Avg Wait (min)", "Total Complaints", "Avg Complaints/Physician"]
            st.dataframe(dept_display, use_container_width=True, hide_index=True,
                         column_config={
                             "Total Visits": st.column_config.ProgressColumn(
                                 min_value=0, max_value=int(dept_display["Total Visits"].max()), format="%d"),
                             "Total Complaints": st.column_config.ProgressColumn(
                                 min_value=0, max_value=max(1, int(dept_display["Total Complaints"].max())), format="%d"),
                         })

        st.markdown("---")

        # ── Division drill-down ───────────────────────────────────────────────
        st.markdown('<div class="section-header">🔬 Division Drill-Down</div>', unsafe_allow_html=True)

        # ── Filter + controls row ─────────────────────────────────────────────
        ddf1, ddf2, ddf3 = st.columns([1.5, 1.2, 1])
        with ddf1:
            dept_opts = ["All Departments"] + sorted(df_filt["Department"].dropna().unique().tolist()) \
                        if "Department" in df_filt.columns else ["All Departments"]
            sel_dept = st.selectbox("Filter by Department", dept_opts, key="div_dept")
        with ddf2:
            div_metric = st.selectbox(
                "Chart metric",
                ["Clinic Visits", "Avg Wait Time (min)", "Patient Complaints"],
                key="div_metric"
            )
        with ddf3:
            top_n = st.slider("Top N divisions", min_value=5, max_value=30, value=10, step=5, key="div_topn")

        df_div = df_filt if sel_dept == "All Departments" else df_filt[df_filt["Department"] == sel_dept]

        if "Division_norm" in df_div.columns:
            div_agg = {}
            id_col2 = "Aubnetid" if "Aubnetid" in df_div.columns else df_div.columns[0]
            div_agg["Physicians"] = (id_col2, "nunique")
            if "ClinicVisits"      in df_div.columns: div_agg["Total_Visits"]     = ("ClinicVisits",      "sum")
            if "ClinicWaitingTime" in df_div.columns: div_agg["Avg_Wait"]         = ("ClinicWaitingTime", "mean")
            if "PatientComplaints" in df_div.columns: div_agg["Total_Complaints"] = ("PatientComplaints", "sum")

            div_summary = (
                df_div.groupby("Division_norm", as_index=False)
                .agg(**div_agg)
                .reset_index(drop=True)
            )
            for c in ["Total_Visits", "Total_Complaints", "Avg_Wait"]:
                if c not in div_summary.columns:
                    div_summary[c] = 0
            for c in ["Total_Visits", "Total_Complaints"]:
                div_summary[c] = div_summary[c].fillna(0).astype(int)
            div_summary["Avg_Wait"] = div_summary["Avg_Wait"].round(1)

            # Map UI label → column + colour logic
            metric_col_map = {
                "Clinic Visits":          "Total_Visits",
                "Avg Wait Time (min)":    "Avg_Wait",
                "Patient Complaints":     "Total_Complaints",
            }
            metric_col   = metric_col_map[div_metric]
            sort_col_div = metric_col if metric_col in div_summary.columns else "Total_Visits"
            div_plot = div_summary.sort_values(sort_col_div, ascending=False).head(top_n).sort_values(sort_col_div, ascending=True)

            # Colour by complaints presence (consistent cue regardless of metric)
            if "Total_Complaints" in div_plot.columns:
                bar_colours_div = [
                    "#ef4444" if c >= 3 else "#f59e0b" if c >= 1 else "#3b82f6"
                    for c in div_plot["Total_Complaints"]
                ]
            else:
                bar_colours_div = ["#3b82f6"] * len(div_plot)

            fig3, ax3 = plt.subplots(figsize=(9, max(4, len(div_plot) * 0.42)))
            values = div_plot[sort_col_div]
            bars3 = ax3.barh(div_plot["Division_norm"], values,
                             color=bar_colours_div, edgecolor="white", linewidth=0.8, alpha=0.87)
            mx3 = values.max() if values.max() > 0 else 1
            for bar, val, cmp in zip(bars3, values, div_plot.get("Total_Complaints", [0]*len(div_plot))):
                fmt = f"{val:,}" if div_metric == "Clinic Visits" else (
                      f"{val:.1f}" if div_metric == "Avg Wait Time (min)" else str(int(val)))
                warn = f"  ⚠ {int(cmp)}" if cmp > 0 and div_metric != "Patient Complaints" else ""
                ax3.text(val + mx3 * 0.005, bar.get_y() + bar.get_height() / 2,
                         fmt + warn, va="center", fontsize=8,
                         color="#ef4444" if cmp > 0 else "#374151", fontweight="600")

            ax3.set_xlabel(div_metric, fontsize=10)
            ax3.set_title(
                f"Top {min(top_n, len(div_plot))} Divisions — {div_metric}"
                + (f"  [{sel_dept}]" if sel_dept != "All Departments" else ""),
                fontsize=11, fontweight="bold"
            )
            ax3.grid(axis="x", alpha=0.3, linestyle="--")
            ax3.set_facecolor("#fafafa")
            fig3.patch.set_facecolor("white")
            ax3.legend(handles=[
                mpatches.Patch(color="#3b82f6", alpha=0.87, label="No complaints"),
                mpatches.Patch(color="#f59e0b", alpha=0.87, label="1–2 complaints"),
                mpatches.Patch(color="#ef4444", alpha=0.87, label="3+ complaints"),
            ], fontsize=8, loc="lower right")
            plt.tight_layout()
            st.pyplot(fig3, use_container_width=True)
            plt.close()

            st.caption(f"Showing top {min(top_n, len(div_plot))} of {len(div_summary)} divisions · coloured by complaint level")

            st.markdown("**Division Detail Table**")
            div_display = div_summary.sort_values(sort_col_div, ascending=False).copy()
            div_display = div_display.drop(columns=["Physicians"], errors="ignore")
            div_display.columns = [
                c.replace("Division_norm", "Division")
                 .replace("Total_Visits", "Total Visits")
                 .replace("Avg_Wait", "Avg Wait (min)")
                 .replace("Total_Complaints", "Total Complaints")
                for c in div_display.columns
            ]
            st.dataframe(div_display, use_container_width=True, hide_index=True,
                         column_config={
                             "Total Visits": st.column_config.ProgressColumn(
                                 min_value=0, max_value=int(div_display["Total Visits"].max()) if "Total Visits" in div_display.columns else 100, format="%d"),
                             "Total Complaints": st.column_config.ProgressColumn(
                                 min_value=0, max_value=max(1, int(div_display["Total Complaints"].max())) if "Total Complaints" in div_display.columns else 1, format="%d"),
                         })

        st.markdown("---")

        # ── Physician-level explorer ──────────────────────────────────────────
        st.markdown('<div class="section-header">👤 Physician-Level Explorer</div>', unsafe_allow_html=True)

        pe1, pe2, pe3 = st.columns(3)
        with pe1:
            dept_opts2 = ["All"] + sorted(df_filt["Department"].dropna().unique().tolist()) \
                         if "Department" in df_filt.columns else ["All"]
            sel_dept2 = st.selectbox("Department", dept_opts2, key="pe_dept")
        df_pe = df_filt if sel_dept2 == "All" else df_filt[df_filt["Department"] == sel_dept2]
        with pe2:
            div_opts = ["All"] + sorted(df_pe["Division_norm"].dropna().unique().tolist()) \
                       if "Division_norm" in df_pe.columns else ["All"]
            sel_div = st.selectbox("Division", div_opts, key="pe_div")
        df_pe = df_pe if sel_div == "All" else df_pe[df_pe["Division_norm"] == sel_div]
        with pe3:
            sort_opts = ["Clinic Visits ↓", "Patient Complaints ↓", "Waiting Time ↓"]
            sel_sort  = st.selectbox("Sort by", sort_opts, key="pe_sort")

        sort_col_map = {
            "Clinic Visits ↓":      "ClinicVisits",
            "Patient Complaints ↓": "PatientComplaints",
            "Waiting Time ↓":       "ClinicWaitingTime",
        }
        sort_col = sort_col_map[sel_sort]
        if sort_col in df_pe.columns:
            df_pe = df_pe.sort_values(sort_col, ascending=False)

        show_cols  = ["Aubnetid", "Division_norm", "Department", "FiscalCycle",
                      "ClinicVisits", "ClinicWaitingTime", "PatientComplaints"]
        avail_show = [c for c in show_cols if c in df_pe.columns]
        show_renamed = df_pe[avail_show].rename(columns={
            "Aubnetid":          "Physician ID",
            "Division_norm":     "Division",
            "FiscalCycle":       "Cycle",
            "ClinicVisits":      "Clinic Visits",
            "ClinicWaitingTime": "Wait Time (min)",
            "PatientComplaints": "Complaints",
        }).reset_index(drop=True)

        max_visits = int(df_filt["ClinicVisits"].max())     if "ClinicVisits"      in df_filt.columns else 100
        max_cmp    = int(df_filt["PatientComplaints"].max())if "PatientComplaints" in df_filt.columns else 10
        st.dataframe(show_renamed, use_container_width=True, hide_index=True,
                     column_config={
                         "Clinic Visits": st.column_config.ProgressColumn(
                             min_value=0, max_value=max_visits, format="%d"),
                         "Complaints":    st.column_config.ProgressColumn(
                             min_value=0, max_value=max(1, max_cmp), format="%d"),
                     })

        csv_ind = show_renamed.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Export filtered table as CSV", csv_ind,
                           "division_physicians.csv", "text/csv")
