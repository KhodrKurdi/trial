import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

st.set_page_config(page_title="AUBMC Behavior Dashboard", layout="wide")

# =========================
# Helpers
# =========================
@st.cache_data
def load_csv(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

def get_question_cols(df):
    return [c for c in df.columns if c.startswith("Q1_") or c.startswith("q_")]

def ensure_overall_score(df):
    df = df.copy()
    if "overall_score" not in df.columns:
        q_cols = get_question_cols(df)
        if len(q_cols) == 0:
            raise ValueError("No overall_score and no question columns found.")
        df["overall_score"] = df[q_cols].mean(axis=1, skipna=True)
    return df

def physician_agg(df, by=["cycle"]):
    """
    Aggregates per physician (and optionally cycle) for outlier detection + reporting.
    """
    gcols = ["physician_id"] + by
    out = (df.groupby(gcols, as_index=False)
             .agg(
                 n_forms=("overall_score", "count"),
                 avg_behavior_score=("overall_score", "mean"),
                 std_behavior_score=("overall_score", "std"),
             ))
    return out

def add_iqr_outliers(df, score_col="avg_behavior_score", group_cols=["cycle"], high=False):
    """
    IQR outliers. Default LOW outliers (behavior). If high=True then flags high outliers.
    """
    df = df.copy()
    def iqr_flag(sub):
        q1 = sub[score_col].quantile(0.25)
        q3 = sub[score_col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        sub["iqr_lower"] = lower
        sub["iqr_upper"] = upper
        sub["outlier_iqr"] = sub[score_col] > upper if high else sub[score_col] < lower
        return sub
    return df.groupby(group_cols, group_keys=False).apply(iqr_flag)

def add_z_outliers(df, score_col="avg_behavior_score", group_cols=["cycle"], z_thresh=2, high=False):
    df = df.copy()
    def zflag(sub):
        mu = sub[score_col].mean()
        sd = sub[score_col].std(ddof=0)
        sub["z_score"] = (sub[score_col] - mu) / (sd if sd and sd > 0 else np.nan)
        sub["outlier_z"] = sub["z_score"] >= z_thresh if high else sub["z_score"] <= -z_thresh
        return sub
    return df.groupby(group_cols, group_keys=False).apply(zflag)

def add_bottom10(df, score_col="avg_behavior_score", group_cols=["cycle"]):
    df = df.copy()
    def b10(sub):
        thr = sub[score_col].quantile(0.10)
        sub["bottom10"] = sub[score_col] <= thr
        sub["bottom10_threshold"] = thr
        return sub
    return df.groupby(group_cols, group_keys=False).apply(b10)

def add_funnel_low(df, score_col="avg_behavior_score", n_col="n_forms", group_cols=["cycle"], z=1.96):
    """
    LOW-only funnel: lower bound = mean - z*(std/sqrt(n))
    """
    df = df.copy()
    def funnel(sub):
        mu = sub[score_col].mean()
        sd = sub[score_col].std(ddof=0)
        sub["se"] = sd / np.sqrt(sub[n_col].clip(lower=1))
        sub["lower_funnel"] = mu - z * sub["se"]
        sub["outlier_funnel_low"] = sub[score_col] < sub["lower_funnel"]
        return sub
    return df.groupby(group_cols, group_keys=False).apply(funnel)

# =========================
# Load data
# =========================
st.sidebar.header("Data Sources")

behavior_aubmc_path = st.sidebar.text_input("AUBMC behavior CSV", "AUBMC_Behavior_survey.csv")
behavior_ed_path = st.sidebar.text_input("ED behavior CSV", "ED_Behavior_survey.csv")
behavior_patho_path = st.sidebar.text_input("Pathology behavior CSV", "Patho_Behavior_survey.csv")
indicators_path = st.sidebar.text_input("Indicators CSV", "indicators_processed_2025.csv")

physmap_path = st.sidebar.text_input("Physician map CSV (recommended)", "physician_map.csv")
sent_path = st.sidebar.text_input("Sentiment by physician CSV (optional)", "sentiment_by_physician.csv")

try:
    aubmc_df = ensure_overall_score(load_csv(behavior_aubmc_path))
    ed_df = ensure_overall_score(load_csv(behavior_ed_path))
    patho_df = ensure_overall_score(load_csv(behavior_patho_path))
    ind_df = load_csv(indicators_path)

    # optional
    phys_map = None
    try:
        phys_map = load_csv(physmap_path)
    except:
        pass

    sent_df = None
    try:
        sent_df = load_csv(sent_path)
    except:
        pass

except Exception as e:
    st.error(f"Failed to load/prepare data: {e}")
    st.stop()

# Combine behavior into one master
aubmc_df["unit_group"] = aubmc_df.get("unit_group", "AUBMC_360")
ed_df["unit_group"] = ed_df.get("unit_group", "ED")
patho_df["unit_group"] = patho_df.get("unit_group", "PATHO")

behavior_all = pd.concat([aubmc_df, ed_df, patho_df], ignore_index=True)

# Apply physician mapping if available
if phys_map is not None and "physician_id" in phys_map.columns:
    behavior_all = behavior_all.merge(phys_map, on="physician_id", how="left")

# =========================
# Sidebar filters
# =========================
st.sidebar.header("Filters")

unit_options = sorted(behavior_all["unit_group"].dropna().unique().tolist())
unit_sel = st.sidebar.selectbox("Unit group", unit_options, index=0)

df = behavior_all[behavior_all["unit_group"] == unit_sel].copy()

if "cycle" in df.columns:
    cycles = sorted(df["cycle"].dropna().unique().tolist())
    cycle_sel = st.sidebar.multiselect("Cycle", cycles, default=cycles)
    df = df[df["cycle"].isin(cycle_sel)]

if "raters_group" in df.columns:
    rg = sorted(df["raters_group"].dropna().unique().tolist())
    default_rg = [x for x in rg if x != "Faculty Self-Evaluation"] or rg
    rg_sel = st.sidebar.multiselect("Raters group", rg, default=default_rg)
    df = df[df["raters_group"].isin(rg_sel)]

min_n = st.sidebar.slider("Min # evaluations per physician", 1, 30, 5)

# =========================
# Build physician table + outliers
# =========================
by = ["cycle"] if "cycle" in df.columns else []
phys = physician_agg(df, by=by)
phys = phys[phys["n_forms"] >= min_n].copy()

if "cycle" not in phys.columns:
    phys["cycle"] = "All"

phys = add_funnel_low(phys, group_cols=["cycle"])
phys = add_iqr_outliers(phys, group_cols=["cycle"], high=False)
phys = add_z_outliers(phys, group_cols=["cycle"], z_thresh=2, high=False)
phys = add_bottom10(phys, group_cols=["cycle"])

# =========================
# UI: Tabs
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Executive Overview", "Department/Unit Drilldown", "Physician Profile", "Sentiment Explorer", "Indicators (2024–2025)"
])

# ---------- Tab 1: Executive ----------
with tab1:
    st.subheader("Executive Overview")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("# Physicians", f"{phys['physician_id'].nunique():,}")
    c2.metric("# Evaluations", f"{int(phys['n_forms'].sum()):,}")
    c3.metric("Avg score", f"{phys['avg_behavior_score'].mean():.3f}")
    c4.metric("Funnel low outliers", f"{int(phys['outlier_funnel_low'].sum()):,}")
    c5.metric("Bottom 10%", f"{int(phys['bottom10'].sum()):,}")

    # Trend
    if "cycle" in df.columns:
        trend = (df.groupby(["cycle"], as_index=False)
                   .agg(avg_score=("overall_score", "mean"),
                        n=("overall_score", "count")))
        fig = px.line(trend, x="cycle", y="avg_score", markers=True, title="Average Behavior Score by Cycle")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Flagged physicians (union of methods)")
    phys["flag_any"] = phys["outlier_funnel_low"] | phys["outlier_iqr"] | phys["outlier_z"] | phys["bottom10"]
    flagged = phys[phys["flag_any"]].sort_values(["cycle", "avg_behavior_score"]).copy()
    st.dataframe(flagged, use_container_width=True)

# ---------- Tab 2: Drilldown ----------
with tab2:
    st.subheader("Distribution & Outliers")

    # Boxplot by cycle
    fig = px.box(phys, x="cycle", y="avg_behavior_score", points="outliers", title="Physician Avg Score (Boxplot / IQR)")
    st.plotly_chart(fig, use_container_width=True)

    # Funnel plot (scatter + lower bound)
    fig2 = px.scatter(
        phys, x="n_forms", y="avg_behavior_score", color="outlier_funnel_low",
        hover_data=["physician_id", "cycle"],
        title="Funnel Plot (Low outliers)"
    )
    fig2.add_scatter(x=phys["n_forms"], y=phys["lower_funnel"], mode="lines", name="Lower funnel (95%)")
    st.plotly_chart(fig2, use_container_width=True)

# ---------- Tab 3: Physician Profile ----------
with tab3:
    st.subheader("Physician Profile")

    pid = st.selectbox("Select physician_id", sorted(phys["physician_id"].unique().tolist()))
    p = phys[phys["physician_id"] == pid].copy()

    st.write("Flags:", {
        "funnel_low": bool(p["outlier_funnel_low"].iloc[0]),
        "iqr_low": bool(p["outlier_iqr"].iloc[0]),
        "z_low": bool(p["outlier_z"].iloc[0]),
        "bottom10": bool(p["bottom10"].iloc[0]),
    })

    # Trend per cycle if available
    if "cycle" in df.columns:
        p_trend = (df[df["physician_id"] == pid]
                   .groupby("cycle", as_index=False)
                   .agg(avg=("overall_score", "mean"), n=("overall_score", "count")))
        fig = px.line(p_trend, x="cycle", y="avg", markers=True, title="Physician Avg Score by Cycle")
        st.plotly_chart(fig, use_container_width=True)

    # Comments view (optional)
    if "comments" in df.columns:
        st.markdown("### Sample comments")
        comments_sample = df[(df["physician_id"] == pid) & (df["comments"].notna()) & (df["comments"].astype(str).str.strip() != "")]
        st.dataframe(comments_sample[["cycle","raters_group","fillout_date","comments"]].head(50), use_container_width=True)

# ---------- Tab 4: Sentiment ----------
with tab4:
    st.subheader("Sentiment Explorer")
    if sent_df is None:
        st.info("Provide sentiment_by_physician.csv to enable this tab (recommended to precompute).")
    else:
        s = sent_df.copy()
        # allow filtering by unit/cycle if columns exist
        if "unit_group" in s.columns:
            s = s[s["unit_group"] == unit_sel]
        if "cycle" in s.columns and "cycle" in phys.columns:
            s = s[s["cycle"].isin(phys["cycle"].unique())]

        fig = px.histogram(s, x="negative_ratio", nbins=15, title="Negative Ratio Distribution")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Negative sentiment outliers")
        st.dataframe(s[s.get("negative_outlier", False) == True].sort_values("negative_ratio", ascending=False),
                     use_container_width=True)

# ---------- Tab 5: Indicators ----------
with tab5:
    st.subheader("Indicators (2024–2025)")

    # Filter indicators to cycle 2024–2025 by default
    ind = ind_df.copy()
    if "FiscalCycle" in ind.columns:
        ind = ind[ind["FiscalCycle"] == "Cycle 2024-2025"].copy()

    # Map Aubnetid -> physician_id (assume same)
    ind["physician_id"] = ind["Aubnetid"]

    # Join flags from phys (use cycle=All or current selection; simplest: latest/All)
    phys_latest = phys.copy()
    # If multiple cycles selected, you can pick one; for now, keep as-is
    merged = ind.merge(phys_latest[["physician_id","avg_behavior_score","outlier_funnel_low","bottom10"]], on="physician_id", how="left")

    # Simple complaints outliers (IQR high)
    merged["PatientComplaints"] = pd.to_numeric(merged.get("PatientComplaints"), errors="coerce")
    q1 = merged["PatientComplaints"].quantile(0.25)
    q3 = merged["PatientComplaints"].quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    merged["complaints_outlier_high"] = merged["PatientComplaints"] > upper

    c1, c2, c3 = st.columns(3)
    c1.metric("Complaints high outliers", f"{int(merged['complaints_outlier_high'].sum()):,}")
    c2.metric("Behavior funnel low outliers (linked)", f"{int(merged['outlier_funnel_low'].fillna(False).sum()):,}")
    c3.metric("Bottom 10% (linked)", f"{int(merged['bottom10'].fillna(False).sum()):,}")

    fig = px.box(merged, y="PatientComplaints", points="outliers", title="Patient Complaints (IQR Outliers)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Linked table (Indicators + Flags)")
    st.dataframe(merged, use_container_width=True)
