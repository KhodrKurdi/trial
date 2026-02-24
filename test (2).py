import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="AUBMC Behavior Dashboard", layout="wide")

# =========================
# Helpers
# =========================
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

def get_question_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("Q1_") or c.startswith("q_")]

def ensure_required_cols(df: pd.DataFrame, unit_group_default: str) -> pd.DataFrame:
    """
    Ensures minimal expected columns exist for the app to work.
    Does not force cycle; cycle is optional.
    """
    df = df.copy()

    # If unit_group missing, create it
    if "unit_group" not in df.columns:
        df["unit_group"] = unit_group_default

    # If comments exists but not string, cast safe
    if "comments" in df.columns:
        df["comments"] = df["comments"].astype(str)

    return df

def ensure_overall_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute overall_score if missing using question columns (Q1_/q_).
    """
    df = df.copy()
    if "overall_score" not in df.columns:
        q_cols = get_question_cols(df)
        if len(q_cols) == 0:
            raise ValueError("Missing overall_score and no Q1_/q_ columns found to compute it.")
        df["overall_score"] = df[q_cols].mean(axis=1, skipna=True)
    df["overall_score"] = pd.to_numeric(df["overall_score"], errors="coerce")
    return df

def physician_agg(df: pd.DataFrame, by_cols: list[str]) -> pd.DataFrame:
    """
    Physician-level aggregation for outlier detection + reporting.
    """
    gcols = ["physician_id"] + by_cols
    out = (
        df.groupby(gcols, as_index=False)
          .agg(
              n_forms=("overall_score", "count"),
              avg_behavior_score=("overall_score", "mean"),
              std_behavior_score=("overall_score", "std"),
          )
    )
    return out

def add_iqr_outliers_low(df: pd.DataFrame, score_col="avg_behavior_score", group_cols=None) -> pd.DataFrame:
    if group_cols is None:
        group_cols = ["cycle_key"]
    df = df.copy()

    def iqr_flag(sub):
        q1 = sub[score_col].quantile(0.25)
        q3 = sub[score_col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        sub["iqr_lower"] = lower
        sub["iqr_upper"] = upper
        sub["outlier_iqr_low"] = sub[score_col] < lower
        return sub

    return df.groupby(group_cols, group_keys=False).apply(iqr_flag)

def add_z_outliers_low(df: pd.DataFrame, score_col="avg_behavior_score", group_cols=None, z_thresh=2.0) -> pd.DataFrame:
    if group_cols is None:
        group_cols = ["cycle_key"]
    df = df.copy()

    def zflag(sub):
        mu = sub[score_col].mean()
        sd = sub[score_col].std(ddof=0)
        sub["z_score"] = (sub[score_col] - mu) / (sd if sd and sd > 0 else np.nan)
        sub["outlier_z_low"] = sub["z_score"] <= -z_thresh
        return sub

    return df.groupby(group_cols, group_keys=False).apply(zflag)

def add_bottom10(df: pd.DataFrame, score_col="avg_behavior_score", group_cols=None) -> pd.DataFrame:
    if group_cols is None:
        group_cols = ["cycle_key"]
    df = df.copy()

    def b10(sub):
        thr = sub[score_col].quantile(0.10)
        sub["bottom10"] = sub[score_col] <= thr
        sub["bottom10_threshold"] = thr
        return sub

    return df.groupby(group_cols, group_keys=False).apply(b10)

def add_funnel_low(df: pd.DataFrame, score_col="avg_behavior_score", n_col="n_forms", group_cols=None, z=1.96) -> pd.DataFrame:
    if group_cols is None:
        group_cols = ["cycle_key"]
    df = df.copy()

    def funnel(sub):
        mu = sub[score_col].mean()
        sd = sub[score_col].std(ddof=0)
        sub["se"] = sd / np.sqrt(sub[n_col].clip(lower=1))
        sub["lower_funnel"] = mu - z * sub["se"]
        sub["outlier_funnel_low"] = sub[score_col] < sub["lower_funnel"]
        return sub

    return df.groupby(group_cols, group_keys=False).apply(funnel)

def safe_bool_sum(s):
    return int(pd.Series(s).fillna(False).astype(bool).sum())

# =========================
# Sidebar: data sources
# =========================
st.sidebar.header("Data Sources")

behavior_aubmc_path = st.sidebar.text_input("AUBMC behavior CSV", "AUBMC_Behavior_survey.csv")
behavior_ed_path = st.sidebar.text_input("ED behavior CSV", "ED_Behavior_survey.csv")
behavior_patho_path = st.sidebar.text_input("Pathology behavior CSV", "Patho_Behavior_survey.csv")

indicators_path = st.sidebar.text_input("Indicators CSV", "indicators_processed_2025.csv")

sent_path = st.sidebar.text_input("Sentiment by physician CSV (optional)", "sentiment_by_physician.csv")

# =========================
# Load data
# =========================
try:
    aubmc_df = load_csv(behavior_aubmc_path)
    ed_df = load_csv(behavior_ed_path)
    patho_df = load_csv(behavior_patho_path)

    aubmc_df = ensure_required_cols(ensure_overall_score(aubmc_df), "AUBMC_360")
    ed_df = ensure_required_cols(ensure_overall_score(ed_df), "ED")
    patho_df = ensure_required_cols(ensure_overall_score(patho_df), "PATHO")

    ind_df = load_csv(indicators_path)

    sent_df = None
    try:
        sent_df = load_csv(sent_path)
    except Exception:
        sent_df = None

except Exception as e:
    st.error(f"Failed to load/prepare data: {e}")
    st.stop()

# Combine
behavior_all = pd.concat([aubmc_df, ed_df, patho_df], ignore_index=True)

# =========================
# Sidebar: filters
# =========================
st.sidebar.header("Filters")

unit_options = sorted(behavior_all["unit_group"].dropna().unique().tolist())
unit_sel = st.sidebar.selectbox("Unit group", unit_options, index=0)

df = behavior_all[behavior_all["unit_group"] == unit_sel].copy()

# cycle is optional; we use a safe key so code doesn't break
if "cycle" in df.columns:
    df["cycle_key"] = df["cycle"].astype(str)
    cycles = sorted(df["cycle_key"].dropna().unique().tolist())
    cycle_sel = st.sidebar.multiselect("Cycle", cycles, default=cycles)
    df = df[df["cycle_key"].isin(cycle_sel)]
else:
    df["cycle_key"] = "All"

# raters group filter (optional)
if "raters_group" in df.columns:
    rg = sorted(df["raters_group"].dropna().unique().tolist())
    default_rg = [x for x in rg if x != "Faculty Self-Evaluation"] or rg
    rg_sel = st.sidebar.multiselect("Raters group", rg, default=default_rg)
    df = df[df["raters_group"].isin(rg_sel)]

min_n = st.sidebar.slider("Min # evaluations per physician", 1, 30, 5)

# =========================
# Build physician table + outliers
# =========================
by_cols = ["cycle_key"]
phys = physician_agg(df, by_cols=by_cols)
phys = phys[phys["n_forms"] >= min_n].copy()

# Outliers
phys = add_funnel_low(phys, group_cols=["cycle_key"])
phys = add_iqr_outliers_low(phys, group_cols=["cycle_key"])
phys = add_z_outliers_low(phys, group_cols=["cycle_key"], z_thresh=2.0)
phys = add_bottom10(phys, group_cols=["cycle_key"])

phys["flag_any"] = (
    phys["outlier_funnel_low"].fillna(False)
    | phys["outlier_iqr_low"].fillna(False)
    | phys["outlier_z_low"].fillna(False)
    | phys["bottom10"].fillna(False)
)

# =========================
# UI Tabs
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Executive Overview",
    "Drilldown (Outliers)",
    "Physician Profile",
    "Sentiment Explorer",
    "Indicators (2024–2025)"
])

# ---------- Debug expander ----------
with st.expander("Debug: Loaded columns & sample rows"):
    st.write("Behavior columns:", behavior_all.columns.tolist())
    st.write("Behavior sample:", behavior_all.head())
    st.write("Indicators columns:", ind_df.columns.tolist())
    st.write("Indicators sample:", ind_df.head())
    if sent_df is not None:
        st.write("Sentiment columns:", sent_df.columns.tolist())
        st.write("Sentiment sample:", sent_df.head())
    else:
        st.write("Sentiment file not loaded (optional).")

# =========================
# Tab 1: Executive
# =========================
with tab1:
    st.subheader("Executive Overview")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("# Physicians", f"{phys['physician_id'].nunique():,}")
    c2.metric("# Evaluations", f"{int(phys['n_forms'].sum()):,}")
    c3.metric("Avg behavior score", f"{phys['avg_behavior_score'].mean():.3f}")
    c4.metric("Funnel LOW outliers", f"{safe_bool_sum(phys['outlier_funnel_low']):,}")
    c5.metric("Bottom 10%", f"{safe_bool_sum(phys['bottom10']):,}")

    # Trend (only if cycle exists in original df)
    if "cycle" in df.columns:
        trend = (
            df.groupby("cycle_key", as_index=False)
              .agg(avg_score=("overall_score", "mean"), n=("overall_score", "count"))
        )
        fig = px.line(trend, x="cycle_key", y="avg_score", markers=True, title="Average Behavior Score by Cycle")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Flagged physicians (any method)")
    flagged = phys[phys["flag_any"]].sort_values(["cycle_key", "avg_behavior_score"]).copy()
    st.dataframe(flagged, use_container_width=True)

# =========================
# Tab 2: Drilldown
# =========================
with tab2:
    st.subheader("Distribution & Outliers")

    fig_box = px.box(
        phys, x="cycle_key", y="avg_behavior_score", points="outliers",
        title="Physician Avg Score (Boxplot / IQR Low Outliers)"
    )
    st.plotly_chart(fig_box, use_container_width=True)

    # Funnel plot
    phys_sorted = phys.sort_values("n_forms")
    fig_funnel = px.scatter(
        phys, x="n_forms", y="avg_behavior_score", color="outlier_funnel_low",
        hover_data=["physician_id", "cycle_key"],
        title="Funnel Plot (LOW outliers)"
    )
    fig_funnel.add_scatter(
        x=phys_sorted["n_forms"], y=phys_sorted["lower_funnel"],
        mode="lines", name="Lower funnel (95%)"
    )
    st.plotly_chart(fig_funnel, use_container_width=True)

# =========================
# Tab 3: Physician Profile
# =========================
with tab3:
    st.subheader("Physician Profile")

    pid = st.selectbox("Select physician_id", sorted(phys["physician_id"].unique().tolist()))
    p = phys[phys["physician_id"] == pid].copy()

    flags = {
        "funnel_low": bool(p["outlier_funnel_low"].iloc[0]),
        "iqr_low": bool(p["outlier_iqr_low"].iloc[0]),
        "z_low": bool(p["outlier_z_low"].iloc[0]),
        "bottom10": bool(p["bottom10"].iloc[0]),
    }
    st.write("Flags:", flags)

    if "cycle" in df.columns:
        p_trend = (
            df[df["physician_id"] == pid]
            .groupby("cycle_key", as_index=False)
            .agg(avg=("overall_score", "mean"), n=("overall_score", "count"))
        )
        fig = px.line(p_trend, x="cycle_key", y="avg", markers=True, title="Physician Avg Score by Cycle")
        st.plotly_chart(fig, use_container_width=True)

    if "comments" in df.columns:
        st.markdown("### Sample comments (filtered)")
        comments_sample = df[
            (df["physician_id"] == pid)
            & (df["comments"].notna())
            & (df["comments"].astype(str).str.strip() != "")
        ]
        show_cols = [c for c in ["cycle_key", "raters_group", "fillout_date", "comments"] if c in comments_sample.columns]
        st.dataframe(comments_sample[show_cols].head(50), use_container_width=True)

# =========================
# Tab 4: Sentiment
# =========================
with tab4:
    st.subheader("Sentiment Explorer")

    if sent_df is None:
        st.info("sentiment_by_physician.csv not loaded. (Optional) Generate it in notebook, then provide the file.")
    else:
        s = sent_df.copy()

        # filter by unit_group if present
        if "unit_group" in s.columns:
            s = s[s["unit_group"] == unit_sel]

        # filter by cycle if present
        if "cycle" in s.columns and "cycle_key" in df.columns:
            # if your sentiment file has cycle values like 2023/2024/2025
            # and df has same, filter by selected cycles
            s["cycle"] = s["cycle"].astype(str)
            selected_cycles = df["cycle_key"].unique().tolist()
            s = s[s["cycle"].isin(selected_cycles)]

        if "negative_ratio" not in s.columns:
            st.error("Sentiment file must contain column 'negative_ratio'.")
        else:
            fig = px.histogram(s, x="negative_ratio", nbins=15, title="Negative Ratio Distribution")
            st.plotly_chart(fig, use_container_width=True)

            if "negative_outlier" in s.columns:
                st.markdown("### Negative sentiment outliers")
                out = s[s["negative_outlier"].astype(str).str.lower().isin(["true", "1", "yes"])] if s["negative_outlier"].dtype != bool else s[s["negative_outlier"]]
                st.dataframe(out.sort_values("negative_ratio", ascending=False), use_container_width=True)
            else:
                st.info("No 'negative_outlier' column found. Showing summary only.")
                st.dataframe(s.sort_values("negative_ratio", ascending=False).head(50), use_container_width=True)

# =========================
# Tab 5: Indicators
# =========================
with tab5:
    st.subheader("Indicators (2024–2025)")

    ind = ind_df.copy()
    ind.columns = ind.columns.str.strip()

    # Keep cycle 2024-2025 if column exists
    if "FiscalCycle" in ind.columns:
        ind["FiscalCycle"] = ind["FiscalCycle"].astype(str).str.strip()
        ind = ind[ind["FiscalCycle"] == "Cycle 2024-2025"].copy()

    # Map Aubnetid -> physician_id (assumes same ID)
    if "Aubnetid" in ind.columns:
        ind["physician_id"] = ind["Aubnetid"]
    else:
        st.error("Indicators file must contain 'Aubnetid' column.")
        st.stop()

    # Merge behavior flags for currently selected unit/cycle_key
    merge_cols = ["physician_id", "cycle_key", "avg_behavior_score", "outlier_funnel_low", "bottom10"]
    merged = ind.merge(phys[merge_cols], on="physician_id", how="left")

    # Complaints outliers (IQR high) if PatientComplaints exists
    if "PatientComplaints" in merged.columns:
        merged["PatientComplaints"] = pd.to_numeric(merged["PatientComplaints"], errors="coerce")
        q1 = merged["PatientComplaints"].quantile(0.25)
        q3 = merged["PatientComplaints"].quantile(0.75)
        iqr = q3 - q1
        upper = q3 + 1.5 * iqr
        merged["complaints_outlier_high"] = merged["PatientComplaints"] > upper

        c1, c2, c3 = st.columns(3)
        c1.metric("Complaints high outliers", f"{safe_bool_sum(merged['complaints_outlier_high']):,}")
        c2.metric("Behavior funnel low outliers (linked)", f"{safe_bool_sum(merged['outlier_funnel_low']):,}")
        c3.metric("Bottom 10% (linked)", f"{safe_bool_sum(merged['bottom10']):,}")

        fig = px.box(merged, y="PatientComplaints", points="outliers", title="Patient Complaints (IQR High Outliers)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No 'PatientComplaints' column found in indicators file.")

    st.markdown("### Linked table (Indicators + Flags)")
    st.dataframe(merged, use_container_width=True)
