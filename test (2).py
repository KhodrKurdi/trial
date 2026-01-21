import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob

# ============================================================
# Page configuration
# ============================================================
st.set_page_config(
    page_title="AUBMC Behavior Survey Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stButton>button {
        width: 100%;
        height: 60px;
        font-size: 1.2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Session state init (DO THIS EARLY)
# ============================================================
if "mode" not in st.session_state:
    st.session_state.mode = None

if "show_comments" not in st.session_state:
    st.session_state.show_comments = None

# ============================================================
# Sidebar: Data upload
# ============================================================
with st.sidebar:
    st.header("📂 Data Source")
    st.caption("Upload CSV/ZIP/Parquet (Cloud uploads may fail >25MB).")

    uploaded_file = st.file_uploader(
        "Upload survey data",
        type=["csv", "zip", "parquet"]
    )

    st.markdown("---")
    st.header("ℹ️ Notes")
    st.write("If you’re on Streamlit Cloud and your file is 75MB, convert it to **Parquet** to shrink it.")

# ============================================================
# Load data
# ============================================================
@st.cache_data
def load_data(uploaded):
    """
    Load from uploaded file if provided, else fallback to local CSV.
    Supports: CSV, ZIP (containing CSV), Parquet
    """
    if uploaded is not None:
        name = uploaded.name.lower()

        if name.endswith(".parquet"):
            df = pd.read_parquet(uploaded)

        elif name.endswith(".zip"):
            # pandas can read zipped CSV directly
            df = pd.read_csv(uploaded)

        else:
            df = pd.read_csv(uploaded)

    else:
        # fallback (if you included it in repo locally)
        df = pd.read_csv("All_Departments_Long_Numeric.csv")

    # ---- Cleaning steps ----
    df["Fillout Date (mm/dd/yy)"] = pd.to_datetime(df["Fillout Date (mm/dd/yy)"], errors="coerce")
    df["Year"] = df["Fillout Date (mm/dd/yy)"].dt.year

    # Combine comments columns safely (in case one column doesn't exist)
    if "Q2_Comments" not in df.columns:
        df["Q2_Comments"] = np.nan
    if "Q2_Comments\n" not in df.columns:
        df["Q2_Comments\n"] = np.nan

    df["Comments_Combined"] = df["Q2_Comments"].fillna("") + " " + df["Q2_Comments\n"].fillna("")
    df["Comments_Combined"] = df["Comments_Combined"].str.strip()
    df["Comments_Combined"] = df["Comments_Combined"].replace("", np.nan)

    return df

# ============================================================
# Sentiment analysis
# ============================================================
@st.cache_data
def analyze_sentiment(text):
    if pd.isna(text) or str(text).strip() == "":
        return "Neutral", 0.0
    try:
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        if polarity > 0.1:
            return "Positive", polarity
        elif polarity < -0.1:
            return "Negative", polarity
        else:
            return "Neutral", polarity
    except Exception:
        return "Neutral", 0.0

# ============================================================
# Try loading data safely
# ============================================================
try:
    df = load_data(uploaded_file)
except Exception as e:
    st.error("❌ Could not load the data file.")
    st.code(str(e))
    st.stop()

# If no file uploaded and fallback file missing, you'll never reach here (it would error above)
if df is None or len(df) == 0:
    st.info("👈 Please upload a dataset from the sidebar to start.")
    st.stop()

# Add sentiment analysis
df["Sentiment"], df["Sentiment_Score"] = zip(*df["Comments_Combined"].apply(analyze_sentiment))

# ============================================================
# Main title
# ============================================================
st.markdown('<div class="main-header">📊 AUBMC Behavior Survey Dashboard</div>', unsafe_allow_html=True)
st.markdown("---")

# ============================================================
# Mode selection
# ============================================================
col1, col2 = st.columns(2)

with col1:
    if st.button("👤 Individual Physician Report", use_container_width=True):
        st.session_state.mode = "individual"

with col2:
    if st.button("🏥 Departmental Report", use_container_width=True):
        st.session_state.mode = "departmental"

st.markdown("---")

# ============================================================================
# INDIVIDUAL PHYSICIAN REPORT
# ============================================================================
if st.session_state.mode == "individual":
    st.header("👤 Individual Physician Report")

    with st.sidebar:
        st.header("🔍 Filters")

        physicians = sorted(df["Subject ID"].dropna().unique())
        selected_physician = st.selectbox("Select Physician", physicians)

        years = sorted(df["Year"].dropna().unique())
        selected_years = st.multiselect("Select Year(s)", years, default=years)

        sources = sorted(df["Source"].dropna().unique())
        selected_source = st.multiselect("Select Department(s)", sources, default=sources)

    physician_data = df[
        (df["Subject ID"] == selected_physician) &
        (df["Year"].isin(selected_years)) &
        (df["Source"].isin(selected_source))
    ]

    if len(physician_data) == 0:
        st.warning("No data available for selected filters.")
    else:
        st.subheader("📈 Overview")
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            overall_avg = physician_data["Response_Numeric"].mean()
            st.metric("Overall Average Score", f"{overall_avg:.2f}/5.00")
        with c2:
            st.metric("Total Responses", f"{len(physician_data):,}")
        with c3:
            st.metric("Unique Evaluators", f"{physician_data['Rater Name'].nunique():,}")
        with c4:
            st.metric("Score Std Dev", f"{physician_data['Response_Numeric'].std():.2f}")

        st.markdown("---")

        st.subheader("📊 Average Score by Year")
        yearly_avg = physician_data.groupby("Year")["Response_Numeric"].agg(["mean", "count"]).reset_index()
        yearly_avg.columns = ["Year", "Average_Score", "Count"]

        fig_yearly = go.Figure()
        fig_yearly.add_trace(go.Bar(
            x=yearly_avg["Year"],
            y=yearly_avg["Average_Score"],
            text=yearly_avg["Average_Score"].round(2),
            textposition="outside",
            hovertemplate="<b>Year %{x}</b><br>Average: %{y:.2f}<br>Responses: %{customdata}<extra></extra>",
            customdata=yearly_avg["Count"]
        ))
        fig_yearly.update_layout(
            xaxis_title="Year",
            yaxis_title="Average Score",
            yaxis_range=[0, 5],
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_yearly, use_container_width=True)

        st.dataframe(
            yearly_avg.style.format({"Average_Score": "{:.2f}", "Count": "{:,}"}),
            use_container_width=True
        )

        st.markdown("---")

        st.subheader("💬 Comment Sentiment Analysis")
        comments_data = physician_data[physician_data["Comments_Combined"].notna()].copy()

        if len(comments_data) == 0:
            st.info("No comments available for selected filters.")
        else:
            c1, c2, c3 = st.columns(3)
            sentiment_counts = comments_data["Sentiment"].value_counts()

            with c1:
                positive_count = sentiment_counts.get("Positive", 0)
                if st.button(f"✅ Positive: {positive_count}", key="positive"):
                    st.session_state.show_comments = "Positive"
            with c2:
                neutral_count = sentiment_counts.get("Neutral", 0)
                if st.button(f"➖ Neutral: {neutral_count}", key="neutral"):
                    st.session_state.show_comments = "Neutral"
            with c3:
                negative_count = sentiment_counts.get("Negative", 0)
                if st.button(f"❌ Negative: {negative_count}", key="negative"):
                    st.session_state.show_comments = "Negative"

            if st.session_state.show_comments is not None:
                sentiment_filter = st.session_state.show_comments
                filtered_comments = comments_data[comments_data["Sentiment"] == sentiment_filter]

                st.markdown(f"### {sentiment_filter} Comments ({len(filtered_comments)})")

                if len(filtered_comments) > 0:
                    display_df = filtered_comments[["Year", "Raters Group", "Comments_Combined", "Sentiment_Score"]].copy()
                    display_df.columns = ["Year", "Rater Group", "Comment", "Sentiment Score"]
                    display_df = display_df.sort_values("Year", ascending=False)

                    st.dataframe(
                        display_df.style.format({"Sentiment Score": "{:.2f}"}),
                        use_container_width=True,
                        height=400
                    )

                    csv = display_df.to_csv(index=False)
                    st.download_button(
                        label=f"
