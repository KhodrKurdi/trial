import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AUBMC Physician Performance Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Load data function
@st.cache_data
def load_data():
    """Load and merge behavior survey data with physician indicators."""
    
    # Load behavior survey statistics (from your analysis)
    try:
        doctor_stats = pd.read_csv('Doctor_Statistics_2025 (1).csv')
        
        # Parse years from behavior survey if available
        # Assuming you have year information in the data
        doctor_stats['Survey_Year'] = 2025  # Adjust based on your data
        
    except FileNotFoundError:
        st.error("❌ Doctor_Statistics_2025.csv not found. Please run the analysis notebook first.")
        return None, None
    
    # Load physician indicators
    try:
        indicators = pd.read_csv('Physicians_Indicators_Anonymized.csv')
        
        # Clean fiscal cycle to extract year
        indicators['Year'] = indicators['FiscalCycle'].str.extract(r'(\d{4})-\d{4}').astype(float)
        
        # Parse contract date
        indicators['ContractEffectiveDate'] = pd.to_datetime(indicators['ContractEffectiveDate'], errors='coerce')
        indicators['Years_of_Service'] = (datetime.now() - indicators['ContractEffectiveDate']).dt.days / 365.25
        
    except FileNotFoundError:
        st.error("❌ Physicians_Indicators_Anonymized.csv not found. Please upload the file.")
        return None, None
    
    # Merge datasets on physician ID
    # Survey uses 'Subject ID', Indicators uses 'Aubnetid'
    merged = doctor_stats.merge(
        indicators, 
        left_on='Subject ID', 
        right_on='Aubnetid', 
        how='left'
    )
    
    return doctor_stats, indicators, merged

# Load data
with st.spinner('Loading data...'):
    doctor_stats, indicators, merged = load_data()

if doctor_stats is None or indicators is None:
    st.stop()

# Main header
st.markdown('<div class="main-header">🏥 AUBMC Physician Performance Dashboard</div>', unsafe_allow_html=True)

# Sidebar - Mode selection
st.sidebar.header("📊 Dashboard Mode")
mode = st.sidebar.radio(
    "Select View:",
    ["🏠 Overview", "👤 Individual Physician", "🏥 Departmental Analysis"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

# ============================================================================
# OVERVIEW MODE
# ============================================================================
if mode == "🏠 Overview":
    st.header("📊 System Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Total Physicians</div>
            <div class="metric-value">{}</div>
        </div>
        """.format(len(doctor_stats)), unsafe_allow_html=True)
    
    with col2:
        avg_score = doctor_stats['Avg_Score'].mean()
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Average Behavior Score</div>
            <div class="metric-value">{:.2f}/5.0</div>
        </div>
        """.format(avg_score), unsafe_allow_html=True)
    
    with col3:
        total_evaluations = doctor_stats['Num_Evaluations'].sum()
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Total Evaluations</div>
            <div class="metric-value">{:,}</div>
        </div>
        """.format(total_evaluations), unsafe_allow_html=True)
    
    with col4:
        if 'Department' in indicators.columns:
            num_depts = indicators['Department'].nunique()
        else:
            num_depts = doctor_stats.get('Source', pd.Series()).nunique()
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Departments</div>
            <div class="metric-value">{}</div>
        </div>
        """.format(num_depts), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Distribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Score Distribution")
        fig = px.histogram(
            doctor_stats, 
            x='Avg_Score',
            nbins=30,
            title="Distribution of Physician Behavior Scores",
            labels={'Avg_Score': 'Average Score', 'count': 'Number of Physicians'},
            color_discrete_sequence=['#667eea']
        )
        fig.add_vline(x=avg_score, line_dash="dash", line_color="red", 
                     annotation_text=f"Mean: {avg_score:.2f}")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💬 Comment Sentiment")
        if 'Sentiment_Negative' in doctor_stats.columns:
            sentiment_data = pd.DataFrame({
                'Sentiment': ['Positive', 'Neutral', 'Negative'],
                'Count': [
                    doctor_stats.get('Sentiment_Positive', pd.Series([0])).sum(),
                    doctor_stats.get('Sentiment_Neutral', pd.Series([0])).sum(),
                    doctor_stats.get('Sentiment_Negative', pd.Series([0])).sum()
                ]
            })
            fig = px.pie(
                sentiment_data, 
                values='Count', 
                names='Sentiment',
                title="Overall Sentiment Distribution",
                color='Sentiment',
                color_discrete_map={'Positive': '#28a745', 'Neutral': '#ffc107', 'Negative': '#dc3545'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Top and bottom performers
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌟 Top 10 Performers")
        top_10 = doctor_stats.nlargest(10, 'Avg_Score')[['Subject ID', 'Avg_Score', 'Num_Evaluations']]
        top_10['Rank'] = range(1, 11)
        top_10 = top_10[['Rank', 'Subject ID', 'Avg_Score', 'Num_Evaluations']]
        st.dataframe(
            top_10.style.format({'Avg_Score': '{:.3f}', 'Num_Evaluations': '{:,.0f}'}),
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        st.subheader("⚠️ Bottom 10 Performers")
        bottom_10 = doctor_stats.nsmallest(10, 'Avg_Score')[['Subject ID', 'Avg_Score', 'Num_Evaluations']]
        bottom_10['Rank'] = range(1, 11)
        bottom_10 = bottom_10[['Rank', 'Subject ID', 'Avg_Score', 'Num_Evaluations']]
        st.dataframe(
            bottom_10.style.format({'Avg_Score': '{:.3f}', 'Num_Evaluations': '{:,.0f}'}),
            use_container_width=True,
            hide_index=True
        )

# ============================================================================
# INDIVIDUAL PHYSICIAN MODE
# ============================================================================
elif mode == "👤 Individual Physician":
    st.header("👤 Individual Physician Report")
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Physician selection
    physicians = sorted(doctor_stats['Subject ID'].unique())
    selected_physician = st.sidebar.selectbox("Select Physician", physicians)
    
    # Year filter if available
    if 'Year' in indicators.columns:
        available_years = sorted(indicators['Year'].dropna().unique())
        if len(available_years) > 0:
            selected_years = st.sidebar.multiselect(
                "Select Year(s) for Indicators",
                available_years,
                default=available_years
            )
        else:
            selected_years = []
    else:
        selected_years = []
    
    # Get physician data
    phys_behavior = doctor_stats[doctor_stats['Subject ID'] == selected_physician].iloc[0]
    
    if 'Year' in indicators.columns and len(selected_years) > 0:
        phys_indicators = indicators[
            (indicators['Aubnetid'] == selected_physician) &
            (indicators['Year'].isin(selected_years))
        ]
    else:
        phys_indicators = indicators[indicators['Aubnetid'] == selected_physician]
    
    # Overview cards
    st.subheader("📋 Performance Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        score = phys_behavior['Avg_Score']
        color = "#28a745" if score >= 4.5 else "#ffc107" if score >= 4.0 else "#dc3545"
        st.markdown(f"""
        <div class="metric-card" style="background: {color};">
            <div class="metric-label">Behavior Score</div>
            <div class="metric-value">{score:.2f}</div>
            <div class="metric-label">out of 5.0</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        num_eval = phys_behavior['Num_Evaluations']
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Evaluations</div>
            <div class="metric-value">{num_eval:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        num_evaluators = phys_behavior.get('Num_Evaluators', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Evaluators</div>
            <div class="metric-value">{num_evaluators:.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        neg_comments = phys_behavior.get('Negative_Comment_Count', 0)
        color = "#28a745" if neg_comments == 0 else "#ffc107" if neg_comments < 3 else "#dc3545"
        st.markdown(f"""
        <div class="metric-card" style="background: {color};">
            <div class="metric-label">Negative Comments</div>
            <div class="metric-value">{neg_comments:.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "📊 Performance Metrics", "💬 Comments", "🏥 Clinical Indicators"])
    
    # Tab 1: Trends
    with tab1:
        st.subheader("📈 Multi-Year Trend Analysis")
        
        if len(phys_indicators) > 0:
            # Prepare trend data
            trend_data = []
            
            for _, row in phys_indicators.iterrows():
                year = row.get('Year', 'Unknown')
                trend_data.append({
                    'Year': year,
                    'Metric': 'Clinic Visits',
                    'Value': row.get('ClinicVisits', 0)
                })
                trend_data.append({
                    'Year': year,
                    'Metric': 'Waiting Time (min)',
                    'Value': row.get('ClinicWaitingTime', 0)
                })
                trend_data.append({
                    'Year': year,
                    'Metric': 'Patient Complaints',
                    'Value': row.get('PatientComplaints', 0)
                })
            
            trend_df = pd.DataFrame(trend_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Clinic visits trend
                visits_data = trend_df[trend_df['Metric'] == 'Clinic Visits']
                fig = px.line(
                    visits_data,
                    x='Year',
                    y='Value',
                    markers=True,
                    title="Clinic Visits Over Time",
                    labels={'Value': 'Number of Visits', 'Year': 'Fiscal Year'}
                )
                fig.update_traces(line_color='#667eea', line_width=3, marker_size=10)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Waiting time trend
                wait_data = trend_df[trend_df['Metric'] == 'Waiting Time (min)']
                fig = px.line(
                    wait_data,
                    x='Year',
                    y='Value',
                    markers=True,
                    title="Average Waiting Time Over Time",
                    labels={'Value': 'Minutes', 'Year': 'Fiscal Year'}
                )
                fig.update_traces(line_color='#ff7f0e', line_width=3, marker_size=10)
                st.plotly_chart(fig, use_container_width=True)
            
            # Combined metrics table
            st.subheader("📋 Historical Metrics")
            display_df = phys_indicators[['FiscalCycle', 'ClinicVisits', 'ClinicWaitingTime', 'PatientComplaints']].copy()
            display_df.columns = ['Fiscal Cycle', 'Clinic Visits', 'Avg Wait Time (min)', 'Complaints']
            st.dataframe(
                display_df.style.format({
                    'Clinic Visits': '{:,.0f}',
                    'Avg Wait Time (min)': '{:.2f}',
                    'Complaints': '{:.0f}'
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("ℹ️ No clinical indicator data available for this physician.")
    
    # Tab 2: Performance Metrics
    with tab2:
        st.subheader("📊 Detailed Performance Breakdown")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Score breakdown
            st.markdown("### Behavior Score Analysis")
            
            metrics = {
                'Average Score': phys_behavior['Avg_Score'],
                'Standard Deviation': phys_behavior['Std_Dev'],
                'Min Score': phys_behavior['Min_Score'],
                'Max Score': phys_behavior['Max_Score'],
                'Score Range': phys_behavior['Max_Score'] - phys_behavior['Min_Score']
            }
            
            for metric, value in metrics.items():
                st.metric(metric, f"{value:.3f}")
            
            # Comparison to average
            overall_avg = doctor_stats['Avg_Score'].mean()
            diff = phys_behavior['Avg_Score'] - overall_avg
            
            if diff > 0:
                st.success(f"✅ {diff:.3f} points above institutional average")
            elif diff < 0:
                st.warning(f"⚠️ {abs(diff):.3f} points below institutional average")
            else:
                st.info("ℹ️ At institutional average")
        
        with col2:
            # Gauge chart for behavior score
            st.markdown("### Behavior Score Gauge")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=phys_behavior['Avg_Score'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Behavior Score"},
                delta={'reference': overall_avg, 'increasing': {'color': "green"}},
                gauge={
                    'axis': {'range': [None, 5], 'tickwidth': 1},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 3], 'color': "lightgray"},
                        {'range': [3, 4], 'color': "gray"},
                        {'range': [4, 5], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 4.5
                    }
                }
            ))
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        # Percentile ranking
        st.markdown("---")
        st.markdown("### 📊 Percentile Ranking")
        
        percentile = (doctor_stats['Avg_Score'] < phys_behavior['Avg_Score']).sum() / len(doctor_stats) * 100
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div class="info-box" style="text-align: center;">
                <h2 style="margin: 0;">{percentile:.1f}th Percentile</h2>
                <p>This physician scores better than {percentile:.1f}% of all physicians</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Tab 3: Comments
    with tab3:
        st.subheader("💬 Comment Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pos_count = phys_behavior.get('Sentiment_Positive', 0)
            st.markdown(f"""
            <div class="success-box">
                <h3 style="margin: 0;">✅ Positive</h3>
                <h1 style="margin: 0.5rem 0;">{pos_count:.0f}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            neu_count = phys_behavior.get('Sentiment_Neutral', 0)
            st.markdown(f"""
            <div class="info-box">
                <h3 style="margin: 0;">➖ Neutral</h3>
                <h1 style="margin: 0.5rem 0;">{neu_count:.0f}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            neg_count = phys_behavior.get('Sentiment_Negative', 0)
            st.markdown(f"""
            <div class="warning-box">
                <h3 style="margin: 0;">❌ Negative</h3>
                <h1 style="margin: 0.5rem 0;">{neg_count:.0f}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        # Sentiment pie chart
        total_comments = pos_count + neu_count + neg_count
        if total_comments > 0:
            sentiment_data = pd.DataFrame({
                'Sentiment': ['Positive', 'Neutral', 'Negative'],
                'Count': [pos_count, neu_count, neg_count]
            })
            
            fig = px.pie(
                sentiment_data,
                values='Count',
                names='Sentiment',
                title="Sentiment Distribution",
                color='Sentiment',
                color_discrete_map={'Positive': '#28a745', 'Neutral': '#ffc107', 'Negative': '#dc3545'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Negative percentage
            neg_pct = phys_behavior.get('Negative_Comment_Pct', 0)
            if neg_pct > 0:
                if neg_pct > 30:
                    st.warning(f"⚠️ {neg_pct:.1f}% of comments are negative - may need attention")
                else:
                    st.info(f"ℹ️ {neg_pct:.1f}% of comments are negative")
    
    # Tab 4: Clinical Indicators
    with tab4:
        st.subheader("🏥 Clinical Performance Indicators")
        
        if len(phys_indicators) > 0:
            latest = phys_indicators.iloc[-1]  # Most recent data
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Latest Period Metrics")
                st.markdown(f"**Fiscal Cycle:** {latest.get('FiscalCycle', 'N/A')}")
                st.markdown(f"**Department:** {latest.get('Department', 'N/A')}")
                
                st.metric("Clinic Visits", f"{latest.get('ClinicVisits', 0):,.0f}")
                st.metric("Avg Waiting Time", f"{latest.get('ClinicWaitingTime', 0):.1f} min")
                st.metric("Patient Complaints", f"{latest.get('PatientComplaints', 0):.0f}")
                
                # Years of service
                if 'Years_of_Service' in latest and not pd.isna(latest['Years_of_Service']):
                    st.metric("Years of Service", f"{latest['Years_of_Service']:.1f} years")
            
            with col2:
                st.markdown("### Performance Indicators")
                
                # Create a radar chart if we have multiple years
                if len(phys_indicators) > 1:
                    categories = ['Clinic Visits', 'Waiting Time', 'Complaints']
                    
                    fig = go.Figure()
                    
                    for idx, row in phys_indicators.iterrows():
                        fig.add_trace(go.Scatterpolar(
                            r=[
                                row.get('ClinicVisits', 0) / 100,  # Normalize
                                100 - row.get('ClinicWaitingTime', 0),  # Inverse (lower is better)
                                100 - row.get('PatientComplaints', 0) * 10  # Inverse
                            ],
                            theta=categories,
                            fill='toself',
                            name=str(row.get('FiscalCycle', 'Unknown'))
                        ))
                    
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                        showlegend=True,
                        title="Multi-Year Comparison"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("ℹ️ Multiple years needed for trend comparison")
        else:
            st.info("ℹ️ No clinical indicator data available for this physician.")

# ============================================================================
# DEPARTMENTAL ANALYSIS MODE
# ============================================================================
elif mode == "🏥 Departmental Analysis":
    st.header("🏥 Departmental Performance Analysis")
    
    # Determine which department column to use
    if 'Department' in merged.columns:
        dept_col = 'Department'
        departments = sorted(merged[dept_col].dropna().unique())
    elif 'Source' in doctor_stats.columns:
        dept_col = 'Source'
        departments = sorted(doctor_stats[dept_col].dropna().unique())
    else:
        st.error("No department information available in the data.")
        st.stop()
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    selected_departments = st.sidebar.multiselect(
        "Select Department(s)",
        departments,
        default=departments
    )
    
    if len(selected_departments) == 0:
        st.warning("⚠️ Please select at least one department")
        st.stop()
    
    # Filter data
    if dept_col in doctor_stats.columns:
        dept_data = doctor_stats[doctor_stats[dept_col].isin(selected_departments)]
    else:
        dept_data = merged[merged[dept_col].isin(selected_departments)]
    
    # Overview metrics
    st.subheader("📊 Departmental Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Selected Departments", len(selected_departments))
    
    with col2:
        total_physicians = len(dept_data)
        st.metric("Total Physicians", f"{total_physicians:,}")
    
    with col3:
        avg_dept_score = dept_data['Avg_Score'].mean()
        st.metric("Average Behavior Score", f"{avg_dept_score:.2f}/5.0")
    
    with col4:
        total_eval = dept_data['Num_Evaluations'].sum()
        st.metric("Total Evaluations", f"{total_eval:,}")
    
    st.markdown("---")
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Comparison", "📈 Rankings", "💬 Sentiment", "🎯 Insights"])
    
    # Tab 1: Department Comparison
    with tab1:
        st.subheader("📊 Department Performance Comparison")
        
        # Group by department
        dept_summary = dept_data.groupby(dept_col).agg({
            'Avg_Score': ['mean', 'std', 'count'],
            'Num_Evaluations': 'sum',
            'Num_Evaluators': 'mean'
        }).round(3)
        
        dept_summary.columns = ['_'.join(col).strip('_') for col in dept_summary.columns]
        dept_summary = dept_summary.rename(columns={
            'Avg_Score_mean': 'Avg_Score',
            'Avg_Score_std': 'Score_StdDev',
            'Avg_Score_count': 'Num_Physicians',
            'Num_Evaluations': 'Total_Evaluations',
            'Num_Evaluators': 'Avg_Evaluators'
        })
        dept_summary = dept_summary.reset_index()
        dept_summary = dept_summary.sort_values('Avg_Score', ascending=False)
        
        # Bar chart comparison
        fig = px.bar(
            dept_summary,
            x=dept_col,
            y='Avg_Score',
            color='Avg_Score',
            title="Average Behavior Score by Department",
            labels={'Avg_Score': 'Average Score'},
            color_continuous_scale='RdYlGn',
            text='Avg_Score'
        )
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Department summary table
        st.markdown("### 📋 Department Statistics")
        st.dataframe(
            dept_summary.style.format({
                'Avg_Score': '{:.3f}',
                'Score_StdDev': '{:.3f}',
                'Num_Physicians': '{:.0f}',
                'Total_Evaluations': '{:,.0f}',
                'Avg_Evaluators': '{:.1f}'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        # Box plot for distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(
                dept_data,
                x=dept_col,
                y='Avg_Score',
                title="Score Distribution by Department",
                labels={'Avg_Score': 'Behavior Score'},
                color=dept_col
            )
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.violin(
                dept_data,
                x=dept_col,
                y='Avg_Score',
                title="Score Density by Department",
                labels={'Avg_Score': 'Behavior Score'},
                color=dept_col,
                box=True
            )
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Rankings
    with tab2:
        st.subheader("📈 Departmental Rankings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🏆 Top Performing Departments")
            top_depts = dept_summary.nlargest(10, 'Avg_Score')
            top_depts['Rank'] = range(1, len(top_depts) + 1)
            display_cols = ['Rank', dept_col, 'Avg_Score', 'Num_Physicians', 'Total_Evaluations']
            st.dataframe(
                top_depts[display_cols].style.format({
                    'Avg_Score': '{:.3f}',
                    'Num_Physicians': '{:.0f}',
                    'Total_Evaluations': '{:,.0f}'
                }),
                use_container_width=True,
                hide_index=True
            )
        
        with col2:
            st.markdown("### ⚠️ Departments Needing Attention")
            bottom_depts = dept_summary.nsmallest(10, 'Avg_Score')
            bottom_depts['Rank'] = range(1, len(bottom_depts) + 1)
            st.dataframe(
                bottom_depts[display_cols].style.format({
                    'Avg_Score': '{:.3f}',
                    'Num_Physicians': '{:.0f}',
                    'Total_Evaluations': '{:,.0f}'
                }),
                use_container_width=True,
                hide_index=True
            )
        
        # Score vs number of physicians scatter
        st.markdown("---")
        st.markdown("### 📊 Score vs Department Size")
        
        fig = px.scatter(
            dept_summary,
            x='Num_Physicians',
            y='Avg_Score',
            size='Total_Evaluations',
            color='Avg_Score',
            hover_name=dept_col,
            title="Department Score vs Size",
            labels={
                'Num_Physicians': 'Number of Physicians',
                'Avg_Score': 'Average Score',
                'Total_Evaluations': 'Total Evaluations'
            },
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Sentiment Analysis
    with tab3:
        st.subheader("💬 Comment Sentiment by Department")
        
        if 'Sentiment_Negative' in dept_data.columns:
            # Aggregate sentiment by department
            sentiment_dept = dept_data.groupby(dept_col).agg({
                'Sentiment_Positive': 'sum',
                'Sentiment_Neutral': 'sum',
                'Sentiment_Negative': 'sum',
                'Total_Comments': 'sum'
            }).reset_index()
            
            # Melt for grouped bar chart
            sentiment_melted = sentiment_dept.melt(
                id_vars=[dept_col],
                value_vars=['Sentiment_Positive', 'Sentiment_Neutral', 'Sentiment_Negative'],
                var_name='Sentiment',
                value_name='Count'
            )
            sentiment_melted['Sentiment'] = sentiment_melted['Sentiment'].str.replace('Sentiment_', '')
            
            # Grouped bar chart
            fig = px.bar(
                sentiment_melted,
                x=dept_col,
                y='Count',
                color='Sentiment',
                title="Comment Sentiment Distribution by Department",
                labels={'Count': 'Number of Comments'},
                color_discrete_map={'Positive': '#28a745', 'Neutral': '#ffc107', 'Negative': '#dc3545'},
                barmode='group'
            )
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate negative percentage
            sentiment_dept['Negative_Pct'] = (
                sentiment_dept['Sentiment_Negative'] / sentiment_dept['Total_Comments'] * 100
            ).fillna(0)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Negative percentage by department
                fig = px.bar(
                    sentiment_dept.sort_values('Negative_Pct', ascending=False),
                    x=dept_col,
                    y='Negative_Pct',
                    title="Negative Comment Percentage by Department",
                    labels={'Negative_Pct': 'Negative %'},
                    color='Negative_Pct',
                    color_continuous_scale='Reds'
                )
                fig.update_xaxis(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Total comments by department
                fig = px.bar(
                    sentiment_dept.sort_values('Total_Comments', ascending=False),
                    x=dept_col,
                    y='Total_Comments',
                    title="Total Comments by Department",
                    labels={'Total_Comments': 'Total Comments'},
                    color='Total_Comments',
                    color_continuous_scale='Blues'
                )
                fig.update_xaxis(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Insights
    with tab4:
        st.subheader("🎯 Key Insights & Recommendations")
        
        # Find best and worst departments
        best_dept = dept_summary.iloc[0]
        worst_dept = dept_summary.iloc[-1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🌟 Best Performing Department")
            st.markdown(f"""
            <div class="success-box">
                <h3>{best_dept[dept_col]}</h3>
                <p><strong>Average Score:</strong> {best_dept['Avg_Score']:.3f}/5.0</p>
                <p><strong>Physicians:</strong> {best_dept['Num_Physicians']:.0f}</p>
                <p><strong>Evaluations:</strong> {best_dept['Total_Evaluations']:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### ⚠️ Department Needing Most Attention")
            st.markdown(f"""
            <div class="warning-box">
                <h3>{worst_dept[dept_col]}</h3>
                <p><strong>Average Score:</strong> {worst_dept['Avg_Score']:.3f}/5.0</p>
                <p><strong>Physicians:</strong> {worst_dept['Num_Physicians']:.0f}</p>
                <p><strong>Evaluations:</strong> {worst_dept['Total_Evaluations']:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Variability analysis
        st.markdown("---")
        st.markdown("### 📊 Consistency Analysis")
        
        high_variability = dept_summary[dept_summary['Score_StdDev'] > dept_summary['Score_StdDev'].quantile(0.75)]
        
        if len(high_variability) > 0:
            st.warning(f"⚠️ **{len(high_variability)} departments** show high score variability:")
            for _, dept in high_variability.iterrows():
                st.markdown(f"- **{dept[dept_col]}**: StdDev = {dept['Score_StdDev']:.3f}")
            st.markdown("*High variability may indicate inconsistent performance or diverse physician populations*")
        
        # Recommendations
        st.markdown("---")
        st.markdown("### 💡 Recommendations")
        
        recommendations = []
        
        # Low score departments
        low_score_depts = dept_summary[dept_summary['Avg_Score'] < dept_summary['Avg_Score'].quantile(0.25)]
        if len(low_score_depts) > 0:
            recommendations.append(f"🎯 **Priority Training**: {', '.join(low_score_depts[dept_col].tolist())} - scores below 25th percentile")
        
        # High negative sentiment
        if 'Sentiment_Negative' in dept_data.columns:
            high_neg = sentiment_dept[sentiment_dept['Negative_Pct'] > 20]
            if len(high_neg) > 0:
                recommendations.append(f"💬 **Address Feedback**: {', '.join(high_neg[dept_col].tolist())} - >20% negative comments")
        
        # High variability
        if len(high_variability) > 0:
            recommendations.append(f"📊 **Standardize Practices**: {', '.join(high_variability[dept_col].tolist())} - high performance variability")
        
        # Small sample size
        small_sample = dept_summary[dept_summary['Total_Evaluations'] < dept_summary['Total_Evaluations'].quantile(0.25)]
        if len(small_sample) > 0:
            recommendations.append(f"📝 **Increase Evaluation Participation**: {', '.join(small_sample[dept_col].tolist())} - low evaluation counts")
        
        if recommendations:
            for rec in recommendations:
                st.markdown(rec)
        else:
            st.success("✅ All departments performing within acceptable ranges!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; padding: 2rem;">
    <p>AUBMC Physician Performance Dashboard | Data updated: 2025</p>
    <p><em>For internal use only</em></p>
</div>
""", unsafe_allow_html=True)
