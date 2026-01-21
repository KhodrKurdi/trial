import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
import re

# Page configuration
st.set_page_config(
    page_title="AUBMC Behavior Survey Dashboard",
    page_icon="📊",
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

# Load data
@st.cache_data
def load_data():
    # Load the numeric long format data
df = pd.read_csv("All_Departments_Long_Numeric.csv.zip")
    
    # Extract year from Fillout Date
    df['Fillout Date (mm/dd/yy)'] = pd.to_datetime(df['Fillout Date (mm/dd/yy)'], errors='coerce')
    df['Year'] = df['Fillout Date (mm/dd/yy)'].dt.year
    
    # Combine comments columns
    df['Comments_Combined'] = df['Q2_Comments'].fillna('') + ' ' + df['Q2_Comments\n'].fillna('')
    df['Comments_Combined'] = df['Comments_Combined'].str.strip()
    df['Comments_Combined'] = df['Comments_Combined'].replace('', np.nan)
    
    return df

# Sentiment analysis function
@st.cache_data
def analyze_sentiment(text):
    if pd.isna(text) or text.strip() == '':
        return 'Neutral', 0.0
    try:
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        if polarity > 0.1:
            return 'Positive', polarity
        elif polarity < -0.1:
            return 'Negative', polarity
        else:
            return 'Neutral', polarity
    except:
        return 'Neutral', 0.0

# Load data
df = load_data()

# Add sentiment analysis
df['Sentiment'], df['Sentiment_Score'] = zip(*df['Comments_Combined'].apply(analyze_sentiment))

# Main title
st.markdown('<div class="main-header">📊 AUBMC Behavior Survey Dashboard</div>', unsafe_allow_html=True)
st.markdown("---")

# Mode selection
col1, col2 = st.columns(2)

with col1:
    if st.button("👤 Individual Physician Report", use_container_width=True):
        st.session_state.mode = "individual"

with col2:
    if st.button("🏥 Departmental Report", use_container_width=True):
        st.session_state.mode = "departmental"

# Initialize session state
if 'mode' not in st.session_state:
    st.session_state.mode = None

st.markdown("---")

# ============================================================================
# INDIVIDUAL PHYSICIAN REPORT
# ============================================================================
if st.session_state.mode == "individual":
    st.header("👤 Individual Physician Report")
    
    # Filters in sidebar
    with st.sidebar:
        st.header("🔍 Filters")
        
        # Physician selection
        physicians = sorted(df['Subject ID'].unique())
        selected_physician = st.selectbox("Select Physician", physicians)
        
        # Year selection
        years = sorted(df['Year'].dropna().unique())
        selected_years = st.multiselect("Select Year(s)", years, default=years)
        
        # Source filter
        sources = sorted(df['Source'].unique())
        selected_source = st.multiselect("Select Department(s)", sources, default=sources)
    
    # Filter data
    physician_data = df[
        (df['Subject ID'] == selected_physician) &
        (df['Year'].isin(selected_years)) &
        (df['Source'].isin(selected_source))
    ]
    
    if len(physician_data) == 0:
        st.warning("No data available for selected filters.")
    else:
        # Overview metrics
        st.subheader("📈 Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            overall_avg = physician_data['Response_Numeric'].mean()
            st.metric("Overall Average Score", f"{overall_avg:.2f}/5.00")
        
        with col2:
            total_responses = len(physician_data)
            st.metric("Total Responses", f"{total_responses:,}")
        
        with col3:
            num_evaluators = physician_data['Rater Name'].nunique()
            st.metric("Unique Evaluators", f"{num_evaluators}")
        
        with col4:
            std_dev = physician_data['Response_Numeric'].std()
            st.metric("Score Std Dev", f"{std_dev:.2f}")
        
        st.markdown("---")
        
        # Average score by year
        st.subheader("📊 Average Score by Year")
        
        yearly_avg = physician_data.groupby('Year')['Response_Numeric'].agg(['mean', 'count']).reset_index()
        yearly_avg.columns = ['Year', 'Average_Score', 'Count']
        
        fig_yearly = go.Figure()
        
        fig_yearly.add_trace(go.Bar(
            x=yearly_avg['Year'],
            y=yearly_avg['Average_Score'],
            text=yearly_avg['Average_Score'].round(2),
            textposition='outside',
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(yearly_avg)],
            hovertemplate='<b>Year %{x}</b><br>Average: %{y:.2f}<br>Responses: %{customdata}<extra></extra>',
            customdata=yearly_avg['Count']
        ))
        
        fig_yearly.update_layout(
            xaxis_title="Year",
            yaxis_title="Average Score",
            yaxis_range=[0, 5],
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_yearly, use_container_width=True)
        
        # Display yearly data table
        st.dataframe(
            yearly_avg.style.format({'Average_Score': '{:.2f}', 'Count': '{:,}'}),
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Sentiment analysis of comments
        st.subheader("💬 Comment Sentiment Analysis")
        
        comments_data = physician_data[physician_data['Comments_Combined'].notna()].copy()
        
        if len(comments_data) == 0:
            st.info("No comments available for selected filters.")
        else:
            col1, col2, col3 = st.columns(3)
            
            sentiment_counts = comments_data['Sentiment'].value_counts()
            
            with col1:
                positive_count = sentiment_counts.get('Positive', 0)
                if st.button(f"✅ Positive: {positive_count}", key="positive"):
                    st.session_state.show_comments = 'Positive'
            
            with col2:
                neutral_count = sentiment_counts.get('Neutral', 0)
                if st.button(f"➖ Neutral: {neutral_count}", key="neutral"):
                    st.session_state.show_comments = 'Neutral'
            
            with col3:
                negative_count = sentiment_counts.get('Negative', 0)
                if st.button(f"❌ Negative: {negative_count}", key="negative"):
                    st.session_state.show_comments = 'Negative'
            
            # Show comments popup
            if 'show_comments' in st.session_state:
                sentiment_filter = st.session_state.show_comments
                filtered_comments = comments_data[comments_data['Sentiment'] == sentiment_filter]
                
                st.markdown(f"### {sentiment_filter} Comments ({len(filtered_comments)})")
                
                if len(filtered_comments) > 0:
                    display_df = filtered_comments[['Year', 'Raters Group', 'Comments_Combined', 'Sentiment_Score']].copy()
                    display_df.columns = ['Year', 'Rater Group', 'Comment', 'Sentiment Score']
                    display_df = display_df.sort_values('Year', ascending=False)
                    
                    st.dataframe(
                        display_df.style.format({'Sentiment Score': '{:.2f}'}),
                        use_container_width=True,
                        height=400
                    )
                    
                    # Download button
                    csv = display_df.to_csv(index=False)
                    st.download_button(
                        label=f"Download {sentiment_filter} Comments CSV",
                        data=csv,
                        file_name=f"{selected_physician}_{sentiment_filter}_comments.csv",
                        mime="text/csv"
                    )
                else:
                    st.info(f"No {sentiment_filter} comments found.")
            
            # Sentiment pie chart
            st.markdown("### Sentiment Distribution")
            fig_sentiment = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                color=sentiment_counts.index,
                color_discrete_map={'Positive': '#2ca02c', 'Neutral': '#ff7f0e', 'Negative': '#d62728'}
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)
        
        st.markdown("---")
        
        # Score breakdown by question
        st.subheader("📋 Score Breakdown by Question")
        
        question_avg = physician_data.groupby('Question')['Response_Numeric'].agg(['mean', 'count']).reset_index()
        question_avg.columns = ['Question', 'Average_Score', 'Count']
        question_avg = question_avg.sort_values('Average_Score')
        
        # Simplify question names for display
        question_avg['Question_Short'] = question_avg['Question'].str[:50] + '...'
        
        fig_questions = px.bar(
            question_avg,
            x='Average_Score',
            y='Question_Short',
            orientation='h',
            text='Average_Score',
            color='Average_Score',
            color_continuous_scale='RdYlGn'
        )
        
        fig_questions.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_questions.update_layout(
            xaxis_title="Average Score",
            yaxis_title="Question",
            xaxis_range=[0, 5],
            height=max(400, len(question_avg) * 25),
            showlegend=False
        )
        
        st.plotly_chart(fig_questions, use_container_width=True)
        
        st.markdown("---")
        
        # Rater group breakdown
        st.subheader("👥 Score by Rater Group")
        
        rater_avg = physician_data.groupby('Raters Group')['Response_Numeric'].agg(['mean', 'count']).reset_index()
        rater_avg.columns = ['Rater_Group', 'Average_Score', 'Count']
        rater_avg = rater_avg.sort_values('Average_Score', ascending=False)
        
        fig_rater = px.bar(
            rater_avg,
            x='Rater_Group',
            y='Average_Score',
            text='Average_Score',
            color='Average_Score',
            color_continuous_scale='RdYlGn'
        )
        
        fig_rater.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_rater.update_layout(
            xaxis_title="Rater Group",
            yaxis_title="Average Score",
            yaxis_range=[0, 5],
            height=400
        )
        
        st.plotly_chart(fig_rater, use_container_width=True)

# ============================================================================
# DEPARTMENTAL REPORT
# ============================================================================
elif st.session_state.mode == "departmental":
    st.header("🏥 Departmental Report")
    
    # Filters in sidebar
    with st.sidebar:
        st.header("🔍 Filters")
        
        # Year selection
        years = sorted(df['Year'].dropna().unique())
        selected_years = st.multiselect("Select Year(s)", years, default=years)
        
        # Source filter
        sources = sorted(df['Source'].unique())
        selected_sources = st.multiselect("Select Department(s)", sources, default=sources)
    
    # Filter data
    dept_data = df[
        (df['Year'].isin(selected_years)) &
        (df['Source'].isin(selected_sources))
    ]
    
    if len(dept_data) == 0:
        st.warning("No data available for selected filters.")
    else:
        # Overview metrics
        st.subheader("📈 Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            overall_avg = dept_data['Response_Numeric'].mean()
            st.metric("Overall Average Score", f"{overall_avg:.2f}/5.00")
        
        with col2:
            total_physicians = dept_data['Subject ID'].nunique()
            st.metric("Total Physicians", f"{total_physicians:,}")
        
        with col3:
            total_responses = len(dept_data)
            st.metric("Total Responses", f"{total_responses:,}")
        
        with col4:
            total_evaluators = dept_data['Rater Name'].nunique()
            st.metric("Total Evaluators", f"{total_evaluators:,}")
        
        st.markdown("---")
        
        # Average score by department and year
        st.subheader("📊 Average Score by Department and Year")
        
        dept_year_avg = dept_data.groupby(['Source', 'Year'])['Response_Numeric'].agg(['mean', 'count']).reset_index()
        dept_year_avg.columns = ['Department', 'Year', 'Average_Score', 'Count']
        
        fig_dept_year = px.line(
            dept_year_avg,
            x='Year',
            y='Average_Score',
            color='Department',
            markers=True,
            text='Average_Score'
        )
        
        fig_dept_year.update_traces(texttemplate='%{text:.2f}', textposition='top center')
        fig_dept_year.update_layout(
            xaxis_title="Year",
            yaxis_title="Average Score",
            yaxis_range=[0, 5],
            height=500
        )
        
        st.plotly_chart(fig_dept_year, use_container_width=True)
        
        st.markdown("---")
        
        # Department comparison
        st.subheader("🏆 Department Comparison")
        
        dept_stats = dept_data.groupby('Source').agg({
            'Response_Numeric': ['mean', 'std', 'count'],
            'Subject ID': 'nunique'
        }).round(2)
        
        dept_stats.columns = ['Average_Score', 'Std_Dev', 'Total_Responses', 'Num_Physicians']
        dept_stats = dept_stats.reset_index()
        dept_stats = dept_stats.sort_values('Average_Score', ascending=False)
        
        # Bar chart
        fig_dept = px.bar(
            dept_stats,
            x='Source',
            y='Average_Score',
            text='Average_Score',
            color='Average_Score',
            color_continuous_scale='RdYlGn'
        )
        
        fig_dept.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_dept.update_layout(
            xaxis_title="Department",
            yaxis_title="Average Score",
            yaxis_range=[0, 5],
            height=400
        )
        
        st.plotly_chart(fig_dept, use_container_width=True)
        
        # Stats table
        st.dataframe(
            dept_stats.style.format({
                'Average_Score': '{:.2f}',
                'Std_Dev': '{:.2f}',
                'Total_Responses': '{:,}',
                'Num_Physicians': '{:,}'
            }),
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Top and bottom performers
        st.subheader("🌟 Top & Bottom Performers")
        
        physician_avg = dept_data.groupby('Subject ID').agg({
            'Response_Numeric': 'mean',
            'Source': 'first'
        }).reset_index()
        physician_avg.columns = ['Physician_ID', 'Average_Score', 'Department']
        physician_avg = physician_avg.sort_values('Average_Score', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏆 Top 10 Performers")
            top_10 = physician_avg.head(10)
            st.dataframe(
                top_10.style.format({'Average_Score': '{:.2f}'}),
                use_container_width=True
            )
        
        with col2:
            st.markdown("#### ⚠️ Bottom 10 Performers")
            bottom_10 = physician_avg.tail(10).sort_values('Average_Score')
            st.dataframe(
                bottom_10.style.format({'Average_Score': '{:.2f}'}),
                use_container_width=True
            )
        
        st.markdown("---")
        
        # Sentiment analysis by department
        st.subheader("💬 Comment Sentiment by Department")
        
        comments_dept = dept_data[dept_data['Comments_Combined'].notna()].copy()
        
        if len(comments_dept) > 0:
            sentiment_dept = comments_dept.groupby(['Source', 'Sentiment']).size().reset_index(name='Count')
            
            fig_sentiment_dept = px.bar(
                sentiment_dept,
                x='Source',
                y='Count',
                color='Sentiment',
                barmode='group',
                color_discrete_map={'Positive': '#2ca02c', 'Neutral': '#ff7f0e', 'Negative': '#d62728'}
            )
            
            fig_sentiment_dept.update_layout(
                xaxis_title="Department",
                yaxis_title="Number of Comments",
                height=400
            )
            
            st.plotly_chart(fig_sentiment_dept, use_container_width=True)
        else:
            st.info("No comments available for selected filters.")
        
        st.markdown("---")
        
        # Distribution of scores
        st.subheader("📈 Score Distribution")
        
        fig_hist = px.histogram(
            dept_data,
            x='Response_Numeric',
            color='Source',
            nbins=20,
            barmode='overlay',
            opacity=0.7
        )
        
        fig_hist.update_layout(
            xaxis_title="Score",
            yaxis_title="Frequency",
            height=400
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)

# ============================================================================
# HOME PAGE (when no mode selected)
# ============================================================================
else:
    st.info("👆 Please select a report type above to get started.")
    
    # Show some overall statistics
    st.subheader("📊 Overall Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Physicians", f"{df['Subject ID'].nunique():,}")
    
    with col2:
        st.metric("Total Responses", f"{len(df):,}")
    
    with col3:
        st.metric("Average Score", f"{df['Response_Numeric'].mean():.2f}/5.00")
    
    st.markdown("---")
    
    # Quick overview chart
    st.subheader("📈 Responses Over Time")
    
    responses_by_year = df.groupby('Year').size().reset_index(name='Count')
    
    fig_overview = px.bar(
        responses_by_year,
        x='Year',
        y='Count',
        text='Count'
    )
    
    fig_overview.update_traces(texttemplate='%{text:,}', textposition='outside')
    fig_overview.update_layout(
        xaxis_title="Year",
        yaxis_title="Number of Responses",
        height=400
    )
    
    st.plotly_chart(fig_overview, use_container_width=True)
