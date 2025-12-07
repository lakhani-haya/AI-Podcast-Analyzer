hi import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="AI Podcast Analyzer",
    page_icon="🎧",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        data_path = Path(__file__).parent.parent / "data"
        
        episodes_file = data_path / "episodes.json"
        topics_file = data_path / "topics_with_sentiment.csv"
        
        data = {}
        
        if episodes_file.exists():
            with open(episodes_file, 'r', encoding='utf-8') as f:
                data['episodes'] = json.load(f)
        else:
            data['episodes'] = None
            
        if topics_file.exists():
            data['topics'] = pd.read_csv(topics_file, encoding='utf-8')
            if 'pub_date' in data['topics'].columns:
                data['topics']['pub_date'] = pd.to_datetime(data['topics']['pub_date'])
        else:
            data['topics'] = None
            
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return {'episodes': None, 'topics': None}

def show_setup_instructions():
    st.title("AI Podcast Analyzer")
    st.warning("No data found. Please run the analysis first.")
    
    st.markdown("""
    ### Setup Instructions:
    
    **1. Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    
    **2. Set up API key:**
    Create a `.env` file with:
    ```
    LISTEN_API_KEY=your_api_key_here
    ```
    
    **3. Run analysis:**
    ```bash
    python run_analysis.py
    ```
    
    **4. Start this app:**
    ```bash
    streamlit run app/streamlit_app.py
    ```
    """)

def main():
    data = load_data()
    
    if not data['episodes'] and not data['topics']:
        show_setup_instructions()
        return
    
    st.markdown('<h1 class="main-header">🎧 AI Podcast Analyzer</h1>', unsafe_allow_html=True)
    
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "Dashboard", 
        "Episodes", 
        "Analytics"
    ])
    
    if page == "Dashboard":
        show_dashboard(data)
    elif page == "Episodes":
        show_episodes(data)
    elif page == "Analytics":
        show_analytics(data)

def show_dashboard(data):
    st.header("Dashboard")
    
    if data['episodes']:
        episodes_df = pd.DataFrame(data['episodes'])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Episodes", len(episodes_df))
        
        with col2:
            if 'podcast_title' in episodes_df.columns:
                unique_podcasts = episodes_df['podcast_title'].nunique()
                st.metric("Podcasts", unique_podcasts)
        
        with col3:
            if data['topics'] is not None:
                avg_sentiment = data['topics']['overall_sentiment_compound'].mean()
                st.metric("Avg Sentiment", f"{avg_sentiment:.3f}")
    
    if data['topics'] is not None:
        df = data['topics']
        
        st.subheader("Sentiment Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            if 'overall_sentiment_label' in df.columns:
                sentiment_counts = df['overall_sentiment_label'].value_counts()
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
                ax.set_title('Sentiment Distribution')
                st.pyplot(fig)
        
        with col2:
            if 'overall_sentiment_compound' in df.columns:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.hist(df['overall_sentiment_compound'], bins=30, alpha=0.7, edgecolor='black')
                ax.set_xlabel('Sentiment Score')
                ax.set_ylabel('Frequency')
                ax.set_title('Sentiment Score Distribution')
                ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
                st.pyplot(fig)

def show_episodes(data):
    st.header("🔍 Episodes")
    
    if data['episodes']:
        episodes_df = pd.DataFrame(data['episodes'])
        
        search_term = st.text_input("Search episodes", placeholder="Enter keywords...")
        if search_term:
            episodes_df = episodes_df[episodes_df['title'].str.contains(search_term, case=False, na=False)]
        
        st.subheader(f"Episodes ({len(episodes_df)} found)")
        
        for _, episode in episodes_df.head(20).iterrows():
            with st.expander(f"{episode['title'][:80]}..."):
                st.write(f"**Podcast:** {episode.get('podcast_title', 'Unknown')}")
                if episode.get('description'):
                    st.write(f"**Description:** {episode['description'][:300]}...")

def show_analytics(data):
    st.header("Analytics")
    
    if data['topics'] is not None:
        df = data['topics']
        
        if 'topic_label' in df.columns:
            st.subheader("Top Topics")
            topic_counts = df['topic_label'].value_counts().head(10)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.barh(range(len(topic_counts)), topic_counts.values)
            ax.set_yticks(range(len(topic_counts)))
            ax.set_yticklabels(topic_counts.index)
            ax.set_xlabel('Number of Episodes')
            ax.set_title('Top 10 Topics')
            st.pyplot(fig)
        
        if 'pub_date' in df.columns:
            st.subheader("Sentiment Over Time")
            valid_dates_df = df.dropna(subset=['pub_date'])
            if not valid_dates_df.empty:
                daily_sentiment = valid_dates_df.groupby(valid_dates_df['pub_date'].dt.date)['overall_sentiment_compound'].mean()
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(daily_sentiment.index, daily_sentiment.values, marker='o', alpha=0.7)
                ax.set_xlabel('Date')
                ax.set_ylabel('Average Sentiment Score')
                ax.set_title('Sentiment Trends Over Time')
                ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                plt.xticks(rotation=45)
                st.pyplot(fig)

if __name__ == "__main__":
    main()

def show_dashboard(df):
    """Show main dashboard"""
    st.header("Podcast Analysis Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Episodes", len(df))
    
    with col2:
        avg_sentiment = df['overall_sentiment_compound'].mean()
        st.metric("Avg Sentiment", f"{avg_sentiment:.3f}")
    
    with col3:
        if 'topic_label' in df.columns:
            unique_topics = df['topic_label'].nunique()
            st.metric("Unique Topics", unique_topics)
    
    with col4:
        positive_pct = (df['overall_sentiment_label'] == 'positive').mean() * 100
        st.metric("Positive Episodes", f"{positive_pct:.1f}%")
    
    # Sentiment distribution
    st.subheader("Sentiment Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        sentiment_counts = df['overall_sentiment_label'].value_counts()
        ax.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
        ax.set_title('Sentiment Distribution')
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(df['overall_sentiment_compound'], bins=30, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Sentiment Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Sentiment Score Distribution')
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        st.pyplot(fig)
    
    # Top topics
    if 'topic_label' in df.columns:
        st.subheader("Top Topics")
        topic_counts = df['topic_label'].value_counts().head(10)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.barh(range(len(topic_counts)), topic_counts.values)
        ax.set_yticks(range(len(topic_counts)))
        ax.set_yticklabels(topic_counts.index)
        ax.set_xlabel('Number of Episodes')
        ax.set_title('Top 10 Topics')
        st.pyplot(fig)

def show_episode_explorer(df):
    """Show episode explorer page"""
    st.header("🔍 Episode Explorer")
    
    # Filters
    st.subheader("Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Sentiment filter
        sentiment_options = ['All'] + list(df['overall_sentiment_label'].unique())
        selected_sentiment = st.selectbox("Filter by Sentiment", sentiment_options)
    
    with col2:
        # Topic filter
        if 'topic_label' in df.columns:
            topic_options = ['All'] + list(df['topic_label'].unique())
            selected_topic = st.selectbox("Filter by Topic", topic_options)
        else:
            selected_topic = 'All'
    
    with col3:
        # Sort by
        sort_options = ['Publication Date', 'Sentiment Score', 'Title']
        sort_by = st.selectbox("Sort by", sort_options)
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_sentiment != 'All':
        filtered_df = filtered_df[filtered_df['overall_sentiment_label'] == selected_sentiment]
    
    if selected_topic != 'All' and 'topic_label' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['topic_label'] == selected_topic]
    
    # Sort
    if sort_by == 'Publication Date' and 'pub_date' in filtered_df.columns:
        filtered_df = filtered_df.sort_values('pub_date', ascending=False)
    elif sort_by == 'Sentiment Score':
        filtered_df = filtered_df.sort_values('overall_sentiment_compound', ascending=False)
    elif sort_by == 'Title':
        filtered_df = filtered_df.sort_values('title')
    
    # Search
    search_term = st.text_input("Search episodes", placeholder="Enter keywords to search in titles...")
    if search_term:
        filtered_df = filtered_df[filtered_df['title'].str.contains(search_term, case=False, na=False)]
    
    # Display results
    st.subheader(f"Episodes ({len(filtered_df)} found)")
    
    # Pagination
    episodes_per_page = 10
    total_pages = (len(filtered_df) - 1) // episodes_per_page + 1
    
    if total_pages > 1:
        page_num = st.selectbox("Page", range(1, total_pages + 1))
        start_idx = (page_num - 1) * episodes_per_page
        end_idx = start_idx + episodes_per_page
        page_df = filtered_df.iloc[start_idx:end_idx]
    else:
        page_df = filtered_df
    
    # Display episodes
    for _, episode in page_df.iterrows():
        display_episode_card(episode.to_dict())
        st.markdown("---")

def show_recommendations(df, recommendations_data):
    """Show recommendations page"""
    st.header("💡 Episode Recommendations")
    
    # Episode selection
    st.subheader("Get Recommendations")
    
    # Method selection
    method = st.radio("Choose recommendation method:", [
        "Search by Title",
        "Select from List",
        "Browse by Topic"
    ])
    
    if method == "Search by Title":
        search_title = st.text_input("Enter part of an episode title:")
        
        if search_title:
            # Find matching episodes
            matches = df[df['title'].str.contains(search_title, case=False, na=False)]
            
            if not matches.empty:
                selected_episode = st.selectbox("Select episode:", matches['title'].tolist())
                
                if selected_episode and recommendations_data:
                    show_episode_recommendations(selected_episode, recommendations_data, df)
            else:
                st.warning("No episodes found matching your search.")
    
    elif method == "Select from List":
        episode_titles = df['title'].tolist()
        selected_episode = st.selectbox("Choose an episode:", episode_titles)
        
        if selected_episode and recommendations_data:
            show_episode_recommendations(selected_episode, recommendations_data, df)
    
    elif method == "Browse by Topic":
        if 'topic_label' in df.columns:
            topics = df['topic_label'].unique()
            selected_topic = st.selectbox("Choose a topic:", topics)
            
            if selected_topic:
                topic_episodes = df[df['topic_label'] == selected_topic]
                st.subheader(f"Episodes in topic: {selected_topic}")
                
                for _, episode in topic_episodes.head(10).iterrows():
                    display_episode_card(episode.to_dict())
                    st.markdown("---")

def show_episode_recommendations(episode_title, recommendations_data, df):
    """Show recommendations for a specific episode"""
    if episode_title in recommendations_data:
        rec_data = recommendations_data[episode_title]
        
        st.subheader("Selected Episode")
        episode_info = rec_data['episode_info']
        display_episode_card(episode_info)
        
        st.subheader("Recommended Episodes")
        recommendations = rec_data['recommendations']
        
        if recommendations:
            for i, rec in enumerate(recommendations):
                st.markdown(f"**Recommendation {i+1}**")
                display_episode_card(rec, show_similarity=True)
                st.markdown("---")
        else:
            st.info("No recommendations available for this episode.")
    else:
        st.warning("Recommendations not available for this episode.")

def show_analytics(df):
    """Show analytics page"""
    st.header("📈 Advanced Analytics")
    
    # Temporal analysis
    if 'pub_date' in df.columns:
        st.subheader("Temporal Analysis")
        
        valid_dates_df = df.dropna(subset=['pub_date'])
        if not valid_dates_df.empty:
            # Sentiment over time
            daily_sentiment = valid_dates_df.groupby(valid_dates_df['pub_date'].dt.date)['overall_sentiment_compound'].mean()
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(daily_sentiment.index, daily_sentiment.values, marker='o', alpha=0.7)
            ax.set_xlabel('Date')
            ax.set_ylabel('Average Sentiment Score')
            ax.set_title('Sentiment Trends Over Time')
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            plt.xticks(rotation=45)
            st.pyplot(fig)
    
    # Topic-Sentiment relationship
    if 'topic_label' in df.columns:
        st.subheader("Topic-Sentiment Analysis")
        
        topic_sentiment = df.groupby('topic_label')['overall_sentiment_compound'].agg(['mean', 'count']).reset_index()
        topic_sentiment = topic_sentiment.sort_values('mean', ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in topic_sentiment['mean']]
        
        bars = ax.barh(range(len(topic_sentiment)), topic_sentiment['mean'], color=colors, alpha=0.7)
        ax.set_yticks(range(len(topic_sentiment)))
        ax.set_yticklabels(topic_sentiment['topic_label'])
        ax.set_xlabel('Average Sentiment Score')
        ax.set_title('Average Sentiment by Topic')
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Add count annotations
        for i, (bar, count) in enumerate(zip(bars, topic_sentiment['count'])):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'n={count}', va='center', fontsize=8)
        
        st.pyplot(fig)
    
    # Correlation analysis
    st.subheader("Feature Correlations")
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 1:
        correlation_matrix = df[numeric_columns].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Feature Correlation Matrix')
        st.pyplot(fig)

if __name__ == "__main__":
    main()
