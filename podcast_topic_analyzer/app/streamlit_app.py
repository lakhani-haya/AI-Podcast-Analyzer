import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
import sys

# Add the scripts directory to the path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

try:
    from recommender import PodcastRecommender
except ImportError:
    st.error("Could not import recommender module. Please ensure all dependencies are installed.")

# Set page config
st.set_page_config(
    page_title="AI Podcast Analyzer",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .episode-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load podcast data with caching"""
    try:
        df = pd.read_csv("../data/topics_with_sentiment.csv", encoding='utf-8')
        if 'pub_date' in df.columns:
            df['pub_date'] = pd.to_datetime(df['pub_date'])
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please run the analysis scripts first.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def load_recommendations():
    """Load recommendations data"""
    try:
        with open("../data/recommendations.json", 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error loading recommendations: {e}")
        return None

def display_episode_card(episode_data, show_similarity=False):
    """Display an episode in a card format"""
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**{episode_data.get('title', 'No Title')}**")
            
            # Topic and sentiment badges
            col_topic, col_sentiment = st.columns(2)
            with col_topic:
                topic = episode_data.get('topic_label', 'Unknown')
                st.markdown(f"🏷️ **Topic:** {topic}")
            
            with col_sentiment:
                sentiment = episode_data.get('overall_sentiment_label', 'Unknown')
                sentiment_score = episode_data.get('overall_sentiment_compound', 0)
                
                if sentiment == 'positive':
                    emoji = "😊"
                elif sentiment == 'negative':
                    emoji = "😞"
                else:
                    emoji = "😐"
                
                st.markdown(f"{emoji} **Sentiment:** {sentiment} ({sentiment_score:.3f})")
        
        with col2:
            if show_similarity and 'similarity_score' in episode_data:
                similarity = episode_data['similarity_score']
                st.metric("Similarity", f"{similarity:.3f}")
            
            if 'pub_date' in episode_data:
                pub_date = episode_data['pub_date']
                if pub_date and pub_date != 'Unknown':
                    try:
                        date_obj = pd.to_datetime(pub_date)
                        st.markdown(f"📅 {date_obj.strftime('%Y-%m-%d')}")
                    except:
                        st.markdown(f"📅 {pub_date}")

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">🎧 AI Podcast Topic Analyzer & Recommender</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    recommendations_data = load_recommendations()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "📊 Dashboard", 
        "🔍 Episode Explorer", 
        "💡 Recommendations", 
        "📈 Analytics"
    ])
    
    if page == "📊 Dashboard":
        show_dashboard(df)
    elif page == "🔍 Episode Explorer":
        show_episode_explorer(df)
    elif page == "💡 Recommendations":
        show_recommendations(df, recommendations_data)
    elif page == "📈 Analytics":
        show_analytics(df)

def show_dashboard(df):
    """Show main dashboard"""
    st.header("📊 Podcast Analysis Dashboard")
    
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
