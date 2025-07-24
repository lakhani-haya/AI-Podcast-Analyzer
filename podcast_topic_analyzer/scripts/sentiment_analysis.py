import pandas as pd
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import re
from bs4 import BeautifulSoup

def clean_text(text):
    """Clean HTML tags and extra whitespace from text"""
    if not text:
        return ""
    
    # Remove HTML tags
    try:
        text = BeautifulSoup(text, "html.parser").get_text()
    except:
        pass
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def analyze_sentiment(text):
    """Analyze sentiment using VADER"""
    analyzer = SentimentIntensityAnalyzer()
    
    # Clean the text
    clean_text_content = clean_text(text)
    
    if not clean_text_content:
        return {
            'compound': 0,
            'positive': 0,
            'neutral': 1,
            'negative': 0,
            'sentiment_label': 'neutral'
        }
    
    # Get sentiment scores
    scores = analyzer.polarity_scores(clean_text_content)
    
    # Determine sentiment label based on compound score
    compound = scores['compound']
    if compound >= 0.05:
        sentiment_label = 'positive'
    elif compound <= -0.05:
        sentiment_label = 'negative'
    else:
        sentiment_label = 'neutral'
    
    return {
        'compound': scores['compound'],
        'positive': scores['pos'],
        'neutral': scores['neu'],
        'negative': scores['neg'],
        'sentiment_label': sentiment_label
    }

def load_topics_data(file_path="data/topics.csv"):
    """Load the topics data from CSV"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        return df
    except FileNotFoundError:
        print(f"File {file_path} not found. Please run extract_topics.py first.")
        return None
    except Exception as e:
        print(f"Error loading topics data: {e}")
        return None

def add_sentiment_analysis(df):
    """Add sentiment analysis to the dataframe"""
    
    print("Analyzing sentiment for episodes...")
    
    # Initialize lists to store sentiment data
    sentiment_data = []
    
    for index, row in df.iterrows():
        # Combine title and description for sentiment analysis
        title = str(row.get('title', ''))
        description = str(row.get('description', ''))
        
        # Analyze sentiment for title
        title_sentiment = analyze_sentiment(title)
        
        # Analyze sentiment for description
        desc_sentiment = analyze_sentiment(description)
        
        # Analyze combined text
        combined_text = f"{title} {description}"
        combined_sentiment = analyze_sentiment(combined_text)
        
        # Store sentiment data
        sentiment_data.append({
            'title_sentiment_compound': title_sentiment['compound'],
            'title_sentiment_label': title_sentiment['sentiment_label'],
            'description_sentiment_compound': desc_sentiment['compound'],
            'description_sentiment_label': desc_sentiment['sentiment_label'],
            'overall_sentiment_compound': combined_sentiment['compound'],
            'overall_sentiment_positive': combined_sentiment['positive'],
            'overall_sentiment_neutral': combined_sentiment['neutral'],
            'overall_sentiment_negative': combined_sentiment['negative'],
            'overall_sentiment_label': combined_sentiment['sentiment_label']
        })
        
        if (index + 1) % 10 == 0:
            print(f"Processed {index + 1}/{len(df)} episodes")
    
    # Add sentiment columns to dataframe
    sentiment_df = pd.DataFrame(sentiment_data)
    df_with_sentiment = pd.concat([df, sentiment_df], axis=1)
    
    return df_with_sentiment

def analyze_sentiment_patterns(df):
    """Analyze sentiment patterns in the data"""
    
    print("\n=== Sentiment Analysis Summary ===")
    
    # Overall sentiment distribution
    print("Overall Sentiment Distribution:")
    sentiment_counts = df['overall_sentiment_label'].value_counts()
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {sentiment.capitalize()}: {count} ({percentage:.1f}%)")
    
    # Average sentiment scores
    avg_compound = df['overall_sentiment_compound'].mean()
    print(f"\nAverage Sentiment Score: {avg_compound:.3f}")
    
    # Sentiment by topic (if available)
    if 'topic_label' in df.columns:
        print("\nSentiment by Topic:")
        topic_sentiment = df.groupby('topic_label')['overall_sentiment_compound'].agg(['mean', 'count']).round(3)
        topic_sentiment = topic_sentiment.sort_values('mean', ascending=False)
        print(topic_sentiment.head(10))
    
    # Most positive and negative episodes
    most_positive = df.loc[df['overall_sentiment_compound'].idxmax()]
    most_negative = df.loc[df['overall_sentiment_compound'].idxmin()]
    
    print(f"\nMost Positive Episode:")
    print(f"  Title: {most_positive['title']}")
    print(f"  Sentiment Score: {most_positive['overall_sentiment_compound']:.3f}")
    
    print(f"\nMost Negative Episode:")
    print(f"  Title: {most_negative['title']}")
    print(f"  Sentiment Score: {most_negative['overall_sentiment_compound']:.3f}")

def save_sentiment_data(df, output_path="data/topics_with_sentiment.csv"):
    """Save the dataframe with sentiment analysis"""
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\nData with sentiment analysis saved to {output_path}")
    
    return output_path

def main():
    """Main function to run sentiment analysis"""
    print("Starting sentiment analysis...")
    
    # Load topics data
    df = load_topics_data()
    if df is None:
        return None
    
    print(f"Loaded {len(df)} episodes with topics")
    
    # Add sentiment analysis
    df_with_sentiment = add_sentiment_analysis(df)
    
    # Analyze patterns
    analyze_sentiment_patterns(df_with_sentiment)
    
    # Save results
    output_path = save_sentiment_data(df_with_sentiment)
    
    return df_with_sentiment

if __name__ == "__main__":
    df_result = main()
