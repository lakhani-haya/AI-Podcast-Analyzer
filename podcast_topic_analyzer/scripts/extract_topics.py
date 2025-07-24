import json
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import numpy as np
import os

def load_episodes(file_path="data/episodes.json"):
    """Load episodes from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            episodes = json.load(f)
        return episodes
    except FileNotFoundError:
        print(f"File {file_path} not found. Please run fetch_podcasts.py first.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return []

def extract_text_data(episodes):
    """Extract relevant text data from episodes for topic modeling"""
    documents = []
    episode_info = []
    
    for episode in episodes:
        # Combine title and description for better topic modeling
        title = episode.get('title', '')
        description = episode.get('description', '')
        
        # Clean and combine text
        text = f"{title}. {description}".strip()
        
        if text and len(text) > 10:  # Only include episodes with meaningful text
            documents.append(text)
            episode_info.append({
                'id': episode.get('id', ''),
                'title': title,
                'description': description,
                'pub_date_ms': episode.get('pub_date_ms', 0),
                'audio_length_sec': episode.get('audio_length_sec', 0),
                'link': episode.get('link', '')
            })
    
    return documents, episode_info

def extract_topics(documents, n_topics=10):
    """Extract topics using BERTopic"""
    print("Loading sentence transformer model...")
    
    # Use a lighter model for faster processing
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Initializing BERTopic...")
    # Initialize BERTopic with custom settings
    topic_model = BERTopic(
        embedding_model=embedding_model,
        nr_topics=n_topics,
        verbose=True,
        calculate_probabilities=True
    )
    
    print(f"Fitting topic model on {len(documents)} documents...")
    # Fit the model and transform documents
    topics, probabilities = topic_model.fit_transform(documents)
    
    print("Topic modeling completed!")
    
    return topic_model, topics, probabilities

def create_topics_dataframe(episode_info, topics, probabilities, topic_model):
    """Create a comprehensive DataFrame with topics and episode information"""
    
    # Create DataFrame
    df = pd.DataFrame(episode_info)
    df['topic'] = topics
    df['topic_probability'] = [prob.max() if isinstance(prob, np.ndarray) else prob for prob in probabilities]
    
    # Add topic labels
    topic_labels = {}
    topic_info = topic_model.get_topic_info()
    
    for _, row in topic_info.iterrows():
        topic_id = row['Topic']
        # Get top words for the topic
        if topic_id >= 0:  # Skip outlier topic (-1)
            topic_words = topic_model.get_topic(topic_id)
            top_words = [word for word, _ in topic_words[:3]]
            topic_labels[topic_id] = "_".join(top_words)
        else:
            topic_labels[topic_id] = "outlier"
    
    df['topic_label'] = df['topic'].map(topic_labels)
    
    # Convert timestamp to datetime
    df['pub_date'] = pd.to_datetime(df['pub_date_ms'], unit='ms')
    
    return df

def save_topics_results(df, topic_model, output_dir="data"):
    """Save topic modeling results"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save DataFrame
    csv_path = os.path.join(output_dir, "topics.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"Topics data saved to {csv_path}")
    
    # Save topic information
    topic_info = topic_model.get_topic_info()
    topic_info_path = os.path.join(output_dir, "topic_info.csv")
    topic_info.to_csv(topic_info_path, index=False)
    print(f"Topic information saved to {topic_info_path}")
    
    # Save detailed topic words
    topics_details = {}
    for topic_id in topic_info['Topic'].values:
        if topic_id >= 0:
            topics_details[topic_id] = topic_model.get_topic(topic_id)
    
    with open(os.path.join(output_dir, "topic_words.json"), 'w', encoding='utf-8') as f:
        json.dump(topics_details, f, indent=2, ensure_ascii=False)
    
    return df

def main():
    """Main function to run topic extraction"""
    print("Starting topic extraction...")
    
    # Load episodes
    episodes = load_episodes()
    if not episodes:
        return
    
    print(f"Loaded {len(episodes)} episodes")
    
    # Extract text data
    documents, episode_info = extract_text_data(episodes)
    print(f"Extracted text from {len(documents)} episodes")
    
    if len(documents) < 5:
        print("Not enough documents for topic modeling. Need at least 5 episodes.")
        return
    
    # Extract topics
    topic_model, topics, probabilities = extract_topics(documents)
    
    # Create DataFrame
    df = create_topics_dataframe(episode_info, topics, probabilities, topic_model)
    
    # Save results
    final_df = save_topics_results(df, topic_model)
    
    # Print summary
    print("\n=== Topic Extraction Summary ===")
    print(f"Total episodes processed: {len(final_df)}")
    print(f"Number of topics found: {len(final_df['topic'].unique())}")
    print("\nTopic distribution:")
    print(final_df['topic_label'].value_counts().head(10))
    
    return final_df, topic_model

if __name__ == "__main__":
    df, model = main()
