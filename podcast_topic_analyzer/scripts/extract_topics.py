import json
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import numpy as np
import os

def load_episodes(file_path="data/episodes.json"):
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
    documents = []
    episode_info = []
    
    for episode in episodes:
        title = episode.get('title', '')
        description = episode.get('description', '')
        
        text = f"{title}. {description}".strip()
        
        if text and len(text) > 10:
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
    print("Loading sentence transformer model...")
    
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Initializing BERTopic...")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        nr_topics=n_topics,
        verbose=True,
        calculate_probabilities=True
    )
    
    print(f"Fitting topic model on {len(documents)} documents...")
    topics, probabilities = topic_model.fit_transform(documents)
    
    print("Topic modeling completed!")
    
    return topic_model, topics, probabilities

def create_topics_dataframe(episode_info, topics, probabilities, topic_model):
    df = pd.DataFrame(episode_info)
    df['topic'] = topics
    df['topic_probability'] = [prob.max() if isinstance(prob, np.ndarray) else prob for prob in probabilities]
    
    topic_labels = {}
    topic_info = topic_model.get_topic_info()
    
    for _, row in topic_info.iterrows():
        topic_id = row['Topic']
        if topic_id >= 0:
            topic_words = topic_model.get_topic(topic_id)
            top_words = [word for word, _ in topic_words[:3]]
            topic_labels[topic_id] = "_".join(top_words)
        else:
            topic_labels[topic_id] = "outlier"
    
    df['topic_label'] = df['topic'].map(topic_labels)
    df['pub_date'] = pd.to_datetime(df['pub_date_ms'], unit='ms')
    
    return df

def save_topics_results(df, topic_model, output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, "topics.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"Topics data saved to {csv_path}")
    
    topic_info = topic_model.get_topic_info()
    topic_info_path = os.path.join(output_dir, "topic_info.csv")
    topic_info.to_csv(topic_info_path, index=False)
    print(f"Topic information saved to {topic_info_path}")
    
    topics_details = {}
    for topic_id in topic_info['Topic'].values:
        if topic_id >= 0:
            topics_details[topic_id] = topic_model.get_topic(topic_id)
    
    with open(os.path.join(output_dir, "topic_words.json"), 'w', encoding='utf-8') as f:
        json.dump(topics_details, f, indent=2, ensure_ascii=False)
    
    return df

def main():
    print("Starting topic extraction...")
    
    episodes = load_episodes()
    if not episodes:
        return
    
    print(f"Loaded {len(episodes)} episodes")
    
    documents, episode_info = extract_text_data(episodes)
    print(f"Extracted text from {len(documents)} episodes")
    
    if len(documents) < 5:
        print("Not enough documents for topic modeling. Need at least 5 episodes.")
        return
    
    topic_model, topics, probabilities = extract_topics(documents)
    
    df = create_topics_dataframe(episode_info, topics, probabilities, topic_model)
    
    final_df = save_topics_results(df, topic_model)
    
    print("\n=== Topic Extraction Summary ===")
    print(f"Total episodes processed: {len(final_df)}")
    print(f"Number of topics found: {len(final_df['topic'].unique())}")
    print("\nTopic distribution:")
    print(final_df['topic_label'].value_counts().head(10))
    
    return final_df, topic_model

if __name__ == "__main__":
    df, model = main()
