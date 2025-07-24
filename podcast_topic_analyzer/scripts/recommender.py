import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import pickle

class PodcastRecommender:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the podcast recommender
        
        Args:
            model_name (str): Name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.episodes_df = None
        self.similarity_matrix = None
    
    def load_data(self, file_path="data/topics_with_sentiment.csv"):
        """Load episode data with topics and sentiment"""
        try:
            self.episodes_df = pd.read_csv(file_path, encoding='utf-8')
            print(f"Loaded {len(self.episodes_df)} episodes")
            return True
        except FileNotFoundError:
            print(f"File {file_path} not found. Please run previous scripts first.")
            return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def prepare_text_for_embedding(self):
        """Prepare text data for creating embeddings"""
        # Combine relevant text fields
        texts = []
        for _, row in self.episodes_df.iterrows():
            title = str(row.get('title', ''))
            description = str(row.get('description', ''))
            topic_label = str(row.get('topic_label', ''))
            
            # Create a comprehensive text representation
            combined_text = f"{title}. {description}. Topic: {topic_label}"
            texts.append(combined_text)
        
        return texts
    
    def create_embeddings(self, save_path="data/embeddings.pkl"):
        """Create embeddings for all episodes"""
        print("Preparing text for embeddings...")
        texts = self.prepare_text_for_embedding()
        
        print("Creating embeddings...")
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Save embeddings
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(self.embeddings, f)
        
        print(f"Embeddings saved to {save_path}")
        return self.embeddings
    
    def load_embeddings(self, file_path="data/embeddings.pkl"):
        """Load pre-computed embeddings"""
        try:
            with open(file_path, 'rb') as f:
                self.embeddings = pickle.load(f)
            print(f"Loaded embeddings from {file_path}")
            return True
        except FileNotFoundError:
            print(f"Embeddings file not found. Creating new embeddings...")
            self.create_embeddings()
            return True
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            return False
    
    def compute_similarity_matrix(self):
        """Compute cosine similarity matrix"""
        if self.embeddings is None:
            print("Embeddings not found. Creating embeddings...")
            self.create_embeddings()
        
        print("Computing similarity matrix...")
        self.similarity_matrix = cosine_similarity(self.embeddings)
        print("Similarity matrix computed!")
        return self.similarity_matrix
    
    def get_recommendations(self, episode_index, n_recommendations=5, min_similarity=0.1):
        """
        Get recommendations for a specific episode
        
        Args:
            episode_index (int): Index of the episode to get recommendations for
            n_recommendations (int): Number of recommendations to return
            min_similarity (float): Minimum similarity threshold
        
        Returns:
            pd.DataFrame: Recommended episodes with similarity scores
        """
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()
        
        # Get similarity scores for the given episode
        similarities = self.similarity_matrix[episode_index]
        
        # Get indices of most similar episodes (excluding the episode itself)
        similar_indices = np.argsort(similarities)[::-1]
        similar_indices = similar_indices[similar_indices != episode_index]
        
        # Filter by minimum similarity and get top N
        recommendations = []
        for idx in similar_indices:
            similarity_score = similarities[idx]
            if similarity_score >= min_similarity and len(recommendations) < n_recommendations:
                recommendations.append({
                    'index': idx,
                    'similarity_score': similarity_score,
                    'title': self.episodes_df.iloc[idx]['title'],
                    'topic_label': self.episodes_df.iloc[idx].get('topic_label', 'Unknown'),
                    'sentiment_label': self.episodes_df.iloc[idx].get('overall_sentiment_label', 'Unknown'),
                    'sentiment_score': self.episodes_df.iloc[idx].get('overall_sentiment_compound', 0),
                    'pub_date': self.episodes_df.iloc[idx].get('pub_date', 'Unknown')
                })
        
        return pd.DataFrame(recommendations)
    
    def get_recommendations_by_title(self, title_substring, n_recommendations=5):
        """
        Get recommendations for an episode by searching for title
        
        Args:
            title_substring (str): Substring to search for in episode titles
            n_recommendations (int): Number of recommendations to return
        
        Returns:
            dict: Original episode info and recommendations
        """
        # Find episodes matching the title substring
        matches = self.episodes_df[self.episodes_df['title'].str.contains(title_substring, case=False, na=False)]
        
        if matches.empty:
            print(f"No episodes found matching '{title_substring}'")
            return None
        
        # Use the first match
        episode_index = matches.index[0]
        episode_info = matches.iloc[0]
        
        print(f"Found episode: {episode_info['title']}")
        
        # Get recommendations
        recommendations = self.get_recommendations(episode_index, n_recommendations)
        
        return {
            'original_episode': {
                'title': episode_info['title'],
                'topic_label': episode_info.get('topic_label', 'Unknown'),
                'sentiment_label': episode_info.get('overall_sentiment_label', 'Unknown'),
                'sentiment_score': episode_info.get('overall_sentiment_compound', 0),
                'pub_date': episode_info.get('pub_date', 'Unknown')
            },
            'recommendations': recommendations
        }
    
    def get_topic_based_recommendations(self, topic_label, n_recommendations=10):
        """
        Get episodes from a specific topic
        
        Args:
            topic_label (str): Topic label to filter by
            n_recommendations (int): Number of episodes to return
        
        Returns:
            pd.DataFrame: Episodes from the specified topic
        """
        topic_episodes = self.episodes_df[
            self.episodes_df['topic_label'].str.contains(topic_label, case=False, na=False)
        ].head(n_recommendations)
        
        return topic_episodes[['title', 'topic_label', 'overall_sentiment_label', 
                              'overall_sentiment_compound', 'pub_date']]
    
    def save_all_recommendations(self, output_path="data/recommendations.json"):
        """Save recommendations for all episodes"""
        print("Generating recommendations for all episodes...")
        
        all_recommendations = {}
        
        for idx, row in self.episodes_df.iterrows():
            title = row['title']
            recommendations = self.get_recommendations(idx, n_recommendations=3)
            
            all_recommendations[title] = {
                'episode_info': {
                    'title': title,
                    'topic_label': row.get('topic_label', 'Unknown'),
                    'sentiment_label': row.get('overall_sentiment_label', 'Unknown')
                },
                'recommendations': recommendations.to_dict('records') if not recommendations.empty else []
            }
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(self.episodes_df)} episodes")
        
        # Save to JSON
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_recommendations, f, indent=2, ensure_ascii=False)
        
        print(f"All recommendations saved to {output_path}")
        return all_recommendations

def main():
    """Main function to run the recommendation engine"""
    print("Starting Podcast Recommendation Engine...")
    
    # Initialize recommender
    recommender = PodcastRecommender()
    
    # Load data
    if not recommender.load_data():
        return
    
    # Load or create embeddings
    recommender.load_embeddings()
    
    # Compute similarity matrix
    recommender.compute_similarity_matrix()
    
    # Example usage
    print("\n=== Example Recommendations ===")
    
    # Get recommendations for the first episode
    if len(recommender.episodes_df) > 0:
        first_episode_title = recommender.episodes_df.iloc[0]['title']
        print(f"\nRecommendations for: {first_episode_title}")
        
        recommendations = recommender.get_recommendations(0, n_recommendations=5)
        if not recommendations.empty:
            for _, rec in recommendations.iterrows():
                print(f"  - {rec['title']} (Similarity: {rec['similarity_score']:.3f})")
        else:
            print("  No recommendations found")
    
    # Save all recommendations
    recommender.save_all_recommendations()
    
    return recommender

if __name__ == "__main__":
    recommender = main()
