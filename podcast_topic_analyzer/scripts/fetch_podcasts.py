import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("LISTEN_API_KEY")

if not API_KEY:
    raise ValueError("API key not found. Please check your .env file.")

headers = {"X-ListenAPI-Key": API_KEY}

def fetch_episodes(podcast_id, max_episodes=10):
    """
    Fetch episodes for a given podcast ID from ListenNotes API
    
    Args:
        podcast_id (str): The ID of the podcast
        max_episodes (int): Maximum number of episodes to fetch
    
    Returns:
        list: List of episode data
    """
    url = f"https://listen-api.listennotes.com/api/v2/podcasts/{podcast_id}?sort=recent_first"
    
    try:
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}, {response.text}")
            return []
        
        data = response.json()
        episodes = data.get("episodes", [])[:max_episodes]
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        # Save episodes to JSON file
        with open("data/episodes.json", "w", encoding='utf-8') as f:
            json.dump(episodes, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully saved {len(episodes)} episodes to data/episodes.json")
        return episodes
        
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []

def fetch_multiple_podcasts(podcast_ids, max_episodes_per_podcast=10):
    """
    Fetch episodes from multiple podcasts
    
    Args:
        podcast_ids (list): List of podcast IDs
        max_episodes_per_podcast (int): Max episodes per podcast
    
    Returns:
        list: Combined list of all episodes
    """
    all_episodes = []
    
    for podcast_id in podcast_ids:
        print(f"Fetching episodes for podcast ID: {podcast_id}")
        episodes = fetch_episodes(podcast_id, max_episodes_per_podcast)
        all_episodes.extend(episodes)
    
    # Save combined episodes
    with open("data/all_episodes.json", "w", encoding='utf-8') as f:
        json.dump(all_episodes, f, indent=2, ensure_ascii=False)
    
    print(f"Total episodes collected: {len(all_episodes)}")
    return all_episodes

if __name__ == "__main__":
    # Example usage
    # The Daily podcast ID
    daily_podcast_id = "4d3fe717742d4963a85562e9f84d8c79"
    
    # You can add more podcast IDs here
    podcast_ids = [
        daily_podcast_id,
        # Add more podcast IDs as needed
    ]
    
    # Fetch episodes
    episodes = fetch_multiple_podcasts(podcast_ids, max_episodes_per_podcast=20)
    
    if episodes:
        print("Sample episode data:")
        print(json.dumps(episodes[0], indent=2)[:500] + "...")
