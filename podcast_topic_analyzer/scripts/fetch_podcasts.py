import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("LISTEN_API_KEY")

if not API_KEY:
    raise ValueError("API key not found. Please check your .env file.")

headers = {"X-ListenAPI-Key": API_KEY}

def fetch_episodes(podcast_id, max_episodes=10):
    url = f"https://listen-api.listennotes.com/api/v2/podcasts/{podcast_id}?sort=recent_first"
    
    try:
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}, {response.text}")
            return []
        
        data = response.json()
        episodes = data.get("episodes", [])[:max_episodes]
        
        os.makedirs("data", exist_ok=True)
        
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
    all_episodes = []
    
    for podcast_id in podcast_ids:
        print(f"Fetching episodes for podcast ID: {podcast_id}")
        episodes = fetch_episodes(podcast_id, max_episodes_per_podcast)
        all_episodes.extend(episodes)
    
    with open("data/all_episodes.json", "w", encoding='utf-8') as f:
        json.dump(all_episodes, f, indent=2, ensure_ascii=False)
    
    print(f"Total episodes collected: {len(all_episodes)}")
    return all_episodes

if __name__ == "__main__":
    daily_podcast_id = "4d3fe717742d4963a85562e9f84d8c79"
    
    podcast_ids = [
        daily_podcast_id,
    ]
    
    episodes = fetch_multiple_podcasts(podcast_ids, max_episodes_per_podcast=20)
    
    if episodes:
        print("Sample episode data:")
        print(json.dumps(episodes[0], indent=2)[:500] + "...")
