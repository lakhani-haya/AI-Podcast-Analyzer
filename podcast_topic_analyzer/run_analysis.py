import os
import sys
import time
import importlib.util
from datetime import datetime

def print_step(step_num, total_steps, description):
    print(f"\n{'='*50}")
    print(f"STEP {step_num}/{total_steps}: {description}")
    print(f"{'='*50}")

def run_script(script_path, description):
    try:
        print(f"\nRunning: {script_path}")
        
        spec = importlib.util.spec_from_file_location("module", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        print(f" {description} completed!")
        return True
        
    except Exception as e:
        print(f"Error in {description}: {str(e)}")
        return False

def check_dependencies():
    print("Checking dependencies...")
    
    required_packages = [
        'requests', 'pandas', 'matplotlib', 'seaborn', 'scikit-learn',
        'vaderSentiment', 'sentence_transformers', 'bertopic', 'streamlit',
        'python-dotenv', 'beautifulsoup4'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install " + " ".join(missing_packages))
        return False
    
    print("All dependencies installed!")
    return True

def check_api_key():
    env_path = ".env"
    if not os.path.exists(env_path):
        print(" .env file not found!")
        return False
    
    with open(env_path, 'r') as f:
        content = f.read()
        if "LISTEN_API_KEY" not in content:
            print("LISTEN_API_KEY not found in .env file!")
            return False
    
    print("API key found!")
    return True

def main():
    print("AI PODCAST TOPIC ANALYZER & RECOMMENDER")
    print("Starting analysis pipeline...")
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    print_step(0, 6, "CHECKING PREREQUISITES")
    
    if not check_dependencies():
        return False
    
    if not check_api_key():
        return False
    
    steps = [
        {
            'script': 'scripts/fetch_podcasts.py',
            'description': 'Fetching Podcast Data'
        },
        {
            'script': 'scripts/extract_topics.py',
            'description': 'Extracting Topics'
        },
        {
            'script': 'scripts/sentiment_analysis.py',
            'description': 'Analyzing Sentiment'
        },
        {
            'script': 'scripts/recommender.py',
            'description': 'Building Recommendations'
        },
        {
            'script': 'scripts/visualize_data.py',
            'description': 'Creating Visualizations'
        }
    ]
    
    total_steps = len(steps) + 1
    start_time = time.time()
    
    for i, step in enumerate(steps, 1):
        print_step(i, total_steps, step['description'].upper())
        
        step_start = time.time()
        success = run_script(step['script'], step['description'])
        step_duration = time.time() - step_start
        
        if success:
            print(f" Step completed in {step_duration:.2f} seconds")
        else:
            print(f" Step failed after {step_duration:.2f} seconds")
            
            continue_choice = input("Continue with remaining steps? (y/n): ")
            if continue_choice.lower() != 'y':
                print("Pipeline stopped.")
                return False
    
    print_step(total_steps, total_steps, "ANALYSIS COMPLETE")
    
    total_duration = time.time() - start_time
    print(f" Total time: {total_duration:.2f} seconds")
    
    print("\n RESULTS:")
    print("Episode data: data/episodes.json")
    print("Topics: data/topics.csv")
    print("Sentiment: data/topics_with_sentiment.csv")
    print("✅ Recommendations: data/recommendations.json")
    print("✅ Visualizations: visualizations/")
    
    print("\n NEXT STEPS:")
    print("Run the web app: cd app && streamlit run streamlit_app.py")
    
    start_streamlit = input("\nStart Streamlit app now? (y/n): ")
    if start_streamlit.lower() == 'y':
        try:
            os.chdir('app')
            os.system('streamlit run streamlit_app.py')
        except:
            print("You can start it manually: cd app && streamlit run streamlit_app.py")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
