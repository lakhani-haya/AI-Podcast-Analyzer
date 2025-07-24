"""
Main orchestrator script for the AI Podcast Topic Analyzer & Recommender
This script runs the complete analysis pipeline in the correct order.
"""

import os
import sys
import time
from datetime import datetime

def print_step(step_num, total_steps, description):
    """Print a formatted step indicator"""
    print(f"\n{'='*60}")
    print(f"STEP {step_num}/{total_steps}: {description}")
    print(f"{'='*60}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def run_script(script_path, description):
    """Run a Python script and handle errors"""
    try:
        print(f"\nRunning: {script_path}")
        
        # Change to the script directory
        script_dir = os.path.dirname(script_path)
        if script_dir:
            os.chdir(script_dir)
        
        # Import and run the script
        spec = importlib.util.spec_from_file_location("module", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        print(f"✅ {description} completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error in {description}: {str(e)}")
        return False

def check_dependencies():
    """Check if all required packages are installed"""
    print("Checking dependencies...")
    
    required_packages = [
        'requests', 'pandas', 'matplotlib', 'seaborn', 'scikit-learn',
        'vaderSentiment', 'sentence-transformers', 'bertopic', 'streamlit',
        'python-dotenv', 'beautifulsoup4'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All dependencies are installed!")
    return True

def check_api_key():
    """Check if API key is configured"""
    env_path = ".env"
    if not os.path.exists(env_path):
        print("❌ .env file not found!")
        return False
    
    with open(env_path, 'r') as f:
        content = f.read()
        if "LISTEN_API_KEY" not in content:
            print("❌ LISTEN_API_KEY not found in .env file!")
            return False
    
    print("✅ API key configuration found!")
    return True

def main():
    """Main function to run the complete analysis pipeline"""
    
    print("🎧 AI PODCAST TOPIC ANALYZER & RECOMMENDER")
    print("Starting complete analysis pipeline...")
    
    # Change to the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    print(f"Working directory: {os.getcwd()}")
    
    # Check prerequisites
    print_step(0, 6, "CHECKING PREREQUISITES")
    
    if not check_dependencies():
        print("Please install missing dependencies and try again.")
        return False
    
    if not check_api_key():
        print("Please configure your API key and try again.")
        return False
    
    # Define the analysis steps
    steps = [
        {
            'script': 'scripts/fetch_podcasts.py',
            'description': 'Fetching Podcast Data',
            'info': 'Downloading episode data from ListenNotes API'
        },
        {
            'script': 'scripts/extract_topics.py',
            'description': 'Extracting Topics',
            'info': 'Using BERTopic to identify key topics in episodes'
        },
        {
            'script': 'scripts/sentiment_analysis.py',
            'description': 'Analyzing Sentiment',
            'info': 'Using VADER to analyze sentiment of episode content'
        },
        {
            'script': 'scripts/recommender.py',
            'description': 'Building Recommendation Engine',
            'info': 'Creating embeddings and computing similarities'
        },
        {
            'script': 'scripts/visualize_data.py',
            'description': 'Generating Visualizations',
            'info': 'Creating charts and graphs for insights'
        }
    ]
    
    total_steps = len(steps) + 1  # +1 for final step
    
    # Execute each step
    start_time = time.time()
    
    for i, step in enumerate(steps, 1):
        print_step(i, total_steps, step['description'].upper())
        print(f"📋 {step['info']}")
        
        # Add the import statement here since we need it
        import importlib.util
        
        step_start = time.time()
        success = run_script(step['script'], step['description'])
        step_duration = time.time() - step_start
        
        if success:
            print(f"⏱️  Step completed in {step_duration:.2f} seconds")
        else:
            print(f"⚠️  Step failed after {step_duration:.2f} seconds")
            print("You can continue with the remaining steps or fix the issue and rerun.")
            
            continue_choice = input("Continue with remaining steps? (y/n): ")
            if continue_choice.lower() != 'y':
                print("Pipeline stopped by user.")
                return False
    
    # Final step - instructions for Streamlit
    print_step(total_steps, total_steps, "STARTING STREAMLIT APP (OPTIONAL)")
    print("📋 Setting up interactive web interface")
    
    print("\n🎉 Analysis pipeline completed!")
    
    total_duration = time.time() - start_time
    print(f"⏱️  Total execution time: {total_duration:.2f} seconds")
    
    print("\n📊 RESULTS SUMMARY:")
    print("✅ Episode data fetched and saved to data/episodes.json")
    print("✅ Topics extracted and saved to data/topics.csv")
    print("✅ Sentiment analysis completed and saved to data/topics_with_sentiment.csv")
    print("✅ Recommendations generated and saved to data/recommendations.json")
    print("✅ Visualizations created and saved to visualizations/")
    
    print("\n🚀 NEXT STEPS:")
    print("1. Review the generated visualizations in the visualizations/ folder")
    print("2. Check the data files in the data/ folder")
    print("3. Run the Streamlit app for interactive exploration:")
    print("   cd app")
    print("   streamlit run streamlit_app.py")
    
    # Offer to start Streamlit
    start_streamlit = input("\nWould you like to start the Streamlit app now? (y/n): ")
    if start_streamlit.lower() == 'y':
        print("\nStarting Streamlit app...")
        print("The app will open in your default web browser.")
        print("Press Ctrl+C to stop the app when you're done.")
        
        try:
            os.chdir('app')
            os.system('streamlit run streamlit_app.py')
        except KeyboardInterrupt:
            print("\n\nStreamlit app stopped.")
        except Exception as e:
            print(f"\nError starting Streamlit: {e}")
            print("You can start it manually with: cd app && streamlit run streamlit_app.py")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎧 Thank you for using the AI Podcast Topic Analyzer & Recommender!")
    else:
        print("\n❌ Pipeline completed with errors. Please check the output above.")
        sys.exit(1)
