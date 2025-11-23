import os
import sys
import subprocess

def install_requirements():
    print("Installing packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Packages installed")
        return True
    except subprocess.CalledProcessError:
        print("Failed to install packages")
        return False

def create_env_file():
    if os.path.exists(".env"):
        print(" .env file already exists")
        return True
    
    print("Creating .env file...")
    api_key = input("Enter your ListenNotes API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided")
        return False
    
    with open(".env", "w") as f:
        f.write(f"LISTEN_API_KEY={api_key}\n")
    
    print("✅ .env file created")
    return True

def main():
    print("🚀 Setting up AI Podcast Analyzer...")
    
    if not install_requirements():
        return
    
    if not create_env_file():
        return
    
    print("\n🎉 Setup complete!")
    print("Next steps:")
    print("1. Run: python run_analysis.py")
    print("2. Start app: streamlit run app/streamlit_app.py")

if __name__ == "__main__":
    main()
