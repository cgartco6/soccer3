import os
import subprocess
import sys

def install_requirements():
    print("Installing optimized requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def create_files():
    # Create necessary directories
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # Create .env file
    with open('.env', 'w') as f:
        f.write("""# Free API Keys - Get from:
# https://the-odds-api.com (500 free requests/month)
# https://thesportsapi.com (free tier)

ODDS_API_KEY=your_free_key_here
THE_SPORTS_API_KEY=your_free_key_here
""")
    
    print("Created .env file - please add your free API keys")

def main():
    print("Installing Sports Betting AI Platform...")
    install_requirements()
    create_files()
    
    print("\n✅ Installation Complete!")
    print("\nNext steps:")
    print("1. Edit .env file with your free API keys")
    print("2. Run: python app.py")
    print("3. Open http://localhost:5000")
    print("\n💡 Get free API keys from:")
    print("   - https://the-odds-api.com")
    print("   - https://thesportsapi.com")

if __name__ == "__main__":
    main()
