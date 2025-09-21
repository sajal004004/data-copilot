"""
Startup Script for Data Copilot MVP Frontend
Streamlit application with error handling and configuration
"""
import streamlit as st
import subprocess
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config import config

def main():
    """Start the Streamlit frontend"""
    
    print("🚀 Starting Data Copilot MVP Frontend...")
    print(f"📍 Frontend will run on http://{config.API_HOST}:{config.STREAMLIT_PORT}")
    print(f"🔗 Backend API: http://{config.API_HOST}:{config.API_PORT}")
    
    try:
        # Change to frontend directory
        frontend_dir = Path(__file__).parent / "frontend"
        os.chdir(frontend_dir)
        
        # Start Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", str(config.STREAMLIT_PORT),
            "--server.address", config.API_HOST,
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ])
        
    except KeyboardInterrupt:
        print("👋 Frontend stopped by user")
    except Exception as e:
        print(f"❌ Frontend failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()