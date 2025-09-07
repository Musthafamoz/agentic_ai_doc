"""
Simple script to run both FastAPI backend and Streamlit frontend
"""
import subprocess
import threading
import time
import sys
import os

def run_fastapi():
    """Run the FastAPI backend server"""
    print("🚀 Starting FastAPI backend on http://localhost:8000")
    subprocess.run([sys.executable, "main.py"])

def run_streamlit():
    """Run the Streamlit frontend"""
    print("🎨 Starting Streamlit frontend on http://localhost:8501")
    # Wait a bit for FastAPI to start
    time.sleep(3)
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port=8501"])

def main():
    """Run both servers concurrently"""
    print("🏛️ Court Document Processor - Starting both servers...")
    print("=" * 60)
    
    # Create threads for both servers
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    streamlit_thread = threading.Thread(target=run_streamlit, daemon=True)
    
    # Start both servers
    fastapi_thread.start()
    streamlit_thread.start()
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down servers...")
        print("Goodbye! 👋")

if __name__ == "__main__":
    main()