import os
import subprocess
import sys

def main():
    """
    Run the RAVEN Storytelling Dashboard
    """
    print("Starting RAVEN Storytelling Dashboard...")
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dashboard_file = os.path.join(current_dir, "raven_dash.py")
    
    if not os.path.exists(dashboard_file):
        print(f"Error: Could not find {dashboard_file}")
        return
    
    # Run the dashboard
    try:
        cmd = [sys.executable, "-m", "streamlit", "run", dashboard_file]
        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd)
    except Exception as e:
        print(f"Error running dashboard: {e}")
        print("\nIf required packages are not installed, you can install them with:")
        print("pip install streamlit pandas matplotlib pillow numpy PyPDF2")

if __name__ == "__main__":
    main()

