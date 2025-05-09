"""
DZ-MoneyMind Application Launcher

This script sets up the application environment and launches the Streamlit app.
"""

import os
import sys
import streamlit as st
import subprocess

def main():
    """Setup the environment and launch the Streamlit app"""
    
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)  # Move up one directory
    
    # Add the project root to Python path to simplify imports
    if project_dir not in sys.path:
        sys.path.append(project_dir)
    
    # Check if required files exist
    required_files = [
        os.path.join(project_dir, 'data_filtering.py'),
        os.path.join(project_dir, 'money_spending_problem.py'),
        os.path.join(project_dir, 'search_algorithm.py'),
        os.path.join(project_dir, 'test_app.py')
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"ERROR: Required file not found: {file_path}")
            sys.exit(1)
    
    # Run the Streamlit app
    cmd = [sys.executable, "-m", "streamlit", "run", os.path.join(project_dir, "test_app.py")]
    subprocess.run(cmd)

if __name__ == "__main__":
    main()