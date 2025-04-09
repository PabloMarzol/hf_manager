#!/usr/bin/env python
"""
Setup script for the trading system.
This script sets up the environment and creates necessary directories.
"""

import os
import sys
import shutil
from pathlib import Path
import argparse

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Set up the trading system environment.")
    
    parser.add_argument("--install-deps", action="store_true", 
                      help="Install dependencies from requirements.txt")
    parser.add_argument("--create-env", action="store_true",
                      help="Create a .env file template")
    parser.add_argument("--reset", action="store_true",
                      help="Reset the environment (clear cache, logs, etc.)")
    
    return parser.parse_args()

def create_directories():
    """Create necessary directories."""
    directories = [
        "logs",
        "backtest_results",
        "live_trading_results",
        "analysis",
        "data/cache"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")


def create_env_template():
    """Create a .env file template."""
    env_template = """# API Keys
FINANCIAL_DATASETS_API_KEY=your_financial_api_key
GROQ_API_KEY=your_groq_api_key

# Optional API Keys (if using other models)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_gemini_key
DEEPSEEK_API_KEY=your_deepseek_key

# Optional Settings
LOG_LEVEL=INFO
"""
    
    env_path = Path(".env")
    if env_path.exists():
        overwrite = input(".env file already exists. Overwrite? (y/n): ")
        if overwrite.lower() != "y":
            print("Skipping .env creation")
            return
    
    with open(env_path, "w") as f:
        f.write(env_template)
    
    print("Created .env template file. Please edit it with your API keys.")

def install_dependencies():
    """Install dependencies from requirements.txt."""
    print("Installing dependencies...")
    
    try:
        import pip
        pip.main(["install", "-r", "requirements.txt"])
        print("Dependencies installed successfully.")
    except Exception as e:
        print(f"Error installing dependencies: {e}")
        print("Please run: pip install -r requirements.txt")

def reset_environment():
    """Reset the environment by clearing cache, logs, etc."""
    dirs_to_clean = [
        "logs",
        "backtest_results",
        "live_trading_results",
        "analysis",
        "data/cache"
    ]
    
    confirmation = input("This will delete all cached data, logs, and results. Continue? (y/n): ")
    if confirmation.lower() != "y":
        print("Reset cancelled")
        return
    
    for directory in dirs_to_clean:
        dir_path = Path(directory)
        if dir_path.exists():
            # Remove all files in directory but keep the directory
            for file_path in dir_path.glob("*"):
                if file_path.is_file():
                    file_path.unlink()
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
            print(f"Cleaned directory: {directory}")
    
    print("Environment reset complete")

def main():
    """Main function."""
    args = parse_arguments()
    
    # Create directories
    create_directories()
    
    # Process arguments
    if args.create_env:
        create_env_template()
    
    if args.install_deps:
        install_dependencies()
    
    if args.reset:
        reset_environment()
    
    print("Setup complete!")
    print("\nNext steps:")
    print("1. Edit .env file with your API keys")
    print("2. Customize your strategy in config.py")
    print("3. Run a backtest: python run_trading_system.py backtest")
    
    # Check if any command-line arguments were provided
    if not any([args.create_env, args.install_deps, args.reset]):
        print("\nTip: Run with --install-deps to install dependencies or --create-env to create an .env template")

if __name__ == "__main__":
    main()