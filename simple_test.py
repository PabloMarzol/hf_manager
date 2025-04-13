# E:\INMAV\Trading\Algorithmic Devp\Quantitative_Investment\mixgo\simple_test.py

import os
import sys

# Add the project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Print the Python path
print("Python path:")
for path in sys.path:
    print(f"  {path}")

# Print the current directory structure
print("\nCurrent directory structure:")
for root, dirs, files in os.walk(project_root):
    level = root.replace(project_root, '').count(os.sep)
    indent = ' ' * 4 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 4 * (level + 1)
    for f in files:
        print(f"{subindent}{f}")

# Try importing a simple module
print("\nTrying to import a module...")
try:
    # Replace with path to one of your most basic modules that should exist
    from signals.data.models import Position
    print("Success! Position class imported.")
    print(f"Position class info: {Position}")
except ImportError as e:
    print(f"Import failed: {e}")