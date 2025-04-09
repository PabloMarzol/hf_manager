"""
Visualization utilities for the trading system.
"""

import os
from pathlib import Path
import graphviz

def save_graph_as_png(graph, file_path="agent_graph.png"):
    """
    Save a langgraph StateGraph object as a PNG file.
    
    Args:
        graph: The langgraph.StateGraph object
        file_path (str): Path to save the PNG file
    """
    try:
        # Get the dot representation of the graph
        dot = graph.get_graph()
        
        # Create a graphviz.Source object
        src = graphviz.Source(dot)
        
        # Make sure the directory exists
        output_dir = os.path.dirname(file_path)
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Render the graph to a PNG file
        src.render(outfile=file_path, format="png", cleanup=True)
        
        print(f"Graph saved to {file_path}")
    except Exception as e:
        print(f"Error saving graph: {e}")