#!/usr/bin/env python3
"""
Utility script to get the timeout settings from the main server.
This allows all test scripts to use a consistent timeout value.
"""
import os
import sys
import json

# Add the parent directory to the path so we can import modules from the project
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

def get_timeouts():
    """Get the timeout values from the main server file."""
    try:
        # Try to import the constants directly from the server module
        import claude_ollama_server
        
        # Extract the timeout values
        timeouts = {
            "chunk_timeout": claude_ollama_server.CLAUDE_STREAM_CHUNK_TIMEOUT,
            "max_silence": claude_ollama_server.CLAUDE_STREAM_MAX_SILENCE
        }
        
        return timeouts
    except ImportError:
        # If we can't import directly, parse the file to extract the values
        server_file = os.path.join(parent_dir, "claude_ollama_server.py")
        
        chunk_timeout = None
        max_silence = None
        
        with open(server_file, 'r') as f:
            for line in f:
                if "CLAUDE_STREAM_CHUNK_TIMEOUT" in line and "=" in line:
                    try:
                        chunk_timeout = float(line.split("=")[1].split("#")[0].strip())
                    except:
                        pass
                elif "CLAUDE_STREAM_MAX_SILENCE" in line and "=" in line:
                    try:
                        max_silence = float(line.split("=")[1].split("#")[0].strip())
                    except:
                        pass
        
        return {
            "chunk_timeout": chunk_timeout,
            "max_silence": max_silence
        }

if __name__ == "__main__":
    timeouts = get_timeouts()
    print(json.dumps(timeouts, indent=2))