#!/usr/bin/env python
"""
Direct test for debugging streaming content problems.
This script will:
1. Make a streaming API request 
2. Check for a process using that PID
3. Directly check the streaming_content_buffer
4. Check the process_outputs
"""

import os
import sys
import time
import json
import logging
import urllib.request
import urllib.parse
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Base URL for the API server
BASE_URL = "http://localhost:22434"

def make_streaming_request():
    """Make a streaming request and return the process ID"""
    logger.info("Making streaming request...")
    
    # Use curl to make a streaming request since we can't easily handle streaming in Python's stdlib
    curl_cmd = [
        "curl", "-s", f"{BASE_URL}/api/chat", 
        "-H", "Content-Type: application/json", 
        "-d", '{"model":"claude-3.7-sonnet","messages":[{"role":"user","content":"Write a short poem about debugging"}],"stream":true}'
    ]
    
    # Execute the curl command
    process = subprocess.Popen(curl_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait a moment for the request to be processed
    time.sleep(5)
    logger.info("Streaming request initiated")
    
    # Get recent process IDs
    processes_url = f"{BASE_URL}/process_outputs"
    try:
        with urllib.request.urlopen(processes_url) as response:
            processes_data = json.loads(response.read().decode('utf-8'))
            processes = processes_data.get("outputs", [])
            
            if not processes:
                logger.error("No processes found")
                return None
                
            # Get the most recent process
            recent_process = processes[0]
            pid = recent_process.get("pid")
            logger.info(f"Found recent process with PID: {pid}")
            return pid
    except Exception as e:
        logger.error(f"Error getting process list: {e}")
        return None

def check_process_output(pid):
    """Check the process output directly"""
    output_url = f"{BASE_URL}/process_output/{pid}"
    logger.info(f"Checking process output at {output_url}")
    
    try:
        with urllib.request.urlopen(output_url) as response:
            html_content = response.read().decode('utf-8')
            
            # Save the HTML for debugging
            with open("debug_output.html", "w") as f:
                f.write(html_content)
                logger.info(f"Saved process output HTML to debug_output.html")
            
            # Check for streaming content
            start_marker = '<pre id="streaming-content" class="streaming-content-box">'
            end_marker = '</pre>'
            start_pos = html_content.find(start_marker) + len(start_marker)
            end_pos = html_content.find(end_marker, start_pos)
            
            if start_pos > 0 and end_pos > start_pos:
                content = html_content[start_pos:end_pos].strip()
                logger.info(f"Streaming content: {content[:100]}...")
                
                # Check if it's just the placeholder
                if "Streaming content will appear here" in content:
                    logger.error("ISSUE: Found placeholder message instead of actual content")
                    return False
                else:
                    logger.info("SUCCESS: Found actual streaming content")
                    return True
            else:
                logger.error("Could not extract streaming content")
                return False
    except Exception as e:
        logger.error(f"Error checking process output: {e}")
        return False

def check_buffer_directly(pid):
    """Make a direct request to check buffer status"""
    buffer_check_url = f"{BASE_URL}/debug/buffer_check?pid={pid}"
    logger.info(f"Attempting to check buffer directly at {buffer_check_url}")
    
    try:
        with urllib.request.urlopen(buffer_check_url) as response:
            result = json.loads(response.read().decode('utf-8'))
            logger.info(f"Buffer check result: {result}")
            return result
    except Exception as e:
        logger.error(f"Error checking buffer directly: {e}")
        return {"error": str(e)}

def main():
    """Run the full test"""
    pid = make_streaming_request()
    if not pid:
        logger.error("Failed to get PID")
        return 1
    
    # Wait for content to be generated
    logger.info("Waiting for content generation...")
    time.sleep(5)
    
    # Check the process output
    success = check_process_output(pid)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())