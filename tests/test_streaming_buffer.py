#!/usr/bin/env python
"""
Test script to verify streaming content buffer functionality.
"""

import sys
import logging
import time
import json
import argparse
import urllib.request
import urllib.error
import urllib.parse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_streaming_buffer():
    """Test if streaming content is properly stored and displayed in the dashboard."""
    parser = argparse.ArgumentParser(description='Test streaming content buffer')
    parser.add_argument('--server-url', type=str, default='http://localhost:22434',
                        help='URL of the Claude API server')
    args = parser.parse_args()
    
    base_url = args.server_url
    
    # Step 1: Make a streaming request using curl since streaming is hard with urllib
    import subprocess
    
    logger.info("Making streaming request using a simple test request...")
    # Let's make a simple curl request to create a streaming process
    curl_cmd = [
        "curl", "-s", f"{base_url}/api/chat", 
        "-H", "Content-Type: application/json", 
        "-d", '{"model":"claude-3.7-sonnet","messages":[{"role":"user","content":"Write a one-line poem"}],"stream":true}'
    ]
    
    try:
        # Run curl but don't wait for it to finish
        subprocess.Popen(curl_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info("Streaming request initiated")
    except Exception as e:
        logger.error(f"Failed to make streaming request: {e}")
        return False
    
    # Wait for process to be registered
    time.sleep(5)
    logger.info("Checking dashboard for recent processes...")
    
    # Step 2: Get recent processes from dashboard
    processes_url = f"{base_url}/process_outputs"
    try:
        with urllib.request.urlopen(processes_url) as response:
            processes_data = json.loads(response.read().decode('utf-8'))
            processes = processes_data.get("outputs", [])
            
            if not processes:
                logger.error("No processes found in dashboard")
                return False
                
            # Get the most recent process
            recent_process = processes[0]
            pid = recent_process.get("pid")
            logger.info(f"Found recent process with PID: {pid}")
    except Exception as e:
        logger.error(f"Failed to get process list: {e}")
        return False
    
    # Step 3: Fetch the process output
    output_url = f"{base_url}/process_output/{pid}"
    logger.info(f"Fetching process output from {output_url}...")
    
    try:
        with urllib.request.urlopen(output_url) as response:
            html_content = response.read().decode('utf-8')
            
            # Simple check for placeholder text
            if "Streaming response - output sent directly to client" in html_content:
                logger.error("ISSUE DETECTED: Dashboard still showing placeholder text")
                return False
            
            # Check for some kind of content
            start_marker = '<pre id="streaming-content" class="streaming-content-box">'
            end_marker = '</pre>'
            start_pos = html_content.find(start_marker) + len(start_marker)
            end_pos = html_content.find(end_marker, start_pos)
            
            if start_pos > 0 and end_pos > start_pos:
                actual_content = html_content[start_pos:end_pos].strip()
                logger.warning(f"Content found in streaming box: {actual_content[:100]}...")
                
                # Write the full HTML to a file for inspection
                with open("debug_dashboard.html", "w") as f:
                    f.write(html_content)
                    logger.warning(f"Wrote full HTML to debug_dashboard.html")
                
                if len(actual_content) > 20 and actual_content != "No streaming content available":
                    # If it's just the placeholder message, that's not real success
                    if "Streaming content will appear here" in actual_content:
                        logger.error("FAILURE: Dashboard is only showing placeholder content")
                        return False
                    else:
                        logger.warning("SUCCESS: Dashboard contains ACTUAL streaming content")
                        return True
                else:
                    logger.error("Content looks like a placeholder or is too short")
                    return False
            else:
                logger.error("Couldn't find streaming content section in HTML")
                return False
    except Exception as e:
        logger.error(f"Failed to get process output: {e}")
        return False

if __name__ == "__main__":
    success = test_streaming_buffer()
    sys.exit(0 if success else 1)