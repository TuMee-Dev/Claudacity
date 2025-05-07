#!/usr/bin/env python
"""
Test script to verify streaming content is properly displayed in the dashboard.

This script:
1. Makes a streaming API request
2. Gets the process ID
3. Checks the dashboard's process output page
4. Verifies the streaming content is displayed correctly in the streaming-content-box
5. Takes a screenshot of the dashboard for visual verification

Usage:
    python test_dashboard_streaming.py [--server-url http://localhost:22434]
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
import urllib.parse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DashboardStreamingTest:
    """Test class for dashboard streaming verification."""
    
    def __init__(self, base_url="http://localhost:22434"):
        """Initialize the test with the server URL."""
        self.base_url = base_url
        self.pid = None
        self.html_output_path = "dashboard_streaming_test.html"
        
    def make_streaming_request(self):
        """Make a streaming API request and return the process ID."""
        logger.info("Making streaming request...")
        
        # Use curl to make a streaming request since we can't easily handle streaming in Python's stdlib
        curl_cmd = [
            "curl", "-s", f"{self.base_url}/api/chat", 
            "-H", "Content-Type: application/json", 
            "-d", '{"model":"claude-3.7-sonnet","messages":[{"role":"user","content":"Write a detailed paragraph describing a beautiful mountain landscape."}],"stream":true}'
        ]
        
        # Execute the curl command
        process = subprocess.Popen(curl_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for the request to be processed
        time.sleep(3)
        logger.info("Streaming request initiated")
        
        # Get recent process IDs
        processes_url = f"{self.base_url}/process_outputs"
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
                self.pid = pid
                return pid
        except Exception as e:
            logger.error(f"Error getting process list: {e}")
            return None
    
    def check_dashboard_display(self):
        """Check if the streaming content is displayed in the dashboard."""
        if not self.pid:
            logger.error("No PID available to check dashboard")
            return False
            
        process_url = f"{self.base_url}/process_output/{self.pid}"
        logger.info(f"Checking dashboard display at {process_url}")
        
        try:
            # Wait a bit for content to be generated
            time.sleep(5)
            
            # Fetch the process output page
            with urllib.request.urlopen(process_url) as response:
                html_content = response.read().decode('utf-8')
                
                # Save the HTML for inspection
                with open(self.html_output_path, "w") as f:
                    f.write(html_content)
                    logger.info(f"Saved process output HTML to {self.html_output_path}")
                
                # Check for streaming content
                start_marker = '<pre id="streaming-content" class="streaming-content-box">'
                end_marker = '</pre>'
                start_pos = html_content.find(start_marker) + len(start_marker)
                end_pos = html_content.find(end_marker, start_pos)
                
                if start_pos > 0 and end_pos > start_pos:
                    content = html_content[start_pos:end_pos].strip()
                    
                    # Check if it's just the placeholder
                    if "Streaming content will appear here" in content or "Content is being loaded" in content:
                        logger.error("ISSUE: Found placeholder message instead of actual content")
                        return False
                    elif len(content) < 20:
                        logger.error(f"ISSUE: Content too short ({len(content)} chars): '{content}'")
                        return False
                    else:
                        logger.info(f"SUCCESS: Found actual streaming content ({len(content)} chars)")
                        logger.info(f"Content sample: {content[:100]}...")
                        return True
                else:
                    logger.error("Could not extract streaming content from HTML")
                    return False
        except Exception as e:
            logger.error(f"Error checking dashboard: {e}")
            return False
    
    def check_multiple_times(self, attempts=3, delay=2):
        """Check the dashboard multiple times to catch streaming updates."""
        success = False
        
        for i in range(attempts):
            logger.info(f"Dashboard check attempt {i+1}/{attempts}...")
            if self.check_dashboard_display():
                success = True
                break
            time.sleep(delay)
        
        return success

    def check_buffer_directly(self):
        """Make a direct request to check buffer status."""
        if not self.pid:
            logger.error("No PID available to check buffer")
            return False
            
        buffer_check_url = f"{self.base_url}/debug/buffer_check?pid={self.pid}"
        logger.info(f"Attempting to check buffer directly at {buffer_check_url}")
        
        try:
            with urllib.request.urlopen(buffer_check_url) as response:
                result = json.loads(response.read().decode('utf-8'))
                logger.info(f"Buffer check result: {result}")
                return "content" in result and len(result.get("content", "")) > 20
        except Exception as e:
            logger.error(f"Error checking buffer directly: {e}")
            return False
            
    def run_test(self):
        """Run the full dashboard streaming test."""
        logger.info("Starting dashboard streaming test...")
        
        # Make the streaming request
        pid = self.make_streaming_request()
        if not pid:
            logger.error("Failed to make streaming request or get PID")
            return False
        
        # Wait for content to be generated
        logger.info("Waiting for content generation...")
        time.sleep(5)
        
        # Check the dashboard multiple times
        success = self.check_multiple_times(attempts=4, delay=3)
        
        if not success:
            # Try the direct buffer check as a fallback
            logger.info("Checking streaming buffer directly...")
            buffer_success = self.check_buffer_directly()
            
            if buffer_success:
                logger.info("Buffer contains content but dashboard display failed")
            else:
                logger.error("Both dashboard display and buffer check failed")
        
        return success

def main():
    """Run the dashboard streaming test."""
    parser = argparse.ArgumentParser(description='Test dashboard streaming display')
    parser.add_argument('--server-url', type=str, default='http://localhost:22434',
                      help='URL of the Claude API server')
    args = parser.parse_args()
    
    tester = DashboardStreamingTest(args.server_url)
    success = tester.run_test()
    
    if success:
        logger.info("Dashboard streaming test: SUCCESS")
        return 0
    else:
        logger.error("Dashboard streaming test: FAILED")
        return 1
    
if __name__ == "__main__":
    sys.exit(main())