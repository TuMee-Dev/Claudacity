"""
Test script to verify streaming content appears in the dashboard.
This test:
1. Makes a streaming request to Claude API
2. Checks the dashboard to see if content appears
3. Verifies streaming works end-to-end
"""

import os
import sys
import json
import time
import aiohttp
import asyncio
import unittest
from bs4 import BeautifulSoup

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Define constants
API_URL = "http://localhost:22434"  # Claude proxy server
PROMPT = "Tell me a short story about a robot learning to cook. Respond in 3-4 sentences."

class TestStreamingDashboard(unittest.TestCase):
    """Test class to verify streaming content appears in the dashboard"""
    
    async def async_setUp(self):
        """Check if server is running before tests"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{API_URL}/status") as response:
                    self.assertTrue(response.status == 200, "API server is not running")
                    print("Server is running and ready for tests")
        except Exception as e:
            self.fail(f"Could not connect to server: {e}")
    
    def setUp(self):
        """Run the async setup"""
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.async_setUp())
    
    async def async_test_streaming_content(self):
        """Test that streaming content shows up in the dashboard"""
        # Step 1: Make a streaming request to the API
        print("\n1. Making streaming request to Claude API...")
        
        # Prepare the request payload
        request_data = {
            "model": "claude-3-sonnet-20240229",
            "messages": [
                {"role": "user", "content": PROMPT}
            ],
            "stream": True,
            "max_tokens": 500
        }
        
        # Send the streaming request
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{API_URL}/v1/chat/completions",
                json=request_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                # Process the first few chunks to ensure streaming starts
                print("Processing stream...")
                chunk_counter = 0
                
                # Read a bit of the response to start the streaming
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line and line.startswith('data:'):
                        chunk_counter += 1
                        print(f"Received chunk {chunk_counter}")
                        
                        # Just get a few chunks to confirm streaming works
                        if chunk_counter >= 3:
                            break
        
        self.assertTrue(chunk_counter > 0, "No streaming chunks received")
        print(f"Received {chunk_counter} initial chunks from streaming API")
        
        # Step 2: Access the dashboard to get process list
        print("\n2. Accessing dashboard to find our process...")
        async with aiohttp.ClientSession() as session:
            async with session.get(API_URL) as dash_response:
                self.assertEqual(dash_response.status, 200, "Failed to access dashboard")
                dash_html = await dash_response.text()
        
        # Save the HTML for debugging if needed
        with open("debug_output.html", "w") as f:
            f.write(dash_html)
            print("Saved dashboard HTML to debug_output.html")
            
        # Parse the dashboard HTML to find our process
        soup = BeautifulSoup(dash_html, 'html.parser')
        pid = None
        
        # Look for the process in the recent process outputs table
        process_rows = soup.select('table tbody tr')
        for row in process_rows:
            # Look for cells with our prompt or command
            cells = row.select('td')
            for cell in cells:
                if cell.text and "robot learning to cook" in cell.text:
                    # This is our process, get the PID from the first cell
                    pid = cells[0].text.strip()
                    break
            
            if pid:
                break
        
        self.assertIsNotNone(pid, "Could not find our process in the dashboard")
        print(f"Found our process with ID: {pid}")
        
        # Step 3: Check the process output page for streaming content
        print("\n3. Checking process output page for streaming content...")
        max_retries = 10
        retry_delay = 1  # second
        content_found = False
        
        for attempt in range(max_retries):
            print(f"Attempt {attempt+1}/{max_retries} to find streaming content...")
            
            # Access the process output page
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{API_URL}/process_output/{pid}") as proc_response:
                    self.assertEqual(proc_response.status, 200, f"Failed to access process page for PID {pid}")
                    process_html = await proc_response.text()
            
            # Parse the process output HTML
            process_soup = BeautifulSoup(process_html, 'html.parser')
            
            # Look for the streaming content element
            streaming_content = process_soup.select_one('#streaming-content')
            
            if streaming_content:
                content_text = streaming_content.text.strip()
                print(f"Content found: {content_text[:100]}...")
                
                # Check if this is real content or just a placeholder
                if (content_text and 
                    "Streaming content will appear here" not in content_text and
                    "Streaming response - output sent directly to client" not in content_text and
                    len(content_text) > 50):
                    
                    content_found = True
                    print("SUCCESS: Real streaming content found in dashboard")
                    
                    # Save the successful process output page for debugging
                    with open("debug_process_output.html", "w") as f:
                        f.write(process_html)
                        print("Saved process output HTML to debug_process_output.html")
                    
                    break
            
            # Wait before retrying
            await asyncio.sleep(retry_delay)
        
        # Final assertion
        self.assertTrue(content_found, "Real streaming content was not found in the dashboard after multiple attempts")
        print("Test completed successfully - streaming content appears in dashboard!")
        
    def test_streaming_content_appears_in_dashboard(self):
        """Runner for the async test"""
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.async_test_streaming_content())

if __name__ == '__main__':
    unittest.main()