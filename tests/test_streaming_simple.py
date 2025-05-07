"""
Simplified test script to verify streaming content appears correctly.
This test doesn't rely on external packages, just verifies the basic streaming functionality.
"""

import os
import sys
import json
import time
import asyncio
import aiohttp
import re

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Define constants
API_URL = "http://localhost:22434"  # Claude proxy server
PROMPT = "Tell me a short story about a robot learning to cook. Respond in 3-4 sentences."

async def test_streaming():
    """Test that streaming content appears correctly"""
    print("Testing streaming content display in dashboard...")
    
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
    
    if chunk_counter == 0:
        print("ERROR: No streaming chunks received")
        return False
    
    print(f"Received {chunk_counter} initial chunks from streaming API")
    
    # Step 2: Access the dashboard to get process list
    print("\n2. Accessing dashboard to find our process...")
    async with aiohttp.ClientSession() as session:
        async with session.get(API_URL) as dash_response:
            dash_html = await dash_response.text()
    
    # Save the HTML for debugging if needed
    with open("debug_output.html", "w") as f:
        f.write(dash_html)
        print("Saved dashboard HTML to debug_output.html")
        
    # Simple regex to find process ID in the HTML
    pid_match = re.search(r'<td>([^<]+)</td>.*?robot learning to cook', dash_html, re.DOTALL)
    
    if not pid_match:
        print("ERROR: Could not find our process in the dashboard")
        return False
    
    pid = pid_match.group(1).strip()
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
                process_html = await proc_response.text()
        
        # Save the HTML for debugging
        with open(f"debug_process_{attempt+1}.html", "w") as f:
            f.write(process_html)
        
        # Check for streaming content using regex
        streaming_content_match = re.search(r'id="streaming-content"[^>]*>(.*?)</pre>', process_html, re.DOTALL)
        
        if streaming_content_match:
            content_text = streaming_content_match.group(1).strip()
            print(f"Content found: {content_text[:100]}...")
            
            # Check if this is real content or just a placeholder
            if (content_text and 
                "Streaming content will appear here" not in content_text and
                "Streaming response - output sent directly to client" not in content_text and
                len(content_text) > 50):
                
                content_found = True
                print("SUCCESS: Real streaming content found in dashboard")
                
                # Save the successful process output page
                with open("debug_success_output.html", "w") as f:
                    f.write(process_html)
                    print("Saved successful process output HTML")
                
                break
        
        # Wait before retrying
        await asyncio.sleep(retry_delay)
    
    if content_found:
        print("Test PASSED: Streaming content appears in dashboard!")
        return True
    else:
        print("Test FAILED: Real streaming content was not found in the dashboard after multiple attempts")
        return False

if __name__ == '__main__':
    try:
        result = asyncio.run(test_streaming())
        # Exit with appropriate code (0 for success, 1 for failure)
        sys.exit(0 if result else 1)
    except Exception as e:
        print(f"Test error: {e}")
        sys.exit(1)