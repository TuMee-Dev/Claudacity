#!/usr/bin/env python3
"""
Simple test script for non-streaming mode
"""
import aiohttp
import asyncio
import json
import time
import sys
import os

# Configuration
proxy_url = "http://localhost:22434/api/chat"
timeout = 180  # 3 minutes

async def test_non_streaming():
    """Test a simple request in non-streaming mode."""
    print("Testing non-streaming mode...")
    
    # Load the complex prompt from file
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        complex_prompt_path = os.path.join(script_dir, "prompts", "03_complex.txt")
        with open(complex_prompt_path, 'r') as f:
            prompt = f.read().strip()
        print(f"Using complex prompt of length {len(prompt)} characters")
    except Exception as e:
        print(f"Error loading complex prompt: {e}")
        prompt = "Create a distributed job scheduling system in Python that uses Redis for the queue."
        print(f"Using fallback prompt: {prompt}")
    
    # Prepare the request
    payload = {
        "model": "gemma2:2b",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False  # Non-streaming
    }
    
    # Make the request
    print(f"Sending request to {proxy_url}")
    start_time = time.time()
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(proxy_url, json=payload, timeout=timeout) as response:
                elapsed = time.time() - start_time
                
                # Check the response
                print(f"Got response in {elapsed:.2f} seconds with status code: {response.status}")
                
                if response.status == 200:
                    try:
                        # Parse the response
                        result = await response.json()
                        print(f"Response type: {type(result)}")
                        print(f"Response structure: {json.dumps(result, indent=2)[:1000]}")
                        
                        # Extract and print the content
                        if "message" in result and "content" in result["message"]:
                            content = result["message"]["content"]
                            print(f"Content: {content}")
                        else:
                            print(f"Could not find content in response: {result}")
                    except json.JSONDecodeError:
                        response_text = await response.text()
                        print(f"Could not parse response as JSON: {response_text[:500]}")
                else:
                    response_text = await response.text()
                    print(f"Error response: {response_text[:500]}")
    
    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        print(f"Request timed out after {elapsed:.2f} seconds")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_non_streaming())