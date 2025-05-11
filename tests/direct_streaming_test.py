#!/usr/bin/env python3
"""
Direct test of the streaming API endpoint
"""

import os
import sys
import asyncio
import aiohttp
import json

async def test_streaming_endpoint():
    """Test the streaming endpoint directly"""
    print("Testing streaming endpoint")
    
    # Define the API URL
    api_url = "http://localhost:22434"
    
    # Define the prompt
    prompt = "Tell me a short story about a robot learning to cook."
    
    # Define the request payload
    request_data = {
        "model": "claude-3-sonnet-20240229",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": True
    }
    
    # Make the streaming request
    print(f"Making streaming request to {api_url}/v1/chat/completions")
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{api_url}/v1/chat/completions",
            json=request_data,
            headers={"Content-Type": "application/json"}
        ) as response:
            if response.status != 200:
                print(f"Error: {response.status} - {await response.text()}")
                return False
                
            print(f"Streaming response started with status: {response.status}")
            
            # Process the streaming response
            chunk_count = 0
            chunks = []
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if line and line.startswith('data:'):
                    chunk_count += 1
                    data = line[5:].strip()  # Remove 'data: ' prefix
                    
                    if data == "[DONE]":
                        print("Received [DONE] marker")
                        break
                        
                    try:
                        json_data = json.loads(data)
                        chunks.append(json_data)
                        
                        # Print some info about the chunk
                        if chunk_count <= 5 or chunk_count % 10 == 0:
                            choice = json_data.get("choices", [{}])[0]
                            content = choice.get("delta", {}).get("content", "")
                            print(f"Chunk {chunk_count}: {content[:30]}...")
                            
                    except json.JSONDecodeError:
                        print(f"Error parsing chunk: {data[:100]}...")
                        
                    # Only process the first 50 chunks to limit output volume
                    if chunk_count >= 50:
                        print("Reached 50 chunks, stopping...")
                        break
            
            print(f"Received {chunk_count} total chunks")
            
            # Check if we received any chunks
            if chunk_count > 0:
                print("Streaming test PASSED")
                return True
            else:
                print("Streaming test FAILED - no chunks received")
                return False

if __name__ == "__main__":
    # Run the test
    result = asyncio.run(test_streaming_endpoint())
    
    # Exit with appropriate code
    sys.exit(0 if result else 1)