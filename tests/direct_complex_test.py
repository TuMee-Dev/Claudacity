#!/usr/bin/env python3
"""
Direct test for complex prompt with hardcoded job scheduling system prompt
"""
import aiohttp
import asyncio
import json
import time

# Configuration
proxy_url = "http://localhost:22434/api/chat"
timeout = 180  # seconds
model = "gemma2:2b"

# The complex prompt directly hardcoded
COMPLEX_PROMPT = """Create a distributed job scheduling system using Python. The system should allow multiple worker nodes to pick up tasks from a shared queue, handle task dependencies, provide fault tolerance with automatic retries, and include a monitoring dashboard. Your implementation should use Redis for the queue, include proper logging, handle edge cases like worker failures, and use type hints throughout. Also explain the architectural decisions you made and how you'd scale this system to handle thousands of workers."""

async def test_streaming():
    """Test streaming mode"""
    print(f"\n=== TESTING STREAMING MODE ===")
    print(f"Sending request to {proxy_url}")
    
    # Prepare payload
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": COMPLEX_PROMPT}],
        "stream": True
    }
    
    start_time = time.time()
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(proxy_url, json=payload, timeout=timeout) as response:
                print(f"Got response with status: {response.status}")
                
                if response.status == 200:
                    chunk_count = 0
                    total_size = 0
                    first_chunk_time = None
                    
                    async for chunk in response.content:
                        current_time = time.time()
                        chunk_count += 1
                        total_size += len(chunk)
                        
                        if chunk_count == 1:
                            first_chunk_time = current_time - start_time
                            print(f"First chunk received after {first_chunk_time:.2f} seconds")
                        
                        # Status update every 5 chunks
                        if chunk_count % 5 == 0:
                            print(f"Received {chunk_count} chunks, total size: {total_size} bytes")
                    
                    total_time = time.time() - start_time
                    print(f"Streaming completed in {total_time:.2f} seconds")
                    print(f"Total chunks: {chunk_count}, Total size: {total_size} bytes")
                    return total_time
                else:
                    print(f"Error: {await response.text()}")
                    return None
    except Exception as e:
        print(f"Error in streaming test: {e}")
        return None

async def test_non_streaming():
    """Test non-streaming mode"""
    print(f"\n=== TESTING NON-STREAMING MODE ===")
    print(f"Sending request to {proxy_url}")
    
    # Prepare payload
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": COMPLEX_PROMPT}],
        "stream": False
    }
    
    start_time = time.time()
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(proxy_url, json=payload, timeout=timeout) as response:
                print(f"Got response with status: {response.status}")
                
                if response.status == 200:
                    result = await response.json()
                    total_time = time.time() - start_time
                    
                    # Extract content
                    if "message" in result and "content" in result["message"]:
                        content = result["message"]["content"]
                        content_size = len(content)
                        print(f"Non-streaming completed in {total_time:.2f} seconds")
                        print(f"Response size: {content_size} bytes")
                        print(f"Content preview: {content[:100]}...")
                        return total_time
                    else:
                        print(f"Could not find content in response: {json.dumps(result)[:200]}")
                        return None
                else:
                    print(f"Error: {await response.text()}")
                    return None
    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        print(f"Request timed out after {elapsed:.2f} seconds")
        return None
    except Exception as e:
        print(f"Error in non-streaming test: {e}")
        return None

async def main():
    """Run both tests and compare"""
    print(f"Starting tests for the complex job scheduling prompt")
    print(f"Prompt length: {len(COMPLEX_PROMPT)} characters")
    
    # Run streaming test
    streaming_time = await test_streaming()
    
    # Run non-streaming test
    non_streaming_time = await test_non_streaming()
    
    # Compare results
    if streaming_time and non_streaming_time:
        print("\n=== COMPARISON ===")
        diff = abs(streaming_time - non_streaming_time)
        faster = "streaming" if streaming_time < non_streaming_time else "non-streaming"
        print(f"{faster.capitalize()} was faster by {diff:.2f} seconds")
        print(f"Streaming time: {streaming_time:.2f}s, Non-streaming time: {non_streaming_time:.2f}s")
        
        if streaming_time < non_streaming_time:
            print(f"Streaming was {non_streaming_time/streaming_time:.1f}x faster than non-streaming")
        else:
            print(f"Non-streaming was {streaming_time/non_streaming_time:.1f}x faster than streaming")
    else:
        print("\n=== COMPARISON ===")
        print("Could not compare times because at least one test failed")
        if streaming_time:
            print(f"Streaming time: {streaming_time:.2f}s")
        if non_streaming_time:
            print(f"Non-streaming time: {non_streaming_time:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())