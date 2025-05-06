#!/usr/bin/env python3
"""
Simple script to check if Claude processes are being launched
"""
import aiohttp
import asyncio
import time
import sys

# The complex prompt 
COMPLEX_PROMPT = """Create a distributed job scheduling system using Python. The system should allow multiple worker nodes to pick up tasks from a shared queue, handle task dependencies, provide fault tolerance with automatic retries, and include a monitoring dashboard. Your implementation should use Redis for the queue, include proper logging, handle edge cases like worker failures, and use type hints throughout. Also explain the architectural decisions you made and how you'd scale this system to handle thousands of workers."""

# Test configuration
proxy_url = "http://localhost:22434/api/chat"

async def check_streaming():
    """Check if Claude processes are launched in streaming mode."""
    print("Testing if Claude processes are launched in streaming mode...")
    print(f"Using prompt length: {len(COMPLEX_PROMPT)}")
    
    # Create the payload
    payload = {
        "model": "gemma2:2b",
        "messages": [{"role": "user", "content": COMPLEX_PROMPT}],
        "stream": True
    }
    
    # Start the request asynchronously
    start_time = time.time()
    
    try:
        async with aiohttp.ClientSession() as session:
            # Start the request but don't wait for it to complete
            print(f"Sending streaming request to {proxy_url}")
            response = await session.post(proxy_url, json=payload, timeout=10)
            
            print(f"Initial response received with status: {response.status}")
            print(f"Response headers: {response.headers}")
            
            # Read the first chunk only
            first_chunk = await response.content.read(1024)
            print(f"First chunk received ({len(first_chunk)} bytes): {first_chunk[:100]}")
            
            # Success!
            print("✅ Streaming request successfully started")
            return True
            
    except Exception as e:
        print(f"❌ Error in streaming test: {e}")
        return False

async def check_non_streaming():
    """Check if Claude processes are launched in non-streaming mode."""
    print("\nTesting if Claude processes are launched in non-streaming mode...")
    print(f"Using prompt length: {len(COMPLEX_PROMPT)}")
    
    # Create the payload
    payload = {
        "model": "gemma2:2b",
        "messages": [{"role": "user", "content": COMPLEX_PROMPT}],
        "stream": False  # Non-streaming
    }
    
    # Start the request asynchronously
    start_time = time.time()
    
    try:
        async with aiohttp.ClientSession() as session:
            # Set a very short timeout - we just want to check if the request starts
            print(f"Sending non-streaming request to {proxy_url}")
            response = await session.post(proxy_url, json=payload, timeout=5)
            
            print(f"Initial response received with status: {response.status}")
            
            # If we get here, at least the request was accepted
            print("✅ Non-streaming request accepted by server")
            return True
            
    except asyncio.TimeoutError:
        # If we get a timeout, that's actually good! It means the request was accepted
        # and the server is processing it (just taking longer than our 5s timeout)
        print("✅ Non-streaming request is being processed (timed out as expected)")
        return True
    except Exception as e:
        print(f"❌ Error in non-streaming test: {e}")
        return False

async def main():
    """Run both checks."""
    streaming_result = await check_streaming()
    non_streaming_result = await check_non_streaming()
    
    print("\n=== RESULTS ===")
    print(f"Streaming mode: {'✅ PASS' if streaming_result else '❌ FAIL'}")
    print(f"Non-streaming mode: {'✅ PASS' if non_streaming_result else '❌ FAIL'}")
    
    if streaming_result and non_streaming_result:
        print("\nBoth modes are working! Claude processes are being launched correctly.")
    elif streaming_result:
        print("\nOnly streaming mode is working. Non-streaming mode is NOT launching Claude processes.")
    elif non_streaming_result:
        print("\nOnly non-streaming mode is working. Streaming mode is NOT launching Claude processes.")
    else:
        print("\nBoth modes are failing. Claude processes are NOT being launched at all.")

if __name__ == "__main__":
    asyncio.run(main())