#!/usr/bin/env python3
"""
Script to test the complex prompt response times in both streaming and non-streaming modes.
"""
import asyncio
import time
import json
import aiohttp # type: ignore
import argparse
from datetime import datetime

# Constants
DEFAULT_PROXY_URL = "http://localhost:22434/api/chat"
DEFAULT_MODEL = "gemma2:2b"

# Get the timeout values from the server
import os
import sys
import json
import subprocess

# Use the get_server_timeouts.py script to get the current timeout values
script_dir = os.path.dirname(os.path.abspath(__file__))
timeouts_script = os.path.join(script_dir, "get_server_timeouts.py")
result = subprocess.run([sys.executable, timeouts_script], capture_output=True, text=True)
timeouts = json.loads(result.stdout)

# Set the test timeout to slightly more than the server max_silence to avoid client timeouts
DEFAULT_TIMEOUT = min(max(timeouts["max_silence"] * 1.1, 120), 300)  # Between 2-5 minutes
print(f"Using timeout of {DEFAULT_TIMEOUT:.1f} seconds (server max_silence = {timeouts['max_silence']} seconds)")

# Load the complex prompt
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COMPLEX_PROMPT_PATH = os.path.join(SCRIPT_DIR, "prompts", "03_complex.txt")

def log_message(message):
    """Print a timestamped log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

async def test_streaming_mode(proxy_url, timeout):
    """Test the complex prompt in streaming mode."""
    log_message("Testing STREAMING mode with complex prompt...")
    
    # Load the prompt
    with open(COMPLEX_PROMPT_PATH, 'r') as f:
        prompt_content = f.read().strip()
    
    # Prepare the request payload
    payload = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "user", "content": prompt_content}
        ],
        "stream": True
    }
    
    # Execute the request
    start_time = time.time()
    total_size = 0
    chunk_count = 0
    first_chunk_time = None
    chunks = []
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(proxy_url, json=payload, timeout=timeout) as response:
                if response.status != 200:
                    error_text = await response.text()
                    log_message(f"Error: HTTP {response.status}: {error_text[:100]}...")
                    return
                
                log_message("Connected to streaming endpoint, receiving response...")
                
                async for chunk in response.content:
                    chunk_count += 1
                    current_time = time.time()
                    
                    # Record first chunk time
                    if chunk_count == 1:
                        first_chunk_time = current_time - start_time
                        log_message(f"Time to first chunk: {first_chunk_time:.2f} seconds")
                    
                    # Store chunk info
                    chunk_text = chunk.decode('utf-8', errors='replace')
                    total_size += len(chunk_text)
                    chunks.append(chunk_text)
                    
                    # Log every 2nd chunk
                    if chunk_count % 2 == 0:
                        elapsed = current_time - start_time
                        log_message(f"Received {chunk_count} chunks ({total_size} bytes) after {elapsed:.2f} seconds")
                    
                    # Force a flush to ensure output is visible
                    import sys
                    sys.stdout.flush()
        
        # Calculate stats
        total_time = time.time() - start_time
        log_message(f"Streaming response completed in {total_time:.2f} seconds")
        log_message(f"Received {chunk_count} chunks, total size: {total_size} bytes")
        log_message(f"Time to first chunk: {first_chunk_time:.2f} seconds")
        
        # Try to extract content from chunks
        try:
            # Extract content from chunks
            parts = []
            for chunk in chunks:
                try:
                    chunk_json = json.loads(chunk)
                    if isinstance(chunk_json, dict) and "message" in chunk_json and "content" in chunk_json["message"]:
                        parts.append(chunk_json["message"]["content"])
                except:
                    continue
            
            if parts:
                log_message(f"Content preview - First chunk: {parts[0][:100]}")
                log_message(f"Content preview - Last chunk: {parts[-1][-100:]}")
        except Exception as e:
            log_message(f"Error parsing content: {e}")
            
        return {
            "success": True,
            "time_to_first_chunk": first_chunk_time,
            "total_time": total_time,
            "chunk_count": chunk_count,
            "total_size": total_size
        }
            
    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        log_message(f"Timeout after {elapsed:.2f}s waiting for streaming response")
        return {
            "success": False,
            "error": "Timeout",
            "elapsed_time": elapsed
        }
    except Exception as e:
        elapsed = time.time() - start_time
        log_message(f"Error during streaming test: {e}")
        return {
            "success": False,
            "error": str(e),
            "elapsed_time": elapsed
        }

async def test_non_streaming_mode(proxy_url, timeout):
    """Test the complex prompt in non-streaming mode."""
    log_message("Testing NON-STREAMING mode with complex prompt...")
    
    # Load the prompt
    with open(COMPLEX_PROMPT_PATH, 'r') as f:
        prompt_content = f.read().strip()
    
    # Prepare the request payload
    payload = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "user", "content": prompt_content}
        ],
        "stream": False
    }
    
    # Execute the request
    start_time = time.time()
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(proxy_url, json=payload, timeout=timeout) as response:
                if response.status != 200:
                    error_text = await response.text()
                    log_message(f"Error: HTTP {response.status}: {error_text[:100]}...")
                    return
                
                log_message("Connected to non-streaming endpoint, waiting for response...")
                
                # Log progress while waiting
                wait_timer = 0
                while wait_timer < timeout:
                    # Check if the response is ready
                    if response.content_length is not None:
                        log_message(f"Response has content length of {response.content_length} bytes")
                        break
                        
                    # Log and wait
                    wait_timer += 5
                    if wait_timer % 15 == 0:  # Log every 15 seconds
                        log_message(f"Still waiting for response after {wait_timer} seconds...")
                        import sys
                        sys.stdout.flush()
                    
                    # Wait a bit before checking again
                    await asyncio.sleep(5)
                
                # Read the entire response
                response_json = await response.json()
                response_text = str(response_json)
                total_size = len(response_text)
                
                total_time = time.time() - start_time
                log_message(f"Non-streaming response completed in {total_time:.2f} seconds")
                log_message(f"Received response, total size: {total_size} bytes")
                
                # Try to extract content
                try:
                    if isinstance(response_json, dict) and "message" in response_json and "content" in response_json["message"]:
                        content = response_json["message"]["content"]
                        log_message(f"Content preview - Start: {content[:100]}")
                        log_message(f"Content preview - End: {content[-100:]}")
                except Exception as e:
                    log_message(f"Error extracting content: {e}")
                
                return {
                    "success": True,
                    "total_time": total_time,
                    "total_size": total_size
                }
                
    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        log_message(f"Timeout after {elapsed:.2f}s waiting for non-streaming response")
        return {
            "success": False,
            "error": "Timeout",
            "elapsed_time": elapsed
        }
    except Exception as e:
        elapsed = time.time() - start_time
        log_message(f"Error during non-streaming test: {e}")
        return {
            "success": False,
            "error": str(e),
            "elapsed_time": elapsed
        }

async def main():
    parser = argparse.ArgumentParser(description="Test complex prompt response times")
    parser.add_argument("--proxy-url", type=str, default=DEFAULT_PROXY_URL,
                        help=f"Proxy API URL (default: {DEFAULT_PROXY_URL})")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT})")
    parser.add_argument("--both", action="store_true", help="Run both streaming and non-streaming tests")
    parser.add_argument("--non-streaming", action="store_true", help="Run only non-streaming test")
    parser.add_argument("--streaming", action="store_true", help="Run only streaming test")
    
    args = parser.parse_args()
    
    # Determine which tests to run
    run_streaming = args.streaming or args.both or (not args.non_streaming and not args.both)
    run_non_streaming = args.non_streaming or args.both
    
    results = {}
    
    # Run the tests
    if run_streaming:
        log_message("=== STARTING STREAMING TEST ===")
        streaming_result = await test_streaming_mode(args.proxy_url, args.timeout)
        results["streaming"] = streaming_result
        log_message("=== STREAMING TEST COMPLETE ===\n")
    
    if run_non_streaming:
        log_message("=== STARTING NON-STREAMING TEST ===")
        non_streaming_result = await test_non_streaming_mode(args.proxy_url, args.timeout)
        results["non_streaming"] = non_streaming_result
        log_message("=== NON-STREAMING TEST COMPLETE ===\n")
    
    # Print summary
    log_message("=== RESULTS SUMMARY ===")
    if "streaming" in results:
        streaming_data = results["streaming"]
        if streaming_data.get("success", False):
            log_message(f"STREAMING: Success in {streaming_data.get('total_time', 0):.2f}s")
            log_message(f"  - Time to first chunk: {streaming_data.get('time_to_first_chunk', 0):.2f}s")
            log_message(f"  - Chunk count: {streaming_data.get('chunk_count', 0)}")
            log_message(f"  - Total size: {streaming_data.get('total_size', 0)} bytes")
        else:
            log_message(f"STREAMING: Failed - {streaming_data.get('error', 'Unknown error')}")
            log_message(f"  - Elapsed time: {streaming_data.get('elapsed_time', 0):.2f}s")
    
    if "non_streaming" in results:
        non_streaming_data = results["non_streaming"]
        if non_streaming_data.get("success", False):
            log_message(f"NON-STREAMING: Success in {non_streaming_data.get('total_time', 0):.2f}s")
            log_message(f"  - Total size: {non_streaming_data.get('total_size', 0)} bytes")
        else:
            log_message(f"NON-STREAMING: Failed - {non_streaming_data.get('error', 'Unknown error')}")
            log_message(f"  - Elapsed time: {non_streaming_data.get('elapsed_time', 0):.2f}s")
    
    # Comparison if both were run
    if "streaming" in results and "non_streaming" in results:
        s_success = results["streaming"].get("success", False)
        ns_success = results["non_streaming"].get("success", False)
        
        if s_success and ns_success:
            s_time = results["streaming"].get("total_time", 0)
            ns_time = results["non_streaming"].get("total_time", 0)
            difference = abs(ns_time - s_time)
            percentage = (difference / min(s_time, ns_time)) * 100
            
            log_message(f"COMPARISON: Non-streaming took {difference:.2f}s {'more' if ns_time > s_time else 'less'} than streaming")
            log_message(f"  - Percentage difference: {percentage:.1f}%")
    
    log_message("=== TEST COMPLETE ===")

if __name__ == "__main__":
    asyncio.run(main())