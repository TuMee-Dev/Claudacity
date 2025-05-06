#!/usr/bin/env python3
"""
Unified Test Framework for Claudacity

This framework tests both Ollama API and Claudacity proxy with multiple prompts,
testing both streaming and non-streaming responses for each prompt.
"""
import os
import sys
import json
import time
import asyncio
import argparse
import aiohttp
from datetime import datetime
from pathlib import Path

# Add the parent directory to the path so we can import project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Global constants
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_PROXY_URL = "http://localhost:22434/api/chat"
DEFAULT_MODEL = "gemma2:2b"  # Use a model that exists in Ollama
DEFAULT_TIMEOUT = 300  # 5 minutes
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

# Configure logging
def log_message(message):
    """Print a timestamped log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

class TestResult:
    """Class to store test results."""
    def __init__(self, prompt_file, endpoint, streaming, success, elapsed_time=None, error=None, response_preview=None):
        self.prompt_file = prompt_file
        self.endpoint = endpoint
        self.streaming = streaming
        self.success = success
        self.elapsed_time = elapsed_time
        self.error = error
        self.response_preview = response_preview  # Preview of the actual response content
        
    def __str__(self):
        status = "SUCCESS" if self.success else "FAILURE"
        time_str = f" (completed in {self.elapsed_time:.2f}s)" if self.elapsed_time else ""
        error_str = f" - Error: {self.error}" if self.error else ""
        streaming_str = "Streaming" if self.streaming else "Non-streaming"
        result = f"{status}: {self.endpoint} - {streaming_str} - {os.path.basename(self.prompt_file)}{time_str}{error_str}"
        
        # Add a response preview if available
        if self.response_preview and self.success:
            result += f"\n    Response: {self.response_preview}"
            
        return result

async def test_endpoint(endpoint, prompt, is_streaming, timeout=DEFAULT_TIMEOUT):
    """Test a specific endpoint with a prompt."""
    model = DEFAULT_MODEL
    log_message(f"Testing {'streaming' if is_streaming else 'non-streaming'} request to {endpoint}")
    log_message(f"Using prompt: {os.path.basename(prompt)}")
    
    # Read the prompt content
    try:
        with open(prompt, 'r') as f:
            prompt_content = f.read().strip()
    except Exception as e:
        return TestResult(prompt, endpoint, is_streaming, False, error=f"Failed to read prompt: {e}")
    
    # Prepare the request payload
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt_content}
        ],
        "stream": is_streaming
    }
    
    # Execute the request
    start_time = time.time()
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=payload, timeout=timeout) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return TestResult(
                        prompt, endpoint, is_streaming, False,
                        elapsed_time=time.time() - start_time,
                        error=f"HTTP {response.status}: {error_text[:100]}..."
                    )
                
                if is_streaming:
                    # For streaming requests, process the stream
                    chunk_count = 0
                    total_content_size = 0
                    all_chunks = []
                    
                    async for chunk in response.content:
                        chunk_count += 1
                        chunk_text = chunk.decode('utf-8', errors='replace')
                        total_content_size += len(chunk)
                        all_chunks.append(chunk_text)
                    
                    log_message(f"Received {chunk_count} chunks from {endpoint}, total size: {total_content_size} bytes")
                    
                    # Extract content from chunks (format varies by endpoint)
                    combined_content = "".join(all_chunks)
                    content_preview = ""
                    
                    try:
                        # Try to parse the chunks and extract content
                        if "message" in combined_content and "content" in combined_content:
                            # Try to find content in JSON chunks
                            parts = []
                            for chunk in all_chunks:
                                try:
                                    chunk_json = json.loads(chunk)
                                    if isinstance(chunk_json, dict) and "message" in chunk_json and "content" in chunk_json["message"]:
                                        parts.append(chunk_json["message"]["content"])
                                except:
                                    continue
                            
                            if parts:
                                content_preview = f"Start: {parts[0][:50]}... End: ...{parts[-1][-50:]}"
                            else:
                                # Just use the first and last parts of the raw content
                                content_preview = f"Start: {combined_content[:50]}... End: ...{combined_content[-50:]}"
                        else:
                            # Just use the first and last parts of the raw content
                            content_preview = f"Start: {combined_content[:50]}... End: ...{combined_content[-50:]}"
                        
                    except Exception as e:
                        log_message(f"Error parsing streaming content: {e}")
                        content_preview = f"Start: {combined_content[:50]}... End: ...{combined_content[-50:]}"
                    
                    elapsed_time = time.time() - start_time
                    return TestResult(
                        prompt, endpoint, is_streaming, True, 
                        elapsed_time=elapsed_time,
                        response_preview=content_preview
                    )
                else:
                    # For non-streaming requests, just read the response
                    response_json = await response.json()
                    response_text = str(response_json)
                    
                    # Log a small sample of the response
                    log_message(f"Received non-streaming response from {endpoint}, size: {len(response_text)} bytes")
                    log_message(f"Response sample: {response_text[:100]}...")
                    
                    # Extract the content from the response
                    content_preview = ""
                    try:
                        if isinstance(response_json, dict) and "message" in response_json and "content" in response_json["message"]:
                            content = response_json["message"]["content"]
                            if len(content) > 100:
                                content_preview = f"Start: {content[:50]}... End: ...{content[-50:]}"
                            else:
                                content_preview = content
                        else:
                            # Just use the start and end of the raw response
                            content_preview = f"Start: {response_text[:50]}... End: ...{response_text[-50:]}"
                    except Exception as e:
                        log_message(f"Error extracting content: {e}")
                        content_preview = f"Start: {response_text[:50]}... End: ...{response_text[-50:]}"
                    
                    # Check if the response contains an error
                    if isinstance(response_json, dict) and ('error' in response_json or 
                            (isinstance(response_json.get('message', {}), dict) and 
                             'content' in response_json['message'] and 
                             'Error:' in response_json['message']['content'])):
                        error_msg = response_json.get('error', response_json.get('message', {}).get('content', ''))
                        return TestResult(
                            prompt, endpoint, is_streaming, False,
                            elapsed_time=time.time() - start_time,
                            error=f"Error in response: {error_msg}"
                        )
                    
                    if not response_json:
                        return TestResult(
                            prompt, endpoint, is_streaming, False,
                            elapsed_time=time.time() - start_time,
                            error="Empty response"
                        )
                        
                    elapsed_time = time.time() - start_time
                    return TestResult(
                        prompt, endpoint, is_streaming, True, 
                        elapsed_time=elapsed_time,
                        response_preview=content_preview
                    )
        
        # This code should never be reached since we return from both branches above
        elapsed_time = time.time() - start_time
        log_message(f"Test completed in {elapsed_time:.2f} seconds")
        return TestResult(prompt, endpoint, is_streaming, True, elapsed_time=elapsed_time, 
                         response_preview="No content extracted")
        
    except asyncio.TimeoutError:
        elapsed_time = time.time() - start_time
        return TestResult(
            prompt, endpoint, is_streaming, False,
            elapsed_time=elapsed_time,
            error=f"Timeout after {elapsed_time:.2f}s"
        )
    except Exception as e:
        elapsed_time = time.time() - start_time
        return TestResult(
            prompt, endpoint, is_streaming, False,
            elapsed_time=elapsed_time,
            error=str(e)
        )

async def run_tests(ollama_url, proxy_url, timeout, skip_ollama=False, skip_proxy=False,
                   skip_streaming=False, skip_non_streaming=False):
    """Run all tests with all prompts."""
    # Get all prompt files sorted by name
    prompt_files = sorted([os.path.join(PROMPTS_DIR, f) for f in os.listdir(PROMPTS_DIR) if f.endswith('.txt')])
    
    if not prompt_files:
        log_message("No prompt files found in the prompts directory!")
        return []
    
    log_message(f"Found {len(prompt_files)} prompt files")
    for prompt in prompt_files:
        log_message(f"  - {os.path.basename(prompt)}")
    
    # Generate test combinations
    test_configs = []
    
    # Only add Ollama tests if not skipped
    if not skip_ollama:
        if not skip_streaming:
            test_configs.extend([(ollama_url, prompt, True) for prompt in prompt_files])
        if not skip_non_streaming:
            test_configs.extend([(ollama_url, prompt, False) for prompt in prompt_files])
    
    # Only add Proxy tests if not skipped
    if not skip_proxy:
        if not skip_streaming:
            test_configs.extend([(proxy_url, prompt, True) for prompt in prompt_files])
        if not skip_non_streaming:
            test_configs.extend([(proxy_url, prompt, False) for prompt in prompt_files])
    
    # Run all tests in parallel
    log_message(f"Running {len(test_configs)} test configurations...")
    results = await asyncio.gather(*[test_endpoint(endpoint, prompt, streaming, timeout) 
                                   for endpoint, prompt, streaming in test_configs])
    
    return results

def format_summary(results):
    """Format a summary of the test results."""
    total = len(results)
    successful = sum(1 for r in results if r.success)
    
    # Group results by endpoint and streaming mode
    endpoint_results = {}
    for r in results:
        key = f"{r.endpoint} - {'Streaming' if r.streaming else 'Non-streaming'}"
        if key not in endpoint_results:
            endpoint_results[key] = {'total': 0, 'success': 0}
        endpoint_results[key]['total'] += 1
        if r.success:
            endpoint_results[key]['success'] += 1
    
    # Group results by prompt file
    prompt_results = {}
    for r in results:
        prompt_name = os.path.basename(r.prompt_file)
        if prompt_name not in prompt_results:
            prompt_results[prompt_name] = {'total': 0, 'success': 0}
        prompt_results[prompt_name]['total'] += 1
        if r.success:
            prompt_results[prompt_name]['success'] += 1
    
    # Format the summary
    summary = [
        "=" * 80,
        f"TEST SUMMARY: {successful}/{total} tests passed ({successful/total*100:.1f}%)",
        "=" * 80,
        "\nResults by Endpoint:",
        "-" * 40
    ]
    
    for endpoint, counts in endpoint_results.items():
        rate = counts['success'] / counts['total'] * 100 if counts['total'] > 0 else 0
        summary.append(f"{endpoint}: {counts['success']}/{counts['total']} passed ({rate:.1f}%)")
    
    summary.extend([
        "\nResults by Prompt:",
        "-" * 40
    ])
    
    for prompt, counts in prompt_results.items():
        rate = counts['success'] / counts['total'] * 100 if counts['total'] > 0 else 0
        summary.append(f"{prompt}: {counts['success']}/{counts['total']} passed ({rate:.1f}%)")
    
    # Add detailed success results showing response content
    summary.extend([
        "\nResponse Samples from Successful Tests:",
        "-" * 80
    ])
    
    success_tests = [r for r in results if r.success and r.response_preview]
    for test in success_tests:
        prompt_name = os.path.basename(test.prompt_file)
        stream_type = "Streaming" if test.streaming else "Non-streaming"
        endpoint_short = test.endpoint.split('/')[-3]  # Extract the hostname:port part
        summary.append(f"{prompt_name} - {endpoint_short} - {stream_type} ({test.elapsed_time:.2f}s):")
        summary.append(f"  {test.response_preview}")
        summary.append("")
    
    if not success_tests:
        summary.append("No successful tests with response content found!")
    
    summary.extend([
        "\nFailed Tests:",
        "-" * 40
    ])
    
    failed_tests = [r for r in results if not r.success]
    for test in failed_tests:
        summary.append(str(test))
    
    if not failed_tests:
        summary.append("No failed tests!")
    
    return "\n".join(summary)

async def main():
    """Main entry point for the test framework."""
    parser = argparse.ArgumentParser(description='Unified Test Framework for Claudacity')
    parser.add_argument('--ollama-url', type=str, default=os.environ.get('OLLAMA_URL', DEFAULT_OLLAMA_URL),
                        help=f'Ollama API URL (default: {DEFAULT_OLLAMA_URL})')
    parser.add_argument('--proxy-url', type=str, default=os.environ.get('PROXY_URL', DEFAULT_PROXY_URL),
                        help=f'Proxy API URL (default: {DEFAULT_PROXY_URL})')
    parser.add_argument('--timeout', type=int, default=int(os.environ.get('TEST_TIMEOUT', DEFAULT_TIMEOUT)),
                        help=f'Request timeout in seconds (default: {DEFAULT_TIMEOUT})')
    parser.add_argument('--skip-ollama', action='store_true',
                        help='Skip Ollama API tests')
    parser.add_argument('--skip-proxy', action='store_true',
                        help='Skip Proxy API tests')
    parser.add_argument('--skip-streaming', action='store_true',
                        help='Skip streaming tests')
    parser.add_argument('--skip-non-streaming', action='store_true',
                        help='Skip non-streaming tests')
    parser.add_argument('--output', type=str,
                        help='Output file for test results (optional)')
    
    args = parser.parse_args()
    
    log_message("Starting Unified Test Framework for Claudacity")
    log_message(f"Ollama URL: {args.ollama_url}")
    log_message(f"Proxy URL: {args.proxy_url}")
    log_message(f"Timeout: {args.timeout}s")
    log_message(f"Skip Ollama: {args.skip_ollama}")
    log_message(f"Skip Proxy: {args.skip_proxy}")
    log_message(f"Skip Streaming: {args.skip_streaming}")
    log_message(f"Skip Non-streaming: {args.skip_non_streaming}")
    
    # Check if the prompts directory exists
    if not os.path.exists(PROMPTS_DIR):
        log_message(f"Prompts directory not found: {PROMPTS_DIR}")
        return 1
    
    # Run the tests
    results = await run_tests(
        args.ollama_url, args.proxy_url, args.timeout,
        args.skip_ollama, args.skip_proxy,
        args.skip_streaming, args.skip_non_streaming
    )
    
    # Print the summary
    summary = format_summary(results)
    print("\n" + summary)
    
    # Write the summary to a file if requested
    if args.output:
        try:
            with open(args.output, 'w') as f:
                f.write(summary)
            log_message(f"Summary written to {args.output}")
        except Exception as e:
            log_message(f"Error writing summary to {args.output}: {e}")
    
    # Return the exit code based on test success
    all_tests_passed = all(r.success for r in results)
    return 0 if all_tests_passed else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)