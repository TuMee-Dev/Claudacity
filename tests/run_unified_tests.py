#!/usr/bin/env python3
"""
Runner script for the unified test framework.
"""
import os
import sys
import time
import argparse
import asyncio
from datetime import datetime

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the unified test framework
from unified_test_framework import run_tests, format_summary

async def main():
    """Main entry point for the runner script."""
    parser = argparse.ArgumentParser(description='Run unified Claudacity tests')
    parser.add_argument('--ollama-url', type=str, 
                        default=os.environ.get('OLLAMA_URL', 'http://localhost:11434/api/chat'),
                        help='Ollama API URL')
    parser.add_argument('--proxy-url', type=str, 
                        default=os.environ.get('PROXY_URL', 'http://localhost:8000/api/chat'),
                        help='Proxy API URL')
    parser.add_argument('--timeout', type=int, 
                        default=int(os.environ.get('TEST_TIMEOUT', 300)),
                        help='Request timeout in seconds')
    parser.add_argument('--output-dir', type=str, 
                        default='test_results',
                        help='Directory to store test results')
    parser.add_argument('--skip-ollama', action='store_true',
                        help='Skip Ollama API tests')
    parser.add_argument('--skip-proxy', action='store_true',
                        help='Skip Proxy API tests')
    parser.add_argument('--skip-streaming', action='store_true',
                        help='Skip streaming tests')
    parser.add_argument('--skip-non-streaming', action='store_true',
                        help='Skip non-streaming tests')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output filename based on timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"test_results_{timestamp}.txt")
    
    print(f"Starting unified Claudacity tests...")
    print(f"Test results will be saved to {output_file}")
    
    # Run the tests
    start_time = time.time()
    
    results = await run_tests(
        args.ollama_url, args.proxy_url, args.timeout,
        args.skip_ollama, args.skip_proxy,
        args.skip_streaming, args.skip_non_streaming
    )
    
    elapsed_time = time.time() - start_time
    
    # Generate and print the summary
    summary = format_summary(results)
    header = [
        f"Claudacity Unified Test Results",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total test time: {elapsed_time:.2f} seconds",
        f"Ollama URL: {args.ollama_url}",
        f"Proxy URL: {args.proxy_url}",
        f"Timeout: {args.timeout}s",
        f"Skip Ollama: {args.skip_ollama}",
        f"Skip Proxy: {args.skip_proxy}",
        f"Skip Streaming: {args.skip_streaming}",
        f"Skip Non-streaming: {args.skip_non_streaming}",
        "",
    ]
    
    full_report = "\n".join(header) + "\n" + summary
    
    print("\n" + full_report)
    
    # Write the summary to the output file
    try:
        with open(output_file, 'w') as f:
            f.write(full_report)
        print(f"\nTest results saved to {output_file}")
    except Exception as e:
        print(f"Error writing results to {output_file}: {e}")
    
    # Return an exit code based on test success
    all_tests_passed = all(r.success for r in results)
    return 0 if all_tests_passed else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)