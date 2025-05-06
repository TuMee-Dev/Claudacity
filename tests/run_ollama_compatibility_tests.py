#!/usr/bin/env python3
"""
Run Ollama compatibility tests against both the Claude service and Ollama.
This will compare the API responses from both services.
"""

import argparse
import subprocess
import time
import os
import sys
import signal
import json
import asyncio

def start_test_server(port):
    """Start the test chat endpoint server on the specified port"""
    print(f"Starting test server on port {port}...")
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "test_chat_endpoint.py")]
    env = os.environ.copy()
    env["PORT"] = str(port)
    
    # Start the test server as a background process
    server_process = subprocess.Popen(
        cmd, 
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for the server to start
    print("Waiting for server to start...")
    time.sleep(2)
    
    return server_process

def run_tests(claude_url, ollama_url, test_server_port=None):
    """Run the ollama compatibility tests"""
    test_server_process = None
    
    try:
        # If test_server_port is provided, start the test server
        if test_server_port:
            test_server_process = start_test_server(test_server_port)
            claude_url = f"http://localhost:{test_server_port}"
            print(f"Using test server at {claude_url}")
        
        # Run the test script
        test_script = os.path.join(os.path.dirname(__file__), "test_ollama_compatibility.py")
        cmd = [
            sys.executable, 
            test_script,
            "--claude-url", claude_url,
            "--ollama-url", ollama_url,
            "--compare"
        ]
        
        print("\nRunning tests...")
        subprocess.run(cmd)
        
    finally:
        # Clean up the test server if we started one
        if test_server_process:
            print("\nStopping test server...")
            test_server_process.terminate()
            stdout, stderr = test_server_process.communicate(timeout=5)
            if stdout:
                print("Server stdout:", stdout)
            if stderr and stderr.strip():
                print("Server stderr:", stderr)

def main():
    parser = argparse.ArgumentParser(description="Run Ollama compatibility tests")
    parser.add_argument("--claude-url", default="http://localhost:22434", 
                      help="URL of the Claude service (default: http://localhost:22434)")
    parser.add_argument("--ollama-url", default="http://localhost:11434",
                      help="URL of the Ollama service (default: http://localhost:11434)")
    parser.add_argument("--test-server", action="store_true",
                      help="Start a test server instead of connecting to an existing Claude service")
    parser.add_argument("--port", type=int, default=8080,
                      help="Port to use for the test server (default: 8080)")
    
    args = parser.parse_args()
    
    # Handle CTRL+C gracefully
    def signal_handler(sig, frame):
        print("\nTest run interrupted. Cleaning up...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    if args.test_server:
        # Run with the test server
        run_tests(args.claude_url, args.ollama_url, test_server_port=args.port)
    else:
        # Run against existing services
        run_tests(args.claude_url, args.ollama_url)
    
    print("\nTests completed.")

if __name__ == "__main__":
    main()