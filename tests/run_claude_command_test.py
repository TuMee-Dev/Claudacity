#!/usr/bin/env python3
"""
Simple test for the run_claude_command function with stream support
This validates that the function works correctly with both streaming and non-streaming modes.
"""

import os
import sys
import asyncio
import json

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the modules we want to test
import internal.process_tracking as process_tracking
from internal.claude_metrics import ClaudeMetrics

async def test_run_claude_command():
    """Test the run_claude_command function"""
    print("Testing run_claude_command function")
    
    # Create a mock metrics object
    metrics = ClaudeMetrics()
    
    # Test with stream=False (non-streaming mode)
    print("\nTest 1: Non-streaming mode (stream=False)")
    try:
        # Mock the process with a custom command that simulates Claude output
        cmd = "echo '{\"role\": \"system\", \"result\": \"test response\"}'"
        prompt = "{\"type\": \"test\"}"
        
        # Call the function in non-streaming mode
        print("Calling run_claude_command with stream=False")
        result = await process_tracking.run_claude_command(
            claude_cmd=cmd,
            prompt=prompt,
            stream=False,
            metrics=metrics
        )
        
        # Check the result
        print(f"Result type: {type(result)}")
        print(f"Result: {str(result)[:100]}...")
        print("Non-streaming test passed")
        
    except Exception as e:
        print(f"Non-streaming test failed: {e}")
        return False
    
    # Test with stream=True (streaming mode)
    print("\nTest 2: Streaming mode (stream=True)")
    try:
        # Mock the process with a custom command that simulates Claude output
        cmd = "echo '{\"content\": \"test streaming response\"}'"
        prompt = "{\"type\": \"test\"}"
        
        # Call the function in streaming mode
        print("Calling run_claude_command with stream=True")
        result = await process_tracking.run_claude_command(
            claude_cmd=cmd,
            prompt=prompt,
            stream=True,
            metrics=metrics
        )
        
        # Check the result - should be a tuple of (process, process_id, cmd, start_time, model)
        print(f"Result type: {type(result)}")
        
        # Verify it's a tuple with the expected structure
        if not isinstance(result, tuple) or len(result) != 5:
            print(f"Error: Expected a tuple of length 5, got {result}")
            return False
            
        # Unpack the tuple
        process, process_id, cmd, start_time, model = result
        
        print(f"Process: {process}")
        print(f"Process ID: {process_id}")
        print(f"Command: {cmd}")
        print(f"Start time: {start_time}")
        print(f"Model: {model}")
        
        print("Streaming test passed")
        return True
        
    except Exception as e:
        print(f"Streaming test failed: {e}")
        return False

if __name__ == "__main__":
    # Run the test
    result = asyncio.run(test_run_claude_command())
    
    # Exit with appropriate code
    sys.exit(0 if result else 1)