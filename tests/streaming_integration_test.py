#!/usr/bin/env python3
"""
Test the integration between process_tracking.run_claude_command and streaming.stream_claude_output
"""

import os
import sys
import asyncio
import json

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the modules we want to test
import internal.process_tracking as process_tracking
import internal.streaming as streaming
from internal.claude_metrics import ClaudeMetrics
from unittest.mock import MagicMock, patch

# Mock class for testing
class MockProcess:
    def __init__(self, output=None):
        self.pid = 12345
        self.returncode = 0
        self.stdout = self
        self.stderr = self
        self._output = output or [b'{"content": "Test streaming content"}', b'{"done": true}']
        self._index = 0
        self._stderr_index = 0
        self._stderr_output = [b'']  # Empty stderr

    async def read(self, size=1024):
        if self._index < len(self._output):
            data = self._output[self._index]
            self._index += 1
            return data
        return b''

    # Add a stderr read method
    async def stderr_read(self, size=1024):
        if self._stderr_index < len(self._stderr_output):
            data = self._stderr_output[self._stderr_index]
            self._stderr_index += 1
            return data
        return b''

async def test_streaming_integration():
    """Test the integration between process_tracking and streaming modules"""
    print("Testing streaming integration between process_tracking and streaming modules")
    
    # Create a mock for run_claude_command to return a custom process
    mock_process = MockProcess()
    
    process_result = (
        mock_process,  # process object
        "claude-process-test123",  # process ID
        "claude -p test",  # command
        asyncio.get_event_loop().time(),  # start time
        "anthropic/claude-3.7-sonnet"  # model
    )
    
    # Create a patch for run_claude_command
    with patch('internal.process_tracking.run_claude_command', return_value=process_result):
        # Create a metrics object
        metrics = ClaudeMetrics()
        
        # Test the stream_claude_output function
        print("\nTest: stream_claude_output using run_claude_command")
        chunks = []
        
        # Stream is an async generator that yields chunks
        print("Calling stream_claude_output...")
        stream = streaming.stream_claude_output(
            metrics=metrics,
            claude_cmd="claude",
            prompt="test prompt",
            conversation_id=None
        )
        
        # Process the chunks
        async for chunk in stream:
            print(f"Received chunk: {chunk}")
            chunks.append(chunk)
        
        # Verify we received the expected chunks
        if len(chunks) == 2:
            print("Streaming test PASSED - received expected 2 chunks")
            return True
        else:
            print(f"Streaming test FAILED - expected 2 chunks, received {len(chunks)}")
            return False

if __name__ == "__main__":
    # Run the test
    try:
        result = asyncio.run(test_streaming_integration())
        
        # Exit with appropriate code
        sys.exit(0 if result else 1)
    except Exception as e:
        print(f"Test failed with exception: {e}")
        sys.exit(1)