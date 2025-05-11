#!/usr/bin/env python3
"""
Integration test for the streaming functionality using the refactored run_claude_command function.
"""

import os
import sys
import asyncio
import unittest
from unittest.mock import patch, MagicMock, AsyncMock

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the functions we want to test
from internal.process_tracking import run_claude_command
from internal.streaming import stream_claude_output

class IntegratedStreamingTests(unittest.TestCase):
    """Tests for the integration between run_claude_command and streaming functionality."""

    async def _test_stream_claude_output(self):
        """Test the integration between run_claude_command and stream_claude_output."""
        # Mock stdout data that simulates Claude's streaming JSON output
        mock_stdout_chunks = [
            b'{"type": "message", "content": [{"type": "text", "text": "This is "}]}',
            b'{"type": "message", "content": [{"type": "text", "text": "a test "}]}',
            b'{"type": "message", "content": [{"type": "text", "text": "response."}]}',
            b'{"stop_reason": "end_turn"}'
        ]
        
        # Create a mock process with a custom stdout read method
        mock_process = AsyncMock()
        mock_process.pid = 12345
        mock_process.returncode = None
        
        # Set up the stdout mock with a custom read method that returns chunks sequentially
        mock_stdout = AsyncMock()
        mock_stdout.read = AsyncMock()
        
        # Make read return the chunks one at a time and then empty bytes
        read_side_effect = mock_stdout_chunks + [b'']
        mock_stdout.read.side_effect = read_side_effect
        
        # Set up stderr mock
        mock_stderr = AsyncMock()
        mock_stderr.read = AsyncMock(return_value=b'')
        
        # Attach the mocks to the process
        mock_process.stdout = mock_stdout
        mock_process.stderr = mock_stderr
        
        # Create a mock for asyncio.create_subprocess_shell
        mock_metrics = AsyncMock()
        mock_metrics.record_claude_start = AsyncMock()
        mock_metrics.record_claude_completion = AsyncMock()
        
        # Patch all the necessary functions
        with patch('asyncio.create_subprocess_shell', return_value=mock_process), \
             patch('internal.claude_metrics.global_metrics.record_claude_start', new_callable=AsyncMock), \
             patch('internal.claude_metrics.global_metrics.record_claude_completion', new_callable=AsyncMock), \
             patch('internal.process_tracking.track_claude_process'), \
             patch('internal.process_tracking.untrack_claude_process'):
                
            # Call the generator function 
            chunks = []
            async for chunk in stream_claude_output(mock_metrics, "claude", "test prompt", conversation_id=None):
                chunks.append(chunk)
            
            # Verify we got the expected number of chunks
            self.assertEqual(len(chunks), 4)  # 3 content chunks + 1 done marker
            
            # Check the content of the chunks
            self.assertEqual(chunks[0]["content"], "This is ")
            self.assertEqual(chunks[1]["content"], "a test ")
            self.assertEqual(chunks[2]["content"], "response.")
            self.assertTrue("done" in chunks[3] and chunks[3]["done"] is True)

    def test_stream_claude_output(self):
        """Run the stream claude output test."""
        asyncio.run(self._test_stream_claude_output())

if __name__ == "__main__":
    unittest.main()