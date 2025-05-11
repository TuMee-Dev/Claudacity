#!/usr/bin/env python3
"""
Test for run_claude_command function using unittest
"""

import os
import sys
import asyncio
import unittest
from unittest.mock import patch, MagicMock, AsyncMock

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the function we want to test
from internal.process_tracking import run_claude_command

class RunClaudeCommandTests(unittest.TestCase):
    """Tests for the run_claude_command function."""

    async def _test_non_streaming_mode(self):
        """Test run_claude_command in non-streaming mode."""
        # Create mock results
        mock_stdout = b'{"role": "system", "result": "test response"}'
        mock_stderr = b''
        
        # Create a mock process
        mock_process = AsyncMock()
        mock_process.pid = 12345
        mock_process.returncode = 0
        mock_process.communicate.return_value = (mock_stdout, mock_stderr)
        
        # Create a mock for asyncio.create_subprocess_shell
        with patch('asyncio.create_subprocess_shell', return_value=mock_process), \
             patch('internal.claude_metrics.global_metrics.record_claude_start', new_callable=AsyncMock), \
             patch('internal.claude_metrics.global_metrics.record_claude_completion', new_callable=AsyncMock), \
             patch('internal.process_tracking.track_claude_process'), \
             patch('internal.process_tracking.untrack_claude_process'):
             
            # Call the function with stream=False
            result = await run_claude_command(
                claude_cmd="claude",
                prompt="test prompt",
                stream=False
            )
            
            # Check that communicate was called
            mock_process.communicate.assert_called_once()
            
            # Check that the result is the parsed JSON response
            self.assertIsInstance(result, dict)
            self.assertTrue('id' in result)
            self.assertTrue('role' in result or ('parsed_json' in result and result['parsed_json'] == False))
    
    async def _test_streaming_mode(self):
        """Test run_claude_command in streaming mode."""
        # Create a mock process
        mock_process = AsyncMock()
        mock_process.pid = 12345
        mock_process.returncode = 0
        
        # Create a mock for asyncio.create_subprocess_shell
        with patch('asyncio.create_subprocess_shell', return_value=mock_process), \
             patch('internal.claude_metrics.global_metrics.record_claude_start', new_callable=AsyncMock), \
             patch('internal.process_tracking.track_claude_process'), \
             patch('internal.process_tracking.untrack_claude_process'):
             
            # Call the function with stream=True
            result = await run_claude_command(
                claude_cmd="claude",
                prompt="test prompt",
                stream=True
            )
            
            # Verify the result is a tuple with the expected elements
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 5)
            
            # Unpack the tuple
            process, process_id, cmd, start_time, model = result
            
            # Verify the elements
            self.assertEqual(process, mock_process)
            self.assertIsInstance(process_id, str)
            self.assertTrue(process_id.startswith("claude-process-"))
            self.assertIn("claude", cmd)
            self.assertIsInstance(start_time, float)
    
    def test_non_streaming_mode(self):
        """Run the non-streaming mode test."""
        asyncio.run(self._test_non_streaming_mode())
    
    def test_streaming_mode(self):
        """Run the streaming mode test."""
        asyncio.run(self._test_streaming_mode())

if __name__ == "__main__":
    unittest.main()