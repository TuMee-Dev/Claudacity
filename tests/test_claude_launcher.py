#!/usr/bin/env python3
"""
Better test for the Claude launcher with proper mocking
"""

import os
import sys
import asyncio
import json
import unittest
from unittest.mock import patch, MagicMock, AsyncMock

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the modules we want to test
import internal.process_tracking as process_tracking
from internal.claude_metrics import ClaudeMetrics

class TestClaudeLauncher(unittest.TestCase):
    """Test the Claude launcher functionality with mocking."""
    
    def setUp(self):
        # Mock dependencies
        # Mock asyncio.create_subprocess_shell
        self.subprocess_patcher = patch('asyncio.create_subprocess_shell')
        self.mock_create_subprocess = self.subprocess_patcher.start()
        
        # Create a mock process
        self.mock_process = AsyncMock()
        self.mock_process.pid = 12345
        self.mock_process.returncode = 0
        
        # Mock stdout and stderr for the process
        async def mock_communicate():
            return (b'{"role": "system", "result": "test response"}', b'')
        self.mock_process.communicate = mock_communicate
        
        # Make create_subprocess_shell return our mock process
        self.mock_create_subprocess.return_value = self.mock_process
        
        # Mock other dependencies
        self.metrics_patcher = patch('internal.claude_metrics.global_metrics')
        self.mock_metrics = self.metrics_patcher.start()
        
        # Setup the metrics.record_* methods as AsyncMocks
        self.mock_metrics.record_claude_start = AsyncMock()
        self.mock_metrics.record_claude_completion = AsyncMock()
        
        # Mock track_claude_process
        self.track_patcher = patch('internal.process_tracking.track_claude_process')
        self.mock_track = self.track_patcher.start()
        
        # Mock untrack_claude_process
        self.untrack_patcher = patch('internal.process_tracking.untrack_claude_process')
        self.mock_untrack = self.untrack_patcher.start()
        
    def tearDown(self):
        # Remove all patches
        self.subprocess_patcher.stop()
        self.metrics_patcher.stop()
        self.track_patcher.stop()
        self.untrack_patcher.stop()
    
    async def test_non_streaming_mode(self):
        """Test run_claude_command in non-streaming mode"""
        # Call the function with stream=False
        result = await process_tracking.run_claude_command(
            claude_cmd="claude",
            prompt="test prompt",
            stream=False
        )
        
        # Check the command was executed correctly
        self.mock_create_subprocess.assert_called_once()
        cmd_arg = self.mock_create_subprocess.call_args[0][0]
        self.assertIn("claude", cmd_arg)
        self.assertIn("test prompt", cmd_arg)
        
        # Verify communicate was called
        self.mock_process.communicate.assert_called_once()
        
        # Check the result is correct
        self.assertEqual(result, '{"role": "system", "result": "test response"}')
        
        # Verify process was tracked and untracked
        self.mock_track.assert_called()
        self.mock_untrack.assert_called()
    
    async def test_streaming_mode(self):
        """Test run_claude_command in streaming mode"""
        # Call the function with stream=True
        result = await process_tracking.run_claude_command(
            claude_cmd="claude",
            prompt="test prompt",
            stream=True
        )
        
        # Check the command was executed correctly
        self.mock_create_subprocess.assert_called_once()
        cmd_arg = self.mock_create_subprocess.call_args[0][0]
        self.assertIn("claude", cmd_arg)
        self.assertIn("test prompt", cmd_arg)
        
        # Verify the result is a tuple with the expected elements
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 5)
        
        # Unpack the tuple
        process, process_id, cmd, start_time, model = result
        
        # Verify the elements
        self.assertEqual(process, self.mock_process)
        self.assertIsInstance(process_id, str)
        self.assertTrue(process_id.startswith("claude-process-"))
        self.assertIn("claude", cmd)
        self.assertIsInstance(start_time, float)
        self.assertEqual(model, "anthropic/claude-3.7-sonnet")
        
        # Verify process was tracked but NOT untracked (responsibility of caller in streaming mode)
        self.mock_track.assert_called()
        self.mock_untrack.assert_not_called()

class AsyncioTestCase(unittest.TestCase):
    """Base test case for asyncio tests."""

    def run_async(self, coro):
        """Run a coroutine in the event loop."""
        return asyncio.run(coro)

    def test_non_streaming_mode_wrapper(self):
        """Wrapper for running the async test."""
        self.run_async(TestClaudeLauncher().test_non_streaming_mode())

    def test_streaming_mode_wrapper(self):
        """Wrapper for running the async test."""
        self.run_async(TestClaudeLauncher().test_streaming_mode())

if __name__ == "__main__":
    # Run the tests
    unittest.main()