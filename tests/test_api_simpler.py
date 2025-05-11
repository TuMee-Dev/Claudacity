#!/usr/bin/env python3
"""
Simplified API integration tests for Claude Ollama Proxy.
"""

import unittest
import sys
import os
import json
import time
import datetime
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modules after path setup
from fastapi.testclient import TestClient # type: ignore
import config
import internal.formatters as formatters
from claude_ollama_server import app
import internal.process_tracking as process_tracking

class TestClaudeOllamaAPI(unittest.TestCase):
    """Tests for the FastAPI endpoints in Claude Ollama Proxy."""
    
    def setUp(self):
        """Set up the test client."""
        self.client = TestClient(app)
        
        # Create a mock process
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.returncode = 0
        
        # Add communicate method to mock process
        async def mock_communicate():
            return (b'{"role": "system", "result": "Test response"}', b'')
        mock_process.communicate = mock_communicate
        
        # Mock stdout and stderr
        mock_stdout = AsyncMock()
        mock_stderr = AsyncMock()
        mock_stdout.read.return_value = b'{"content": "test"}'
        mock_stderr.read.return_value = b''
        mock_process.stdout = mock_stdout
        mock_process.stderr = mock_stderr
        
        # Store mock process
        self.mock_process = mock_process
        
        # Set up mock for run_claude_command
        self.run_claude_patcher = patch('internal.process_tracking.run_claude_command')
        self.mock_run_claude = self.run_claude_patcher.start()
        
        # Configure run_claude_command mock
        async def mock_run_claude(*args, **kwargs):
            if kwargs.get('stream', False):
                # Return process tuple for streaming mode
                return (self.mock_process, "test-process-id", "test-cmd", time.time(), "claude-3-sonnet-20240229")
            else:
                # Return JSON response for non-streaming mode
                return json.dumps({"role": "system", "result": "Test response"})
                
        self.mock_run_claude.side_effect = mock_run_claude
    
    def tearDown(self):
        """Clean up after each test."""
        self.run_claude_patcher.stop()
    
    def test_non_streaming_api(self):
        """Test the non-streaming API endpoint."""
        # Prepare request data
        request_data = {
            "model": "claude-3-sonnet-20240229",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False
        }
        
        # Test the endpoint
        response = self.client.post("/chat/completions", json=request_data)
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("choices", data)
        self.assertEqual(len(data["choices"]), 1)
        self.assertEqual(data["choices"][0]["message"]["role"], "assistant")
    
    def test_ollama_api_endpoint(self):
        """Test the Ollama-compatible API endpoint."""
        # Prepare request data
        request_data = {
            "model": "gemma2:2b",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False
        }
        
        # Test the endpoint
        response = self.client.post("/api/chat", json=request_data)
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("model", data)
        self.assertIn("message", data)
        self.assertEqual(data["message"]["role"], "assistant")
        self.assertTrue(data["done"])

if __name__ == "__main__":
    unittest.main()