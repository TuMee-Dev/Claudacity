#!/usr/bin/env python3
"""
API integration tests for Claude Ollama Proxy.
"""

import unittest
import sys
import os
import json
import asyncio
from unittest.mock import patch, MagicMock

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modules after path setup
from fastapi.testclient import TestClient
import claude_ollama_server
from claude_ollama_server import app

class TestClaudeOllamaAPI(unittest.TestCase):
    """Tests for the FastAPI endpoints in Claude Ollama Proxy."""
    
    def setUp(self):
        """Set up the test client."""
        self.client = TestClient(app)
        
        # Mock the run_claude_command function to avoid actual CLI calls
        self.run_claude_patcher = patch('claude_ollama_server.run_claude_command')
        self.mock_run_claude = self.run_claude_patcher.start()
        
        # Configure the mock to return a valid response
        self.mock_claude_response = {
            "model": "claude-3.7-sonnet",
            "result": "This is a test response from Claude"
        }
        self.mock_run_claude.return_value = json.dumps(self.mock_claude_response)
    
    def tearDown(self):
        """Clean up after each test."""
        self.run_claude_patcher.stop()
    
    def test_root_endpoint(self):
        """Test the root (dashboard) endpoint."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Claude Proxy Dashboard", response.text)
    
    def test_status_endpoint(self):
        """Test the status endpoint."""
        response = self.client.get("/status")
        self.assertEqual(response.status_code, 200)
        self.assertIn("running", response.text.lower())
    
    def test_metrics_endpoint(self):
        """Test the metrics endpoint."""
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Check structure of metrics response
        self.assertIn("uptime", data)
        self.assertIn("claude_invocations", data)
        self.assertIn("cost", data)
    
    def test_chat_completions_non_streaming(self):
        """Test the non-streaming chat completions endpoint."""
        # Prepare the request
        request_data = {
            "model": "anthropic/claude-3.7-sonnet",
            "messages": [{"role": "user", "content": "Say hello"}],
            "stream": False
        }
        
        # Send the request
        response = self.client.post("/chat/completions", json=request_data)
        
        # Check the response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify OpenAI-compatible format
        self.assertIn("id", data)
        self.assertIn("object", data)
        self.assertEqual(data["object"], "chat.completion")
        self.assertIn("choices", data)
        self.assertTrue(len(data["choices"]) > 0)
        self.assertEqual(data["choices"][0]["message"]["role"], "assistant")
    
    def test_conversation_tracking(self):
        """Test conversation tracking with multiple requests."""
        # First request with a conversation ID
        request_data = {
            "model": "anthropic/claude-3.7-sonnet",
            "messages": [{"role": "user", "content": "Say hello"}],
            "stream": False,
            "conversation_id": "test-conversation-1"
        }
        
        # Send the first request
        response = self.client.post("/chat/completions", json=request_data)
        self.assertEqual(response.status_code, 200)
        
        # Second request with the same conversation ID
        request_data = {
            "model": "anthropic/claude-3.7-sonnet",
            "messages": [
                {"role": "user", "content": "Say hello"},
                {"role": "assistant", "content": "Hello there!"},
                {"role": "user", "content": "How are you?"}
            ],
            "stream": False,
            "conversation_id": "test-conversation-1"
        }
        
        # Send the second request
        response = self.client.post("/chat/completions", json=request_data)
        self.assertEqual(response.status_code, 200)
        
        # Check that the conversation is tracked in metrics
        response = self.client.get("/metrics")
        data = response.json()
        
        # This may need to be adjusted based on how you expose conversation metrics
        # This test will need to be updated based on the actual implementation
    
    def test_openwebui_test_endpoint(self):
        """Test the OpenWebUI test endpoint."""
        request_data = {
            "message": "Test message for OpenWebUI"
        }
        
        response = self.client.post("/openwebui_test", json=request_data)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Check format of the response
        self.assertIn("id", data)
        self.assertEqual(data["object"], "chat.completion")
        self.assertIn("choices", data)
        self.assertEqual(data["choices"][0]["message"]["content"], "Test message for OpenWebUI")
    
    def test_error_handling(self):
        """Test error handling for malformed requests."""
        # Invalid request missing required fields
        request_data = {
            "model": "anthropic/claude-3.7-sonnet"
            # Missing messages field
        }
        
        response = self.client.post("/chat/completions", json=request_data)
        self.assertEqual(response.status_code, 422)  # Validation error
        
        # Test with mock raising an exception
        self.mock_run_claude.side_effect = Exception("Test error")
        
        request_data = {
            "model": "anthropic/claude-3.7-sonnet",
            "messages": [{"role": "user", "content": "Say hello"}],
            "stream": False
        }
        
        response = self.client.post("/chat/completions", json=request_data)
        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertIn("error", data)

if __name__ == '__main__':
    unittest.main()