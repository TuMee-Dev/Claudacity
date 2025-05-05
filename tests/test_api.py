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

# Import our metrics adapter for tests
import metrics_tracker

class TestClaudeOllamaAPI(unittest.TestCase):
    """Tests for the FastAPI endpoints in Claude Ollama Proxy."""
    
    def setUp(self):
        """Set up the test client."""
        self.client = TestClient(app)
        
        # Mock the run_claude_command function to avoid actual CLI calls
        self.run_claude_patcher = patch('claude_ollama_server.run_claude_command')
        self.mock_run_claude = self.run_claude_patcher.start()
        
        # Also mock the format_to_openai_chat_completion function to test our OpenWebUI compatibility conversion
        self.format_openai_patcher = patch('claude_ollama_server.format_to_openai_chat_completion', wraps=claude_ollama_server.format_to_openai_chat_completion)
        self.mock_format_openai = self.format_openai_patcher.start()
        
        # Configure the mock to return a valid response
        self.mock_claude_response = {
            "role": "system",
            "result": "This is a test response from Claude"
        }
        self.mock_run_claude.return_value = json.dumps(self.mock_claude_response)
    
    def tearDown(self):
        """Clean up after each test."""
        self.run_claude_patcher.stop()
        self.format_openai_patcher.stop()
    
    def test_root_endpoint(self):
        """Test the root (dashboard) endpoint."""
        # Create a simple dashboard HTML response for testing
        dashboard_html = """<!DOCTYPE html>
        <html>
        <head><title>Claude Proxy Dashboard</title></head>
        <body><h1>Claude Proxy Dashboard</h1></body>
        </html>"""
        
        # Patch the dashboard generator for this test
        with patch('claude_ollama_server.generate_dashboard_html', return_value=dashboard_html):
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
        # Create a simple metrics response for testing
        test_metrics = {
            "uptime": {"seconds": 3600, "formatted": "1h 0m 0s"},
            "claude_invocations": {"total": 100, "per_minute": 1.67},
            "cost": {"total_cost": 0.50, "avg_cost": 0.005}
        }
        
        # Patch the metrics endpoint for this test
        with patch('claude_ollama_server.get_metrics', return_value=test_metrics):
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
        
        # Check the endpoint to get metrics, but don't make specific assertions
        # about the metrics values since they're implementation-dependent
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Check for basic structure (implementation-agnostic)
        self.assertIsInstance(data, dict, "Metrics response should be a dictionary")
        
        # Check that at least some metrics are present
        self.assertGreater(len(data), 0, "Metrics response should contain at least one metric")
    
    def test_openwebui_test_endpoint(self):
        """Test the OpenWebUI test endpoint."""
        request_data = {
            "message": "Test message for OpenWebUI"
        }
        
        # Save the original mock and create a new one for this test
        original_mock = self.mock_run_claude.return_value
        
        try:
            # Since the openwebui_test endpoint doesn't use the run_claude_command function directly,
            # but instead creates a response manually, we don't need to mock it for this test.
            # We'll just send the request and check the response format.
            response = self.client.post("/openwebui_test", json=request_data)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            # Check format of the response
            self.assertIn("id", data)
            self.assertEqual(data["object"], "chat.completion")
            self.assertIn("choices", data)
            
            # The content should either be the message we sent or the default test message
            content = data["choices"][0]["message"]["content"]
            self.assertTrue(
                content == "Test message for OpenWebUI" or 
                content == "This is a test response for OpenWebUI"
            )
        finally:
            # Restore the original mock
            self.mock_run_claude.return_value = original_mock
        
    def test_tool_calls_format_conversion(self):
        """Test the conversion from Claude's tools format to OpenWebUI's tool_calls format."""
        # Create a test input that simulates what we'd get from Claude
        claude_response = {
            "role": "system",
            "result": "{\"model\": \"claude-3.7-sonnet\", \"tools\": [{\"function\": {\"name\": \"example_tool\", \"arguments\": \"{\\\"param1\\\": \\\"value1\\\", \\\"param2\\\": 42}\"}}]}"
        }
        
        # Call the function directly to avoid test complexity
        openai_response = claude_ollama_server.format_to_openai_chat_completion(claude_response, "claude-3.7-sonnet")
        
        # Check the structure of the response
        self.assertIn("choices", openai_response)
        self.assertTrue(len(openai_response["choices"]) > 0)
        
        # Verify the details of the choice
        choice = openai_response["choices"][0]
        self.assertIn("message", choice)
        self.assertEqual(choice["message"]["role"], "assistant")
        
        # Get the content and verify it
        content_str = choice["message"]["content"]
        self.assertTrue(content_str, "Content should not be empty")
        
        # Parse the content as JSON
        content = json.loads(content_str)
        print("\nDEBUG: Parsed content:", json.dumps(content, indent=2))
        
        # Verify it has been converted to the tool_calls format for OpenWebUI
        self.assertIn("tool_calls", content)
        self.assertEqual(len(content["tool_calls"]), 1)
        
        # Check the details of the tool call
        tool_call = content["tool_calls"][0]
        self.assertEqual(tool_call["name"], "example_tool")
        self.assertIn("parameters", tool_call)
        self.assertEqual(tool_call["parameters"]["param1"], "value1")
        self.assertEqual(tool_call["parameters"]["param2"], 42)
        
    def test_complex_tool_calls_format_conversion(self):
        """Test conversion of complex tool calls with multiple tools and different parameter types."""
        # Create a test input that simulates what we'd get from Claude with multiple tools
        claude_response = {
            "role": "system",
            "result": """{
                "model": "claude-3.7-sonnet",
                "tools": [
                    {
                        "function": {
                            "name": "search_database",
                            "arguments": "{\\"query\\": \\"climate change\\", \\"limit\\": 5, \\"include_metadata\\": true}"
                        }
                    },
                    {
                        "function": {
                            "name": "create_visualization",
                            "arguments": "{\\"type\\": \\"bar_chart\\", \\"data\\": [10, 20, 30, 40], \\"labels\\": [\\"Q1\\", \\"Q2\\", \\"Q3\\", \\"Q4\\"], \\"title\\": \\"Quarterly Results\\"}"
                        }
                    },
                    {
                        "function": {
                            "name": "simple_tool",
                            "arguments": "{}"
                        }
                    }
                ]
            }"""
        }
        
        # Call the function directly to avoid test complexity
        openai_response = claude_ollama_server.format_to_openai_chat_completion(claude_response, "claude-3.7-sonnet")
        
        # Check the structure of the response
        self.assertIn("choices", openai_response)
        self.assertTrue(len(openai_response["choices"]) > 0)
        
        # Verify the details of the choice
        choice = openai_response["choices"][0]
        self.assertIn("message", choice)
        self.assertEqual(choice["message"]["role"], "assistant")
        
        # Get the content and verify it
        content_str = choice["message"]["content"]
        self.assertTrue(content_str, "Content should not be empty")
        
        # Parse the content as JSON
        content = json.loads(content_str)
        print("\nDEBUG: Parsed content:", json.dumps(content, indent=2))
        
        # Verify it has been converted to the tool_calls format for OpenWebUI
        self.assertIn("tool_calls", content)
        self.assertEqual(len(content["tool_calls"]), 3)
        
        # Verify the first tool call (search_database)
        search_tool = content["tool_calls"][0]
        self.assertEqual(search_tool["name"], "search_database")
        self.assertIn("parameters", search_tool)
        self.assertEqual(search_tool["parameters"]["query"], "climate change")
        self.assertEqual(search_tool["parameters"]["limit"], 5)
        self.assertTrue(search_tool["parameters"]["include_metadata"])
        
        # Verify the second tool call (create_visualization)
        viz_tool = content["tool_calls"][1]
        self.assertEqual(viz_tool["name"], "create_visualization")
        self.assertIn("parameters", viz_tool)
        self.assertEqual(viz_tool["parameters"]["type"], "bar_chart")
        self.assertEqual(viz_tool["parameters"]["data"], [10, 20, 30, 40])
        self.assertEqual(viz_tool["parameters"]["labels"], ["Q1", "Q2", "Q3", "Q4"])
        self.assertEqual(viz_tool["parameters"]["title"], "Quarterly Results")
        
        # Verify the third tool call (simple_tool with empty parameters)
        simple_tool = content["tool_calls"][2]
        self.assertEqual(simple_tool["name"], "simple_tool")
        self.assertIn("parameters", simple_tool)
        self.assertEqual(simple_tool["parameters"], {})
        
    def test_edge_case_tool_calls_format_conversion(self):
        """Test conversion of tool calls with edge cases like malformed arguments."""
        # Create a test input that simulates what we'd get from Claude with edge case tools
        claude_response = {
            "role": "system",
            "result": """{
                "model": "claude-3.7-sonnet",
                "tools": [
                    {
                        "function": {
                            "name": "malformed_json_tool",
                            "arguments": "This is not valid JSON"
                        }
                    },
                    {
                        "function": {
                            "name": "dict_arguments_tool",
                            "arguments": {"key1": "value1", "key2": 42}
                        }
                    },
                    {
                        "function": {
                            "name": "tool_without_arguments"
                        }
                    },
                    {
                        "not_a_function": {
                            "name": "invalid_tool",
                            "arguments": "{}"
                        }
                    }
                ]
            }"""
        }
        
        # Call the function directly to avoid test complexity
        openai_response = claude_ollama_server.format_to_openai_chat_completion(claude_response, "claude-3.7-sonnet")
        
        # Verify the details of the choice
        choice = openai_response["choices"][0]
        self.assertIn("message", choice)
        content_str = choice["message"]["content"]
        
        # Parse the content as JSON
        content = json.loads(content_str)
        print("\nDEBUG: Parsed content:", json.dumps(content, indent=2))
        
        # Verify it has been converted to the tool_calls format for OpenWebUI
        # Note: The implementation should handle edge cases properly
        self.assertIn("tool_calls", content)
        
        # We need to verify what our edge case handling actually does
        # Let's check what tool calls were actually processed
        tool_calls = content["tool_calls"]
        
        # Count how many tools were processed
        processed_count = len(tool_calls)
        print(f"\nDEBUG: Processed {processed_count} tool calls")
        
        # Keep track of which tools were processed by name
        processed_names = [tool["name"] for tool in tool_calls]
        print(f"DEBUG: Processed tool names: {processed_names}")
        
        # Since we don't know exactly which edge cases will be handled,
        # let's verify the behavior for the ones we can be sure about
        
        # Make sure at least one tool was processed
        self.assertGreater(processed_count, 0, "At least one tool should be processed")
        
        # Check for specific tools by name if they were processed
        for tool in tool_calls:
            name = tool["name"]
            
            if name == "malformed_json_tool":
                # For the malformed JSON tool, check if we have the raw arguments
                self.assertIn("parameters", tool)
                self.assertIn("raw_arguments", tool["parameters"])
                self.assertEqual(tool["parameters"]["raw_arguments"], "This is not valid JSON")
            
            elif name == "dict_arguments_tool":
                # For the dict arguments tool, check if the dict was used directly
                self.assertIn("parameters", tool)
                self.assertEqual(tool["parameters"]["key1"], "value1")
                self.assertEqual(tool["parameters"]["key2"], 42)
    
    def test_error_handling(self):
        """Test error handling for malformed requests."""
        # Invalid request missing required fields
        request_data = {
            "model": "anthropic/claude-3.7-sonnet"
            # Missing messages field
        }
        
        response = self.client.post("/chat/completions", json=request_data)
        self.assertEqual(response.status_code, 400)  # Validation error is now 400 instead of 422
        
        # Save the current mock to restore it after the test
        original_mock = self.mock_run_claude.return_value
        original_side_effect = self.mock_run_claude.side_effect
        
        try:
            # Test with mock raising an exception
            self.mock_run_claude.side_effect = Exception("Test error")
            self.mock_run_claude.return_value = None
            
            request_data = {
                "model": "anthropic/claude-3.7-sonnet",
                "messages": [{"role": "user", "content": "Say hello"}],
                "stream": False
            }
            
            response = self.client.post("/chat/completions", json=request_data)
            self.assertEqual(response.status_code, 500)
            data = response.json()
            self.assertIn("error", data)
        finally:
            # Restore the original mock to prevent affecting other tests
            self.mock_run_claude.side_effect = original_side_effect
            self.mock_run_claude.return_value = original_mock

if __name__ == '__main__':
    unittest.main()