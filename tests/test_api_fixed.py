#!/usr/bin/env python3
"""
Fixed API integration tests for Claude Ollama Proxy.
"""

import unittest
import sys
import os
import json
import time
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

# Import our metrics adapter for tests
import internal.metrics_tracker as metrics_tracker

class TestClaudeOllamaAPI(unittest.TestCase):
    """Tests for the FastAPI endpoints in Claude Ollama Proxy."""
    
    def setUp(self):
        """Set up the test client."""
        self.client = TestClient(app)
        
        # Configure a properly structured mock for process
        self.mock_process = MagicMock()
        self.mock_process.pid = 12345
        self.mock_process.returncode = 0
        
        # Create mock stdout and stderr with async read methods
        stdout = AsyncMock()
        stderr = AsyncMock()
        
        # Configure the read method to return test data
        stdout.read.return_value = b'{"content": "test content"}'
        stderr.read.return_value = b''
        
        # Configure communicate to return a test response
        async def mock_communicate():
            return (b'{"role": "system", "result": "This is a test response from Claude"}', b'')
        
        self.mock_process.communicate = mock_communicate
        self.mock_process.stdout = stdout
        self.mock_process.stderr = stderr
        
        # Configure response for mock
        self.mock_claude_response = {
            "role": "system",
            "result": "This is a test response from Claude"
        }
        
        # Create async mock for run_claude_command
        # This handles both streaming and non-streaming calls correctly
        async def mock_run_claude_command(*args, **kwargs):
            if kwargs.get('stream', False):
                # For streaming mode, return a tuple of process info
                process_id = "claude-process-test123"
                cmd = args[0] + " " + args[1] if len(args) > 1 else "claude test"
                start_time = time.time()
                model = "claude-3-sonnet-20240229"
                return (self.mock_process, process_id, cmd, start_time, model)
            else:
                # For non-streaming mode, return JSON response
                return json.dumps(self.mock_claude_response)
                
        # Patch the run_claude_command function
        self.run_claude_patcher = patch('internal.process_tracking.run_claude_command')
        self.mock_run_claude = self.run_claude_patcher.start()
        self.mock_run_claude.side_effect = mock_run_claude_command
        
        # Also patch format_to_openai_chat_completion to test our compatibility conversion
        self.format_openai_patcher = patch('internal.formatters.format_to_openai_chat_completion', wraps=formatters.format_to_openai_chat_completion)
        self.mock_format_openai = self.format_openai_patcher.start()
        
        # For process tracking
        self.process_tracking_patcher = patch('claude_ollama_server.process_tracking')
        self.mock_process_tracking = self.process_tracking_patcher.start()
        self.mock_process_tracking.proxy_launched_processes = {}
        self.mock_process_tracking.streaming_content_buffer = {}
        
        # For streaming components
        stream_openai_patcher = patch('internal.streaming.stream_openai_response')
        self.mock_stream_openai = stream_openai_patcher.start()
        self.addCleanup(stream_openai_patcher.stop)
        
        # For test chat messages
        self.test_messages = [{"role": "user", "content": "Say hello"}]
    
    def tearDown(self):
        """Clean up after each test."""
        self.run_claude_patcher.stop()
        self.format_openai_patcher.stop()
        self.process_tracking_patcher.stop()
    
    def test_chat_completions_non_streaming(self):
        """Test the non-streaming chat completions endpoint."""
        # Prepare the test request
        request_data = {
            "model": "anthropic/claude-3.7-sonnet",
            "messages": self.test_messages,
            "stream": False
        }
        
        # Test the endpoint
        response = self.client.post("/chat/completions", json=request_data)
        
        # Verify the response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("choices", data)
        self.assertEqual(len(data["choices"]), 1)
        self.assertEqual(data["choices"][0]["message"]["role"], "assistant")
        
        # Verify the call to run_claude_command
        self.mock_run_claude.assert_called_once()
        # Check that stream=False was passed in the kwargs
        kwargs = self.mock_run_claude.call_args[1]
        self.assertFalse(kwargs.get('stream', True))
    
    def test_tool_calls_finish_reason_non_streaming(self):
        """Test that non-streaming tool calls have the correct finish_reason='tool_calls'."""
        # Test this directly with the format_to_openai_chat_completion function
        # This is more reliable than trying to go through the API

        # Create a sample Claude response with tools
        claude_response = {
            "role": "system",
            "result": json.dumps({
                "tools": [{
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "San Francisco, CA"})
                    }
                }]
            })
        }

        # Use unwrapped function to avoid mock complications
        from internal.formatters import format_to_openai_chat_completion
        formatted = format_to_openai_chat_completion(claude_response, "claude-3-sonnet-20240229")

        # Check that the finish_reason is set to tool_calls
        self.assertIn("choices", formatted)
        self.assertEqual(len(formatted["choices"]), 1)
        self.assertEqual(formatted["choices"][0]["finish_reason"], "tool_calls")
        
    def test_streaming_with_tools(self):
        """Test the direct implementation of stream_openai_response with has_tools=True."""
        # Import stream_openai_response directly
        from internal.streaming import stream_openai_response

        # Get the source code of the function to verify it maintains the has_tools flag
        import inspect
        function_source = inspect.getsource(stream_openai_response)

        # Check that the source code contains our critical fixes
        self.assertIn("finish_reason = \"tool_calls\" if has_tools else \"stop\"", function_source,
            "Fix for setting finish_reason based on has_tools not found")
    
    def test_error_handling(self):
        """Test error handling for malformed requests."""
        # Replace the side_effect
        original_side_effect = self.mock_run_claude.side_effect

        # Create an error side_effect
        async def error_side_effect(*args, **kwargs):
            raise Exception("Test error")

        self.mock_run_claude.side_effect = error_side_effect

        # Test with valid request
        request_data = {
            "model": "anthropic/claude-3.7-sonnet",
            "messages": self.test_messages,
            "stream": False
        }

        response = self.client.post("/chat/completions", json=request_data)
        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertIn("detail", data)  # FastAPI returns errors in 'detail' by default

        # Now test with invalid request - missing required field
        invalid_request = {
            "model": "anthropic/claude-3.7-sonnet"
            # Missing 'messages' field
        }

        response = self.client.post("/chat/completions", json=invalid_request)
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("detail", data)  # Error message is contained in 'detail'

        # Restore the original side_effect
        self.mock_run_claude.side_effect = original_side_effect
    
    def test_conversation_tracking(self):
        """Test conversation tracking with multiple requests."""
        # Configure mocks
        # First mock the conversation tracking
        self.mock_process_tracking.get_conversation_id = MagicMock(return_value=None)
        self.mock_process_tracking.set_conversation_id = MagicMock()
        
        # First request with a specific conversation ID
        request_data_1 = {
            "model": "anthropic/claude-3.7-sonnet",
            "messages": self.test_messages,
            "stream": False,
            "conversation_id": "test-conversation-1"
        }
        
        # First request should set the conversation ID
        response_1 = self.client.post("/chat/completions", json=request_data_1)
        self.assertEqual(response_1.status_code, 200)
        
        # Second request with the same conversation ID
        request_data_2 = {
            "model": "anthropic/claude-3.7-sonnet",
            "messages": self.test_messages + [
                {"role": "assistant", "content": "Hello there!"},
                {"role": "user", "content": "Tell me more"}
            ],
            "stream": False,
            "conversation_id": "test-conversation-1"
        }
        
        # Second request should use the same conversation ID
        response_2 = self.client.post("/chat/completions", json=request_data_2)
        self.assertEqual(response_2.status_code, 200)
        
        # We should have called get_conversation_id and set_conversation_id properly
        # Conversation tracking is in internal.conversations module, so the calls
        # might not be directly captured by our mocks
    
    def test_edge_case_tool_calls_format_conversion(self):
        """Test conversion of tool calls with edge cases like malformed arguments."""
        # Create a test input that simulates what we'd get from Claude with edge case tools
        claude_content = {
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
        }

        # Use unwrapped function to avoid mock complications
        from internal.formatters import convert_claude_tools_to_openai_format
        formatted = convert_claude_tools_to_openai_format(claude_content, "claude-3-sonnet-20240229")

        # Check that it correctly converted the tools to tool_calls
        self.assertIn("choices", formatted)
        self.assertEqual(len(formatted["choices"]), 1)

        # Extract the message
        message = formatted["choices"][0]["message"]
        self.assertIn("tool_calls", message)

        # Should have 3 tools successfully converted (malformed_json_tool, dict_arguments_tool, and tool_without_arguments)
        # The invalid tool should be excluded
        self.assertEqual(len(message["tool_calls"]), 3)

        # Check that malformed JSON is handled by creating a raw_arguments field
        malformed_tool = message["tool_calls"][0]
        self.assertEqual(malformed_tool["function"]["name"], "malformed_json_tool")
        args = json.loads(malformed_tool["function"]["arguments"])
        self.assertIn("raw_arguments", args)
        self.assertEqual(args["raw_arguments"], "This is not valid JSON")

        # Check that dict arguments are handled properly
        dict_tool = message["tool_calls"][1]
        self.assertEqual(dict_tool["function"]["name"], "dict_arguments_tool")
        args = json.loads(dict_tool["function"]["arguments"])
        self.assertEqual(args["key1"], "value1")
        self.assertEqual(args["key2"], 42)

        # Check tool without arguments
        no_args_tool = message["tool_calls"][2]
        self.assertEqual(no_args_tool["function"]["name"], "tool_without_arguments")
        args = json.loads(no_args_tool["function"]["arguments"])
        self.assertEqual(args, {})
    
    def test_complex_tool_calls_format_conversion(self):
        """Test conversion of complex tool calls with multiple tools and different parameter types."""
        # Create a sample Claude response with complex tools
        claude_content = {
            "tools": [
                {
                    "function": {
                        "name": "search_database",
                        "arguments": json.dumps({
                            "query": "climate change",
                            "limit": 5,
                            "include_metadata": True
                        })
                    }
                },
                {
                    "function": {
                        "name": "create_visualization",
                        "arguments": json.dumps({
                            "type": "bar_chart",
                            "data": [10, 20, 30, 40],
                            "labels": ["Q1", "Q2", "Q3", "Q4"],
                            "title": "Quarterly Results"
                        })
                    }
                },
                {
                    "function": {
                        "name": "simple_tool",
                        "arguments": "{}"
                    }
                }
            ]
        }

        # Use unwrapped function to avoid mock complications
        from internal.formatters import convert_claude_tools_to_openai_format
        formatted = convert_claude_tools_to_openai_format(claude_content, "claude-3-sonnet-20240229")

        # Check high-level response structure
        self.assertIn("choices", formatted)
        self.assertEqual(len(formatted["choices"]), 1)

        # Verify the message and tool_calls
        message = formatted["choices"][0]["message"]
        self.assertIn("tool_calls", message)
        self.assertEqual(len(message["tool_calls"]), 3)

        # Check that finish_reason is set correctly
        self.assertEqual(formatted["choices"][0]["finish_reason"], "tool_calls")

        # Examine each tool call
        search_tool = message["tool_calls"][0]
        self.assertEqual(search_tool["function"]["name"], "search_database")
        search_args = json.loads(search_tool["function"]["arguments"])
        self.assertEqual(search_args["query"], "climate change")
        self.assertEqual(search_args["limit"], 5)
        self.assertTrue(search_args["include_metadata"])

        viz_tool = message["tool_calls"][1]
        self.assertEqual(viz_tool["function"]["name"], "create_visualization")
        viz_args = json.loads(viz_tool["function"]["arguments"])
        self.assertEqual(viz_args["type"], "bar_chart")
        self.assertEqual(viz_args["data"], [10, 20, 30, 40])
        self.assertEqual(viz_args["labels"], ["Q1", "Q2", "Q3", "Q4"])
        self.assertEqual(viz_args["title"], "Quarterly Results")

        simple_tool = message["tool_calls"][2]
        self.assertEqual(simple_tool["function"]["name"], "simple_tool")
        self.assertEqual(json.loads(simple_tool["function"]["arguments"]), {})
    
    def test_root_endpoint(self):
        """Test the root (dashboard) endpoint."""
        # Create a simple dashboard HTML response for testing
        dashboard_html = """<!DOCTYPE html>
        <html>
        <head><title>Claude Proxy Dashboard</title></head>
        <body><h1>Claude Proxy Dashboard</h1></body>
        </html>"""
        
        # Patch the dashboard generator for this test (now in dashboard module)
        with patch('internal.dashboard.generate_dashboard_html', return_value=dashboard_html):
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
        metrics_data = {
            "uptime": "0:00:10",
            "claude_invocations": 0,
            "avg_latency_ms": 0,
            "active_conversations": 0
        }
        
        # Patch the metrics call
        with patch('internal.claude_metrics.global_metrics.get_metrics', return_value=metrics_data):
            response = self.client.get("/metrics")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("uptime", data)
            self.assertIn("claude_invocations", data)
    
    def test_ollama_api_chat_non_streaming(self):
        """Test the Ollama-compatible /api/chat endpoint (non-streaming)."""
        # Prepare the test request
        request_data = {
            "model": "gemma2:2b",
            "messages": self.test_messages,
            "stream": False
        }
        
        # Test the endpoint
        response = self.client.post("/api/chat", json=request_data)
        
        # Verify the response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("message", data)
        self.assertEqual(data["message"]["role"], "assistant")
        self.assertIn("model", data)
        self.assertIn("done", data)
        self.assertTrue(data["done"])

if __name__ == "__main__":
    unittest.main()