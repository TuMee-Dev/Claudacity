#!/usr/bin/env python3
"""
Tests for dashboard rendering in Claude Ollama Proxy.
"""

import unittest
import sys
import os
import re
from unittest.mock import patch, MagicMock

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modules after path setup
import claude_ollama_server
from claude_ollama_server import generate_dashboard_html, MetricsTracker

class TestDashboard(unittest.TestCase):
    """Tests for dashboard HTML generation."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a mock metrics tracker
        self.mock_metrics = MagicMock(spec=MetricsTracker)
        
        # Set up mock values for the metrics
        self.mock_metrics.get_uptime.return_value = 3600  # 1 hour
        self.mock_metrics.get_uptime_formatted.return_value = "1h 0m 0s"
        self.mock_metrics.start_time.isoformat.return_value = "2023-01-01T00:00:00"
        
        self.mock_metrics.total_invocations = 100
        self.mock_metrics.get_invocations_per_minute.return_value = 1.67
        self.mock_metrics.get_invocations_per_hour.return_value = 100
        self.mock_metrics.current_processes = 5
        self.mock_metrics.max_concurrent_processes = 10
        
        self.mock_metrics.first_invocation_time = "2023-01-01T00:01:00"
        self.mock_metrics.last_invocation_time = "2023-01-01T01:00:00"
        self.mock_metrics.last_completion_time = "2023-01-01T01:00:10"
        
        self.mock_metrics.avg_execution_time_ms = 5000  # 5 seconds
        self.mock_metrics.median_execution_time_ms = 4000  # 4 seconds
        
        self.mock_metrics.get_avg_memory_mb.return_value = 200
        self.mock_metrics.get_peak_memory_mb.return_value = 500
        self.mock_metrics.get_avg_cpu_percent.return_value = 25
        
        self.mock_metrics.total_input_tokens = 10000
        self.mock_metrics.total_output_tokens = 5000
        
        self.mock_metrics.total_cost = 0.50
        self.mock_metrics.avg_cost = 0.005
        
        # Set up conversation tracking mock data
        self.mock_metrics.active_conversations = {"conv1", "conv2", "conv3"}
        self.mock_metrics.unique_conversations = {"conv1", "conv2", "conv3", "conv4", "conv5"}
        
        # Mock the get_metrics method
        self.mock_metrics.get_metrics.return_value = {
            'uptime': {
                'seconds': 3600,
                'formatted': "1h 0m 0s",
                'start_time': "2023-01-01T00:00:00"
            },
            'claude_invocations': {
                'total': 100,
                'per_minute': 1.67,
                'per_hour': 100,
                'current_running': 5,
                'max_concurrent': 10
            },
            'timestamps': {
                'first_invocation': "2023-01-01T00:01:00",
                'last_invocation': "2023-01-01T01:00:00",
                'last_completion': "2023-01-01T01:00:10"
            },
            'performance': {
                'avg_execution_time_ms': 5000,
                'median_execution_time_ms': 4000
            },
            'resources': {
                'avg_memory_mb': 200,
                'peak_memory_mb': 500,
                'avg_cpu_percent': 25
            },
            'tokens': {
                'total_prompt': 10000,
                'total_completion': 5000,
                'total': 15000
            },
            'cost': {
                'total_cost': 0.50,
                'avg_cost': 0.005
            }
        }
        
        # Patch the global metrics object
        self.metrics_patcher = patch('claude_ollama_server.metrics', self.mock_metrics)
        self.metrics_patcher.start()
        
        # Patch the get_running_claude_processes function
        self.processes_patcher = patch('claude_ollama_server.get_running_claude_processes')
        self.mock_get_processes = self.processes_patcher.start()
        self.mock_get_processes.return_value = [
            {
                'pid': '12345',
                'cmd': 'claude -p',
                'start_time': '2023-01-01T00:59:00',
                'memory_mb': 300,
                'cpu_percent': 15,
                'runtime_seconds': 60
            },
            {
                'pid': '12346',
                'cmd': 'claude -p',
                'start_time': '2023-01-01T01:00:00',
                'memory_mb': 200,
                'cpu_percent': 10,
                'runtime_seconds': 10
            }
        ]
    
    def tearDown(self):
        """Clean up after each test."""
        self.metrics_patcher.stop()
        self.processes_patcher.stop()
    
    def test_dashboard_html_generation(self):
        """Test that the dashboard HTML generates correctly."""
        # Generate dashboard HTML
        html = generate_dashboard_html()
        
        # Basic structure checks
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("<html", html)
        self.assertIn("</html>", html)
        self.assertIn("<head>", html)
        self.assertIn("</body>", html)
        
        # Check for key sections
        self.assertIn("Claude Proxy Dashboard", html)
        self.assertIn("System Status", html)
        self.assertIn("Claude Usage", html)
        self.assertIn("Performance", html)
        self.assertIn("System Resources", html)
        self.assertIn("Running Processes", html)
        
        # Check for specific metrics
        self.assertIn("1h 0m 0s", html)  # Uptime
        self.assertIn("100", html)  # Total invocations
        self.assertIn("5", html)  # Current processes
        
        # Check for time formatting in minutes and seconds
        # The formatting should show "0m 5.0s" for 5000ms
        self.assertIn("0m 5.0s", html)  # Avg execution time
        
        # Check for conversation metrics
        self.assertIn("Active Conversations", html)
        self.assertIn("3", html)  # 3 active conversations
        self.assertIn("Total Conversations", html)
        self.assertIn("5", html)  # 5 total conversations
        
        # Check for running processes table
        self.assertIn("12345", html)  # Process ID
        self.assertIn("12346", html)  # Process ID
        
        # Check for auto-refresh functionality
        self.assertIn("autoRefresh", html)
        self.assertIn("setInterval", html)
    
    def test_error_handling_in_dashboard(self):
        """Test that the dashboard handles errors gracefully."""
        # Make the metrics.get_metrics method raise an exception
        self.mock_metrics.get_metrics.side_effect = Exception("Test error")
        
        # Generate dashboard HTML - should not raise an exception
        html = generate_dashboard_html()
        
        # Check for error message
        self.assertIn("Error generating dashboard", html)
        self.assertIn("Test error", html)

if __name__ == '__main__':
    unittest.main()