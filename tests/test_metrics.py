#!/usr/bin/env python3
"""
Unit tests for the metrics module of Claude Ollama Proxy.
"""

import unittest
import sys
import os
import time
import asyncio
from datetime import datetime, timedelta

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from claude_ollama_server import MetricsTracker

class TestMetricsTracker(unittest.TestCase):
    """Tests for the MetricsTracker class."""
    
    def setUp(self):
        """Set up a fresh metrics tracker for each test."""
        self.metrics = MetricsTracker()
    
    def test_initialization(self):
        """Test that the metrics tracker initializes correctly."""
        self.assertEqual(self.metrics.total_invocations, 0)
        self.assertEqual(self.metrics.current_processes, 0)
        self.assertEqual(self.metrics.max_concurrent_processes, 0)
        self.assertEqual(len(self.metrics.active_conversations), 0)
        self.assertEqual(len(self.metrics.unique_conversations), 0)
    
    def test_uptime(self):
        """Test that uptime calculation works correctly."""
        # Uptime should be small but positive
        uptime = self.metrics.get_uptime()
        self.assertGreaterEqual(uptime, 0)
        self.assertLess(uptime, 10)  # Should be less than 10 seconds for a new instance
        
        # Test formatted uptime
        formatted = self.metrics.get_uptime_formatted()
        self.assertIsInstance(formatted, str)
    
    def test_record_claude_start_end(self):
        """Test recording of Claude process start and end."""
        # Use asyncio.run to run the async methods
        async def test_async():
            # Record a process start
            await self.metrics.record_claude_start('test-process-1', 'claude-3.7-sonnet', 'conv-123')
            
            # Check that metrics were updated
            self.assertEqual(self.metrics.total_invocations, 1)
            self.assertEqual(self.metrics.current_processes, 1)
            self.assertEqual(self.metrics.max_concurrent_processes, 1)
            self.assertEqual(len(self.metrics.active_conversations), 1)
            self.assertEqual(len(self.metrics.unique_conversations), 1)
            
            # Record another process in the same conversation
            await self.metrics.record_claude_start('test-process-2', 'claude-3.7-sonnet', 'conv-123')
            
            # Check metrics again
            self.assertEqual(self.metrics.total_invocations, 2)
            self.assertEqual(self.metrics.current_processes, 2)
            self.assertEqual(self.metrics.max_concurrent_processes, 2)
            self.assertEqual(len(self.metrics.active_conversations), 1)  # Still just one conversation
            self.assertEqual(len(self.metrics.unique_conversations), 1)
            
            # Record a process end
            await self.metrics.record_claude_end('test-process-1', 'claude-3.7-sonnet', 'conv-123', 
                                                cost=0.01, tokens_in=100, tokens_out=50)
            
            # Check metrics after end
            self.assertEqual(self.metrics.current_processes, 1)  # One process still running
            self.assertEqual(self.metrics.max_concurrent_processes, 2)  # Max remains at 2
            self.assertEqual(self.metrics.total_cost, 0.01)
            self.assertEqual(self.metrics.total_input_tokens, 100)
            self.assertEqual(self.metrics.total_output_tokens, 50)
        
        asyncio.run(test_async())
    
    def test_conversation_tracking(self):
        """Test conversation tracking functionality."""
        async def test_async():
            # Start processes with different conversation IDs
            await self.metrics.record_claude_start('test-process-1', 'claude-3.7-sonnet', 'conv-1')
            await self.metrics.record_claude_start('test-process-2', 'claude-3.7-sonnet', 'conv-2')
            await self.metrics.record_claude_start('test-process-3', 'claude-3.7-sonnet', 'conv-3')
            
            # Check conversation counts
            self.assertEqual(len(self.metrics.active_conversations), 3)
            self.assertEqual(len(self.metrics.unique_conversations), 3)
            
            # End some processes and see if active conversations are updated
            await self.metrics.record_claude_end('test-process-1', 'claude-3.7-sonnet', 'conv-1')
            await self.metrics.record_claude_end('test-process-2', 'claude-3.7-sonnet', 'conv-2')
            
            # Active should decrease, unique should stay the same
            self.assertEqual(len(self.metrics.active_conversations), 1)
            self.assertEqual(len(self.metrics.unique_conversations), 3)
            
            # Start a new process in an existing conversation
            await self.metrics.record_claude_start('test-process-4', 'claude-3.7-sonnet', 'conv-2')
            
            # Active should increase, unique should stay the same
            self.assertEqual(len(self.metrics.active_conversations), 2)
            self.assertEqual(len(self.metrics.unique_conversations), 3)
        
        asyncio.run(test_async())
    
    def test_performance_metrics(self):
        """Test calculation of performance metrics."""
        async def test_async():
            # Record a few executions with different durations
            await self.metrics.record_claude_start('test-process-1', 'claude-3.7-sonnet', 'conv-1')
            await self.metrics.record_claude_end('test-process-1', 'claude-3.7-sonnet', 'conv-1', 
                                               duration_ms=1000)  # 1 second
            
            await self.metrics.record_claude_start('test-process-2', 'claude-3.7-sonnet', 'conv-1')
            await self.metrics.record_claude_end('test-process-2', 'claude-3.7-sonnet', 'conv-1', 
                                               duration_ms=3000)  # 3 seconds
            
            await self.metrics.record_claude_start('test-process-3', 'claude-3.7-sonnet', 'conv-1')
            await self.metrics.record_claude_end('test-process-3', 'claude-3.7-sonnet', 'conv-1', 
                                               duration_ms=2000)  # 2 seconds
            
            # Check average and median execution times
            self.assertEqual(self.metrics.avg_execution_time_ms, 2000)  # (1000 + 3000 + 2000) / 3
            self.assertEqual(self.metrics.median_execution_time_ms, 2000)  # Median of [1000, 2000, 3000]
        
        asyncio.run(test_async())
    
    def test_resource_metrics(self):
        """Test tracking of resource usage metrics."""
        async def test_async():
            # Record processes with different memory and CPU usage
            await self.metrics.record_claude_start('test-process-1', memory_mb=100, cpu_percent=10)
            await self.metrics.record_claude_start('test-process-2', memory_mb=200, cpu_percent=20)
            await self.metrics.record_claude_start('test-process-3', memory_mb=300, cpu_percent=30)
            
            # Check resource metrics
            self.assertEqual(self.metrics.get_avg_memory_mb(), 200)  # (100 + 200 + 300) / 3
            self.assertEqual(self.metrics.get_peak_memory_mb(), 300)
            self.assertEqual(self.metrics.get_avg_cpu_percent(), 20)  # (10 + 20 + 30) / 3
        
        asyncio.run(test_async())
    
    def test_get_metrics(self):
        """Test the get_metrics method returns a complete dictionary."""
        metrics_data = self.metrics.get_metrics()
        
        # Check that all expected sections are present
        self.assertIn('uptime', metrics_data)
        self.assertIn('claude_invocations', metrics_data)
        self.assertIn('timestamps', metrics_data)
        self.assertIn('performance', metrics_data)
        self.assertIn('resources', metrics_data)
        self.assertIn('tokens', metrics_data)
        self.assertIn('cost', metrics_data)
        
        # Check specific values
        self.assertIsInstance(metrics_data['uptime']['seconds'], float)
        self.assertIsInstance(metrics_data['uptime']['formatted'], str)
        self.assertEqual(metrics_data['claude_invocations']['total'], 0)
        self.assertEqual(metrics_data['claude_invocations']['current_running'], 0)

if __name__ == '__main__':
    unittest.main()