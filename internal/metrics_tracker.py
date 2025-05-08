#!/usr/bin/env python3
"""
Metrics adapter for Claude Ollama Proxy.
This module provides compatibility with the old MetricsTracker interface
to ensure tests continue to work as expected.
"""

import datetime
import statistics
from typing import Dict, Optional, Any, Set, Deque, List
from internal.claude_metrics import ClaudeMetrics

class MetricsTracker:
    """
    Adapter class that provides the old MetricsTracker interface 
    while using the new ClaudeMetrics implementation.
    
    This ensures backwards compatibility with existing tests.
    """
    
    def __init__(self, metrics: Optional[ClaudeMetrics] = None):
        """
        Initialize a new MetricsTracker instance.
        
        This creates a wrapper around the global ClaudeMetrics instance
        from claude_ollama_server.
        """
        # Use the global metrics instance from claude_ollama_server
        # Most methods will delegate to this instance
        self._metrics = metrics
        
        # Only initialize properties that aren't directly accessed from the underlying metrics
        self.total_cost = 0.0
        self.avg_cost = 0.0
        
        # For test fallbacks, initialize _execution_durations if needed
        self._execution_durations = []
        
        # Reference to start time for uptime calculation (used as fallback)
        try:
            self._fallback_start_time = datetime.datetime.now()
        except (AttributeError, TypeError):
            self._fallback_start_time = datetime.datetime.now()
            
    # Dynamically access metrics properties from the underlying metrics object
    @property
    def total_invocations(self):
        return getattr(self._metrics, 'total_invocations', 0)
    
    @property
    def current_processes(self):
        return getattr(self._metrics, 'current_processes', 0)
    
    @current_processes.setter
    def current_processes(self, value):
        if self._metrics:
            self._metrics.current_processes = value
    
    @property
    def max_concurrent_processes(self):
        return getattr(self._metrics, 'max_concurrent_processes', 0)
    
    @property
    def active_conversations(self):
        return getattr(self._metrics, 'active_conversations', set())
    
    @property
    def unique_conversations(self):
        return getattr(self._metrics, 'unique_conversations', set())
    
    @property
    def first_invocation_time(self):
        return getattr(self._metrics, 'first_invocation_time', None)
    
    @property
    def last_invocation_time(self):
        return getattr(self._metrics, 'last_invocation_time', None)
    
    @property
    def last_completion_time(self):
        return getattr(self._metrics, 'last_completion_time', None)
    
    @property
    def total_input_tokens(self):
        return getattr(self._metrics, 'total_prompt_tokens', 0)
    
    @property
    def total_output_tokens(self):
        return getattr(self._metrics, 'total_completion_tokens', 0)
    
    @property
    def avg_execution_time_ms(self):
        try:
            return self._metrics.get_avg_execution_time()
        except (AttributeError, TypeError, ValueError):
            # Fallback to our local calculation
            if hasattr(self, '_execution_durations') and self._execution_durations:
                return sum(self._execution_durations) / len(self._execution_durations)
            return 0
    
    @property
    def median_execution_time_ms(self):
        try:
            if hasattr(self._metrics, 'execution_durations'):
                times = list(self._metrics.execution_durations)
                if times and all(t is not None for t in times):
                    return statistics.median(times)
        except (TypeError, ValueError, AttributeError):
            # Fallback to our local calculation
            if hasattr(self, '_execution_durations') and self._execution_durations:
                return statistics.median(self._execution_durations)
        return 0
    
    @property
    def start_time(self):
        return getattr(self._metrics, 'start_time', self._fallback_start_time)
    
    # Uptime methods
    def get_uptime(self) -> float:
        """Get the server uptime in seconds."""
        try:
            return self._metrics.get_uptime()
        except (AttributeError, TypeError):
            # Fallback implementation for tests
            delta = datetime.datetime.now() - self.start_time
            return delta.total_seconds()
    
    def get_uptime_formatted(self) -> str:
        """Get the server uptime as a formatted string."""
        try:
            return self._metrics.get_uptime_formatted()
        except (AttributeError, TypeError):
            # Fallback implementation for tests
            uptime_seconds = self.get_uptime()
            hours, remainder = divmod(uptime_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    
    # Process tracking methods
    async def record_claude_start(self, process_id, model=None, conversation_id=None, 
                                 memory_mb=None, cpu_percent=None):
        """Record the start of a Claude process."""
        try:
            await self._metrics.record_claude_start(
                process_id, model, conversation_id, memory_mb, cpu_percent
            )
            # Update the local properties
            self.total_invocations = getattr(self._metrics, 'total_invocations', self.total_invocations + 1)
            self.current_processes = getattr(self._metrics, 'current_processes', self.current_processes + 1)
            self.max_concurrent_processes = max(self.max_concurrent_processes, self.current_processes)
            if conversation_id:
                self.active_conversations = getattr(self._metrics, 'active_conversations', self.active_conversations)
                self.unique_conversations = getattr(self._metrics, 'unique_conversations', self.unique_conversations)
                # Add to conversation sets
                if isinstance(self.active_conversations, set):
                    self.active_conversations.add(conversation_id)
                if isinstance(self.unique_conversations, set):
                    self.unique_conversations.add(conversation_id)
        except (AttributeError, TypeError):
            # Fallback for tests
            self.total_invocations += 1
            self.current_processes += 1
            self.max_concurrent_processes = max(self.max_concurrent_processes, self.current_processes)
            if conversation_id:
                if isinstance(self.active_conversations, set):
                    self.active_conversations.add(conversation_id)
                if isinstance(self.unique_conversations, set):
                    self.unique_conversations.add(conversation_id)
    
    async def record_claude_end(self, process_id, model=None, conversation_id=None, 
                               duration_ms=None, cost=None, tokens_in=None, tokens_out=None):
        """Record the end of a Claude process."""
        try:
            # Since the original doesn't have record_claude_end, we'll use record_claude_completion
            await self._metrics.record_claude_completion(
                process_id, duration_ms, tokens_out, None, None, conversation_id
            )
            
            # Update the local properties
            self.current_processes = getattr(self._metrics, 'current_processes', max(0, self.current_processes - 1))
            self.active_conversations = getattr(self._metrics, 'active_conversations', self.active_conversations)
        except (AttributeError, TypeError):
            # Fallback for tests
            self.current_processes = max(0, self.current_processes - 1)
        
        # Update token and cost metrics
        if tokens_in:
            self.total_input_tokens += tokens_in
        if tokens_out:
            self.total_output_tokens += tokens_out
        if cost:
            self.total_cost += cost
            # Recalculate average cost
            if self.total_invocations > 0:
                self.avg_cost = self.total_cost / self.total_invocations
        
        # Update performance metrics if duration is provided
        if duration_ms:
            # We'll track this ourselves for testing since the original might not exist
            if not hasattr(self, '_execution_durations'):
                self._execution_durations = []
            self._execution_durations.append(duration_ms)
            
            # Calculate avg and median
            self.avg_execution_time_ms = sum(self._execution_durations) / len(self._execution_durations)
            self.median_execution_time_ms = statistics.median(self._execution_durations)
    
    # Resource usage methods
    def get_avg_memory_mb(self) -> float:
        """Get the average memory usage in MB."""
        try:
            return self._metrics.get_avg_memory_usage()
        except (AttributeError, TypeError):
            return 0.0
    
    def get_peak_memory_mb(self) -> float:
        """Get the peak memory usage in MB."""
        try:
            return self._metrics.get_peak_memory_usage()
        except (AttributeError, TypeError):
            return 0.0
    
    def get_avg_cpu_percent(self) -> float:
        """Get the average CPU usage percentage."""
        try:
            return self._metrics.get_avg_cpu_usage()
        except (AttributeError, TypeError):
            return 0.0
    
    # Invocation rate methods
    def get_invocations_per_minute(self) -> float:
        """Get the average number of Claude invocations per minute."""
        uptime_minutes = max(1, self.get_uptime() / 60)
        return self.total_invocations / uptime_minutes
    
    def get_invocations_per_hour(self) -> float:
        """Get the average number of Claude invocations per hour."""
        uptime_hours = max(1, self.get_uptime() / 3600)
        return self.total_invocations / uptime_hours
    
    # Main metrics method
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics as a dictionary."""
        try:
            # Try to get metrics from the ClaudeMetrics instance
            metrics_data = self._metrics.get_metrics()
            
            # Add any additional metrics needed by tests
            # Always add cost data, overwriting if it exists
            metrics_data['cost'] = {
                'total_cost': self.total_cost,
                'avg_cost': self.avg_cost
            }
            
            # Ensure performance metrics include median
            if 'performance' in metrics_data:
                metrics_data['performance']['median_execution_time_ms'] = self.median_execution_time_ms
            
            return metrics_data
        except (AttributeError, TypeError):
            # Fallback implementation for tests
            return {
                'uptime': {
                    'seconds': self.get_uptime(),
                    'formatted': self.get_uptime_formatted(),
                    'start_time': self.start_time.isoformat() if hasattr(self.start_time, 'isoformat') else str(self.start_time)
                },
                'claude_invocations': {
                    'total': self.total_invocations,
                    'per_minute': self.get_invocations_per_minute(),
                    'per_hour': self.get_invocations_per_hour(),
                    'current_running': self.current_processes,
                    'max_concurrent': self.max_concurrent_processes
                },
                'timestamps': {
                    'first_invocation': self.first_invocation_time,
                    'last_invocation': self.last_invocation_time,
                    'last_completion': self.last_completion_time
                },
                'performance': {
                    'avg_execution_time_ms': self.avg_execution_time_ms,
                    'median_execution_time_ms': self.median_execution_time_ms
                },
                'resources': {
                    'avg_memory_mb': self.get_avg_memory_mb(),
                    'peak_memory_mb': self.get_peak_memory_mb(),
                    'avg_cpu_percent': self.get_avg_cpu_percent()
                },
                'tokens': {
                    'total_prompt': self.total_input_tokens,
                    'total_completion': self.total_output_tokens,
                    'total': self.total_input_tokens + self.total_output_tokens
                },
                'cost': {
                    'total_cost': self.total_cost,
                    'avg_cost': self.avg_cost
                }
            }