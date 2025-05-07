import asyncio
import datetime
import collections
import statistics


METRICS_HISTORY_SIZE = 1000  # Number of requests to keep for metrics calculations

class ClaudeMetrics:
    """
    Tracks metrics for Claude CLI process invocations.
    Collects data on Claude usage patterns, run times, and resource usage.
    """
    def __init__(self, history_size=METRICS_HISTORY_SIZE):
        # Claude invocation timestamps (ISO format)
        self.first_invocation_time = None
        self.last_invocation_time = None
        self.last_completion_time = None
        
        # Claude process performance tracking
        self.execution_durations = collections.deque(maxlen=history_size)  # in milliseconds
        self.invocation_times = collections.deque(maxlen=history_size)
        self.completion_times = collections.deque(maxlen=history_size)
        
        # Claude invocation volume
        self.total_invocations = 0
        self.invocations_by_minute = collections.defaultdict(int)
        self.invocations_by_hour = collections.defaultdict(int)
        self.invocations_by_day = collections.defaultdict(int)
        
        # Memory usage tracking (in MB)
        self.memory_usage = collections.deque(maxlen=history_size)
        
        # Claude tracking by type
        self.invocations_by_model = collections.defaultdict(int)
        self.invocations_by_conversation = collections.defaultdict(int)
        
        # Conversation tracking
        self.unique_conversations = set()
        self.active_conversations = set()  # Conversations seen in the last hour
        
        # File size totals
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        
        # Current concurrent Claude processes
        self.current_processes = 0
        self.max_concurrent_processes = 0
        
        # Error tracking
        self.errors = collections.deque(maxlen=50)  # Keep last 50 errors
        self.error_count = 0
        
        # Starting timestamp
        self.start_time = datetime.datetime.now()
        
        # Track system resource impact
        self.cpu_usage = collections.deque(maxlen=history_size)  # percentage
        
        # Track total Claude processes started during this server run
        self.total_claude_processes = 0
        
        # Synchronization lock for updating concurrent process count
        self._lock = asyncio.Lock() if 'asyncio' in globals() else None
    
    async def record_claude_start(self, process_id, model=None, conversation_id=None, memory_mb=None, cpu_percent=None):
        """Record a new Claude CLI process start"""
        now = datetime.datetime.now()
        iso_now = now.isoformat()
        
        # Update invocation timestamps
        if self.first_invocation_time is None:
            self.first_invocation_time = iso_now
        self.last_invocation_time = iso_now
        
        # Record invocation time
        self.invocation_times.append(now)
        
        # Update counters
        self.total_invocations += 1
        self.total_claude_processes += 1
        
        # Update time-based metrics
        minute_key = now.strftime("%Y-%m-%d %H:%M")
        hour_key = now.strftime("%Y-%m-%d %H")
        day_key = now.strftime("%Y-%m-%d")
        
        self.invocations_by_minute[minute_key] += 1
        self.invocations_by_hour[hour_key] += 1
        self.invocations_by_day[day_key] += 1
        
        # Update model metrics
        if model:
            self.invocations_by_model[model] += 1
        
        # Update conversation metrics
        if conversation_id:
            # Ensure we have the tracking dictionary
            if not hasattr(self, 'conversation_last_seen'):
                self.conversation_last_seen = {}
                
            # Add to sets for tracking
            self.unique_conversations.add(conversation_id)
            self.active_conversations.add(conversation_id)
            
            # Update conversation activity timestamp
            self.conversation_last_seen[conversation_id] = now
            
            # Increment counter
            self.invocations_by_conversation[conversation_id] += 1
        
        # Update resource usage metrics if available
        if memory_mb is not None:
            self.memory_usage.append(memory_mb)
        
        if cpu_percent is not None:
            self.cpu_usage.append(cpu_percent)
        
        # Update concurrent process count
        if self._lock:
            async with self._lock:
                self.current_processes += 1
                if self.current_processes > self.max_concurrent_processes:
                    self.max_concurrent_processes = self.current_processes
        else:
            self.current_processes += 1
            if self.current_processes > self.max_concurrent_processes:
                self.max_concurrent_processes = self.current_processes
    
    async def record_claude_completion(self, process_id, duration_ms, output_tokens=None, memory_mb=None, error=None, conversation_id=None):
        """Record completion of a Claude CLI process"""
        now = datetime.datetime.now()
        self.last_completion_time = now.isoformat()
        
        # Store the execution duration
        self.completion_times.append(now)
        self.execution_durations.append(duration_ms)
        
        # Update token counts if available
        if output_tokens is not None:
            self.total_completion_tokens += output_tokens
        
        # Update resource usage metrics if available
        if memory_mb is not None:
            self.memory_usage.append(memory_mb)
        
        # Record errors if any
        if error:
            self.errors.append({
                'time': now.isoformat(),
                'error': str(error)
            })
            self.error_count += 1
        
        # Update concurrent process count
        if self._lock:
            async with self._lock:
                if self.current_processes > 0:
                    self.current_processes -= 1
        else:
            if self.current_processes > 0:
                self.current_processes -= 1
        
        # Update conversation activity timestamp if provided
        if conversation_id:
            # Ensure we have the tracking dictionary
            if not hasattr(self, 'conversation_last_seen'):
                self.conversation_last_seen = {}
                
            # Update activity timestamp
            self.conversation_last_seen[conversation_id] = now
            
        # Note: We don't remove from active_conversations here because
        # a conversation can have multiple processes. Instead, we have
        # a separate pruning mechanism that removes old conversations.
    
    def prune_old_data(self):
        """Remove old data from time-based metrics to prevent memory leaks"""
        now = datetime.datetime.now()
        
        # Track conversation last activity times in a separate dictionary
        # We'll use this if we don't already have a tracking mechanism
        if not hasattr(self, 'conversation_last_seen'):
            self.conversation_last_seen = {}
            # Initialize with current conversations
            for conv_id in self.active_conversations:
                self.conversation_last_seen[conv_id] = now
        
        # Update timestamps for all active conversations
        for conv_id in list(self.active_conversations):
            self.conversation_last_seen[conv_id] = now
            
        # Remove conversations inactive for more than 1 hour
        one_hour_ago = now - datetime.timedelta(hours=1)
        
        # Find conversations to remove
        inactive_conversations = []
        for conv_id, last_seen in self.conversation_last_seen.items():
            # Check if it's older than our retention period
            if isinstance(last_seen, datetime.datetime) and last_seen < one_hour_ago:
                inactive_conversations.append(conv_id)
        
        # Clean up old conversations
        for conv_id in inactive_conversations:
            # Remove from active set
            self.active_conversations.discard(conv_id)
            # Remove from tracking
            self.conversation_last_seen.pop(conv_id, None)
            
        # Also clean up old invocation tracking data to prevent memory leaks
        # Only keep data from the last day
        cutoff_date = now - datetime.timedelta(days=1)
        date_cutoff = cutoff_date.strftime("%Y-%m-%d")
        
        # Clean up by-minute metrics (keep last 2 hours)
        two_hours_ago = now - datetime.timedelta(hours=2)
        minute_cutoff = two_hours_ago.strftime("%Y-%m-%d %H:%M")
        
        # Clean up data by removing old keys
        for minute_key in list(self.invocations_by_minute.keys()):
            if minute_key < minute_cutoff:
                del self.invocations_by_minute[minute_key]
                
        # Clean up by-hour metrics (keep last day)
        hour_cutoff = cutoff_date.strftime("%Y-%m-%d %H")
        for hour_key in list(self.invocations_by_hour.keys()):
            if hour_key < hour_cutoff:
                del self.invocations_by_hour[hour_key]
                
        # Clean up by-day metrics (keep last month)
        month_ago = now - datetime.timedelta(days=30)
        day_cutoff = month_ago.strftime("%Y-%m-%d")
        for day_key in list(self.invocations_by_day.keys()):
            if day_key < day_cutoff:
                del self.invocations_by_day[day_key]
                
        # Log cleanup metrics
        logger.debug(f"Pruned {len(inactive_conversations)} inactive conversations, {len(self.active_conversations)} remain active")
    
    def get_avg_execution_time(self):
        """Get the average execution time in milliseconds"""
        if not self.execution_durations:
            return 0
        return statistics.mean(self.execution_durations)
    
    def get_median_execution_time(self):
        """Get the median execution time in milliseconds"""
        if not self.execution_durations:
            return 0
        return statistics.median(self.execution_durations)
    
    def get_invocations_per_minute(self, minutes=5):
        """Get the average invocations per minute over the last N minutes"""
        now = datetime.datetime.now()
        count = 0
        
        # Count invocations in the window
        for inv_time in self.invocation_times:
            if (now - inv_time).total_seconds() <= (minutes * 60):
                count += 1
        
        # Avoid division by zero
        if minutes == 0:
            return 0
        
        return count / minutes
    
    def get_invocations_per_hour(self):
        """Get the average invocations per hour over the last hour"""
        return self.get_invocations_per_minute(60)
    
    def get_avg_memory_usage(self):
        """Get the average memory usage in MB"""
        if not self.memory_usage:
            return 0
        return statistics.mean(self.memory_usage)
    
    def get_peak_memory_usage(self):
        """Get the peak memory usage in MB"""
        if not self.memory_usage:
            return 0
        return max(self.memory_usage)
    
    def get_avg_cpu_usage(self):
        """Get the average CPU usage percentage"""
        if not self.cpu_usage:
            return 0
        return statistics.mean(self.cpu_usage)
    
    def get_uptime(self):
        """Get the uptime in seconds"""
        return (datetime.datetime.now() - self.start_time).total_seconds()
    
    def get_uptime_formatted(self):
        """Get the uptime as a formatted string (e.g., '2 days, 3 hours, 4 minutes')"""
        uptime_seconds = self.get_uptime()
        
        days, remainder = divmod(uptime_seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        parts = []
        if days > 0:
            parts.append(f"{int(days)} days")
        if hours > 0 or days > 0:
            parts.append(f"{int(hours)} hours")
        if minutes > 0 or hours > 0 or days > 0:
            parts.append(f"{int(minutes)} minutes")
        if not parts:
            parts.append(f"{int(seconds)} seconds")
        
        return ", ".join(parts)
    
    def get_metrics(self):
        """Get all metrics as a dictionary"""
        return {
            'uptime': {
                'seconds': self.get_uptime(),
                'formatted': self.get_uptime_formatted(),
                'start_time': self.start_time.isoformat()
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
                'avg_execution_time_ms': self.get_avg_execution_time(),
                'median_execution_time_ms': self.get_median_execution_time()
            },
            'resources': {
                'avg_memory_mb': self.get_avg_memory_usage(),
                'peak_memory_mb': self.get_peak_memory_usage(),
                'avg_cpu_percent': self.get_avg_cpu_usage()
            },
            'conversations': {
                'unique_count': len(self.unique_conversations),
                'active_count': len(self.active_conversations)
            },
            'tokens': {
                'total_completion': self.total_completion_tokens
            },
            'errors': {
                'count': self.error_count,
                'recent': list(self.errors)
            },
            'distribution': {
                'by_model': dict(self.invocations_by_model),
                'by_conversation': dict(self.invocations_by_conversation)
            }
        }
