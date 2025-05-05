"""
Claude-compatible Ollama API Server.

This FastAPI application provides an OpenAI-compatible API server that interfaces with
a locally running Claude Code process. It implements both Ollama API and OpenAI API format.
"""

import inspect
import json
import logging
import os
import pprint
import subprocess
import shlex
import time
import traceback
import uuid
import collections
import statistics
import datetime
import tempfile
import shutil
from typing import Dict, List, Optional, Any, Union, Deque
from pydantic import BaseModel, Field
import httpx
import asyncio
import fastapi
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse, HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from starlette.background import BackgroundTask

# Configuration

DEFAULT_MODEL = "anthropic/claude-3.7-sonnet"  # Default Claude model to report (must match /models endpoint)
DEFAULT_MAX_TOKENS = 1000000  # 128k context length
CONVERSATION_CACHE_TTL = 3600 * 3  # 3 hours in seconds
METRICS_HISTORY_SIZE = 1000  # Number of requests to keep for metrics calculations

# Configure logging with more details

logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG to get more detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Metrics tracking class

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

# Initialize metrics
metrics = ClaudeMetrics()

def debug_response(response, prefix=""):
    """Helper to log response details for debugging"""
    try:
        if isinstance(response, dict):
            pretty = pprint.pformat(response, indent=2)
            logger.debug(f"{prefix} Response dict: {pretty}")
        else:
            logger.debug(f"{prefix} Response (not dict): {response}")
    except Exception as e:
        logger.error(f"Error in debug_response: {e}")

# Conversation cache to track active conversations
# Keys are conversation IDs, values are (timestamp, conversation_id) tuples
conversation_cache = {}

# Directory for storing temporary conversation directories
CONV_TEMP_ROOT = os.path.join(tempfile.gettempdir(), "claude_conversations")
os.makedirs(CONV_TEMP_ROOT, exist_ok=True)

# Map of conversation IDs to their temporary directories
conversation_temp_dirs = {}

# Initialize FastAPI app

app = FastAPI(
    title="Claude Ollama API",
    description="OpenAI-compatible API server for Claude Code",
    version="0.1.0",
)

# Background task for cleanup
@app.on_event("startup")
async def startup_event():
    """Run background tasks when the server starts."""
    # Start background cleanup task
    asyncio.create_task(periodic_cleanup())

async def periodic_cleanup():
    """Periodically clean up old data to prevent memory leaks."""
    while True:
        try:
            # Clean up every 5 minutes
            await asyncio.sleep(5 * 60)  # 5 minutes
            
            # Clean up old metrics data
            metrics.prune_old_data()
            logger.debug("Performed periodic cleanup of old metrics data")
            
            # Clean up old conversation temp directories that are no longer active
            try:
                active_conversations = set()
                # Get active conversations from cache
                for request_id, (_, conv_id) in conversation_cache.items():
                    active_conversations.add(conv_id)
                
                # Clean up temp directories for inactive conversations
                for conv_id in list(conversation_temp_dirs.keys()):
                    if conv_id not in active_conversations:
                        cleanup_conversation_temp_dir(conv_id)
                        logger.debug(f"Cleaned up temp directory for inactive conversation: {conv_id}")
            except Exception as e:
                logger.error(f"Error cleaning up conversation temp dirs: {e}")
        except Exception as e:
            logger.error(f"Error in periodic cleanup: {e}")
            # Don't let the background task die
            await asyncio.sleep(60)

# Add CORS middleware to allow requests from any origin
# This is critical for working with web-based clients like OpenWebUI

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins - change to specific domains in production
    allow_credentials=False,  # Changed to False to avoid conflicts when allow_origins=["*"]
    allow_methods=["GET", "POST", "OPTIONS", "HEAD"],  # Explicitly specify allowed methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["Content-Type", "X-Request-ID", "Content-Length"],  # Expose specific headers
    max_age=86400  # Cache preflight requests for 24 hours
)

# Add middleware for global exception and request logging

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests and uncaught exceptions."""
    request_id = str(id(request))
    method = request.method
    url = str(request.url)
    path = request.url.path
    start_time = time.time()
    
    # More detailed logging for chat completions (used by OpenWebUI)
    is_chat_completion = path in ["/chat/completions", "/v1/chat/completions"]
    
    # Log request details
    logger.info(f"Request {request_id}: {method} {url}")
    
    # Enhanced logging for chat completions
    if is_chat_completion:
        try:
            # Log headers for debugging
            headers = dict(request.headers.items())
            user_agent = headers.get("user-agent", "Unknown")
            origin = headers.get("origin", "Unknown")
            
            logger.info(f"Chat completion request from: UA={user_agent}, Origin={origin}")
            
            # Try to read the request body (without consuming it)
            body = await request.body()
            # Create a new request with the same body
            request = Request(request.scope, request.receive)
            
            # Try to parse and log key parts of the request
            try:
                content = body.decode()
                logger.debug(f"Request body: {content[:200]}...")
                
                # Try to parse as JSON and extract key fields
                try:
                    data = json.loads(content)
                    model = data.get("model", "not-specified")
                    stream = data.get("stream", False)
                    logger.info(f"Chat completion details: model={model}, stream={stream}")
                except:
                    pass
            except:
                pass
        except Exception as e:
            logger.warning(f"Error logging chat completion details: {e}")

    try:
        # Process the request
        response = await call_next(request)
        
        # Calculate request duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Log response code
        logger.info(f"Response {request_id}: {response.status_code} ({int(duration_ms)}ms)")
        
        # Enhanced logging for chat completions
        if is_chat_completion:
            # Log headers for debugging
            headers = dict(response.headers.items())
            content_type = headers.get("content-type", "Unknown")
            logger.info(f"Chat completion response: Content-Type={content_type}")
        
        return response
        
    except Exception as e:
        # Log any uncaught exceptions
        logger.error(f"Uncaught exception in {method} {url}: {str(e)}", exc_info=True)
        
        # Return a 500 response
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal Server Error: {str(e)}"}
        )

# Utils

def get_iso_timestamp():
    """Generate ISO-8601 timestamp for created_at field."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat("T", "milliseconds") + "Z"

# Command to interact with Claude Code CLI

# Try to find claude executable
def find_claude_command():
    """Find the claude command executable path"""
    try:
        # First try the simple case - claude in PATH
        if subprocess.run(['which', 'claude'], capture_output=True, text=True, check=False).returncode == 0:
            return "claude"
        
        # Next, try common installation locations
        common_paths = [
            "/opt/homebrew/bin/claude",
            "/usr/local/bin/claude",
            os.path.expanduser("~/.local/bin/claude")
        ]
        
        for path in common_paths:
            if os.path.exists(path) and os.access(path, os.X_OK):
                logger.info(f"Found claude at: {path}")
                return path
        
        # Last resort: try to run command discovery via shell
        try:
            result = subprocess.run(['bash', '-c', 'which claude'], 
                                   capture_output=True, text=True, check=False)
            if result.returncode == 0 and result.stdout.strip():
                path = result.stdout.strip()
                logger.info(f"Found claude via shell at: {path}")
                return path
        except:
            pass
            
        # If we're here, we couldn't find claude
        logger.warning("Could not find claude command. Default to 'claude' and hope for the best.")
        return "claude"
    except Exception as e:
        logger.error(f"Error finding claude command: {e}")
        return "claude"  # Fallback to simple command name

CLAUDE_CMD = find_claude_command()

# Conversation management functions

def get_conversation_id(request_id: str) -> str:
    """
    Get the conversation ID for a request. If it exists in the cache,
    return the cached conversation ID. Otherwise, return None.
    Also cleans up expired conversations.
    """
    current_time = time.time()

    # Clean up expired conversations
    expired_keys = []
    for key, (timestamp, _) in conversation_cache.items():
        if current_time - timestamp > CONVERSATION_CACHE_TTL:
            expired_keys.append(key)

    for key in expired_keys:
        logger.info(f"Removing expired conversation: {key}")
        del conversation_cache[key]

    # Return the conversation ID if it exists
    if request_id in conversation_cache:
        conversation_cache[request_id] = (current_time, conversation_cache[request_id][1])  # Update timestamp
        return conversation_cache[request_id][1]

    return None

def set_conversation_id(request_id: str, conversation_id: str):
    """
    Store a conversation ID in the cache.
    """
    current_time = time.time()
    conversation_cache[request_id] = (current_time, conversation_id)
    logger.info(f"Stored conversation ID {conversation_id} for request {request_id}")

# Pydantic models for request/response validation

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = DEFAULT_MODEL
    messages: List[ChatMessage]
    stream: bool = True
    options: Optional[Dict[str, Any]] = None
    system: Optional[str] = None
    format: Optional[str] = None
    template: Optional[str] = None
    keep_alive: Optional[str] = None
    id: Optional[str] = None  # Request ID
    conversation_id: Optional[str] = None  # Explicit conversation ID

# OpenAI-compatible models for request validation

class OpenAIChatMessage(BaseModel):
    role: str
    content: str

class OpenAIChatRequest(BaseModel):
    model: str = DEFAULT_MODEL
    messages: List[OpenAIChatMessage]
    stream: bool = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    id: Optional[str] = None  # Request ID
    user: Optional[str] = None  # OpenAI field
    conversation_id: Optional[str] = None  # Explicit conversation ID

# Utility functions for working with Claude Code CLI

def get_conversation_temp_dir(conversation_id: str) -> str:
    """Get or create a temporary directory for a conversation."""
    if conversation_id in conversation_temp_dirs:
        # Return existing temp dir
        temp_dir = conversation_temp_dirs[conversation_id]
        if os.path.exists(temp_dir):
            return temp_dir
    
    # Create a new temp dir for this conversation
    temp_dir = os.path.join(CONV_TEMP_ROOT, f"conv_{conversation_id}_{uuid.uuid4().hex[:8]}")
    os.makedirs(temp_dir, exist_ok=True)
    conversation_temp_dirs[conversation_id] = temp_dir
    logger.info(f"Created temporary directory for conversation {conversation_id}: {temp_dir}")
    return temp_dir

def cleanup_conversation_temp_dir(conversation_id: str):
    """Clean up the temporary directory for a conversation."""
    if conversation_id in conversation_temp_dirs:
        temp_dir = conversation_temp_dirs[conversation_id]
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory for conversation {conversation_id}: {temp_dir}")
            except Exception as e:
                logger.error(f"Failed to clean up temporary directory {temp_dir}: {e}")
        # Remove from the map regardless of success
        del conversation_temp_dirs[conversation_id]

async def run_claude_command(prompt: str, conversation_id: str = None) -> str:
    """Run a Claude Code command and return the output."""
    # Base command
    base_cmd = f"{CLAUDE_CMD} -p"

    # Add conversation flag if there's a conversation ID
    if conversation_id:
        base_cmd += f" -c {conversation_id}"
        
        # Check if we're in test mode (only create temp dirs in production)
        is_test = 'unittest' in sys.modules or os.environ.get('TESTING') == '1'
        
        if not is_test:
            # Create or get a temporary directory for this conversation
            temp_dir = get_conversation_temp_dir(conversation_id)
            
            # Set the current working directory for this conversation
            # We'll use environment variable to pass it to the Claude CLI
            os.environ["CLAUDE_CWD"] = temp_dir

    # Use regular JSON output format
    cmd = f"{base_cmd} --output-format json"

    logger.info(f"Running command: {cmd}")
    
    # Generate a unique process ID for tracking
    process_id = f"claude-process-{uuid.uuid4().hex[:8]}"
    start_time = time.time()
    model = DEFAULT_MODEL
    
    # Record the Claude process start in metrics
    await metrics.record_claude_start(process_id, model, conversation_id)

    process = None
    try:
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Track this process
        if process and process.pid:
            track_claude_process(str(process.pid), cmd)
        
        # Send the prompt to Claude
        stdout, stderr = await process.communicate(input=prompt.encode())
        
        # Calculate execution duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Store the process output before untracking
        if process and process.pid:
            try:
                stdout_text = stdout.decode() if stdout else ""
                stderr_text = stderr.decode() if stderr else ""
                # Try to parse the response as JSON
                response_obj = None
                try:
                    if stdout_text and stdout_text.strip().startswith('{'):
                        response_obj = json.loads(stdout_text)
                except json.JSONDecodeError:
                    response_obj = stdout_text
                
                # Convert to OpenAI format
                openai_response = None
                try:
                    openai_response = format_to_openai_chat_completion(response_obj or stdout_text, model)
                except Exception as e:
                    logger.error(f"Failed to convert to OpenAI format: {e}")
                
                store_process_output(
                    str(process.pid),
                    stdout_text,
                    stderr_text,
                    cmd,
                    prompt,
                    response_obj or stdout_text,  # Original response
                    openai_response,  # Converted response
                    model
                )
            except Exception as e:
                logger.error(f"Error storing process output: {e}")
            
            # Now untrack the process
            untrack_claude_process(str(process.pid))
        
        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            logger.error(f"Claude command failed: {error_msg}")
            
            # Record completion with error
            await metrics.record_claude_completion(process_id, duration_ms, error=error_msg, conversation_id=conversation_id)
            
            raise Exception(f"Claude command failed: {error_msg}")
        
        output = stdout.decode()
        logger.debug(f"Raw Claude response: {output}")
        
        # Parse JSON response
        try:
            response = json.loads(output)
            logger.debug(f"Parsed Claude response: {response}")
            
            # Extract token count if available
            output_tokens = None
            if isinstance(response, dict) and "usage" in response:
                output_tokens = response["usage"].get("completion_tokens", None)
            
            # If we have a system response with a result, use that content
            if isinstance(response, dict) and "role" in response and response["role"] == "system" and "result" in response:
                # Use duration from response if available, otherwise use our calculated duration
                response_duration_ms = response.get("duration_ms", duration_ms)
                cost_usd = response.get("cost_usd", 0)
                result = response["result"]
                
                # Record completion metrics
                await metrics.record_claude_completion(process_id, response_duration_ms, output_tokens, conversation_id=conversation_id)
                
                # Try to parse as JSON if it looks like JSON
                try:
                    if isinstance(result, str) and result.strip().startswith('{') and result.strip().endswith('}'):
                        parsed_result = json.loads(result)
                        # Return a structured response with parsed content
                        return {
                            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                            "role": "assistant",
                            "parsed_json": True,
                            "content": parsed_result,
                            "raw_content": result,
                            "duration_ms": response_duration_ms,
                            "cost_usd": cost_usd
                        }
                except json.JSONDecodeError:
                    # Not valid JSON, continue with the original result
                    pass
                
                # Return a structured response with properly extracted content
                return {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                    "role": "assistant",
                    "parsed_json": False,
                    "content": result,
                    "duration_ms": response_duration_ms,
                    "cost_usd": cost_usd
                }
            
            # Record completion metrics using our calculated duration
            await metrics.record_claude_completion(process_id, duration_ms, output_tokens, conversation_id=conversation_id)
            
            # Return the response as-is if it doesn't match expected format
            return response
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response, returning raw output")
            
            # Record completion metrics
            await metrics.record_claude_completion(process_id, duration_ms, conversation_id=conversation_id)
            
            return output
            
    except Exception as e:
        # Record error in metrics
        duration_ms = (time.time() - start_time) * 1000
        await metrics.record_claude_completion(process_id, duration_ms, error=e, conversation_id=conversation_id)
        
        # Untrack the process if it's still tracked
        if process and process.pid:
            untrack_claude_process(str(process.pid))
            
        logger.error(f"Error running Claude command: {str(e)}")
        raise

async def stream_claude_output(prompt: str, conversation_id: str = None):
    """
    Run Claude with streaming JSON output and extract the content.
    Processes multiline JSON objects in the stream.
    """
    # Use the stream-json output format for true streaming
    base_cmd = f"{CLAUDE_CMD} -p"

    # Add conversation flag if there's a conversation ID
    if conversation_id:
        base_cmd += f" -c {conversation_id}"
        
        # Check if we're in test mode (only create temp dirs in production)
        is_test = 'unittest' in sys.modules or os.environ.get('TESTING') == '1'
        
        if not is_test:
            # Create or get a temporary directory for this conversation
            temp_dir = get_conversation_temp_dir(conversation_id)
            
            # Set the current working directory for this conversation
            os.environ["CLAUDE_CWD"] = temp_dir

    # Use stream-json format
    cmd = f"{base_cmd} --output-format stream-json"

    logger.info(f"Running command for streaming: {cmd}")
    
    # Generate a unique process ID for tracking
    process_id = f"claude-process-{uuid.uuid4().hex[:8]}"
    start_time = time.time()
    model = DEFAULT_MODEL
    
    # Record the Claude process start in metrics
    await metrics.record_claude_start(process_id, model, conversation_id)

    process = await asyncio.create_subprocess_shell(
        cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    # Track this process
    if process and process.pid:
        track_claude_process(str(process.pid), cmd)

    # Write the prompt to stdin
    try:
        # Write the prompt and ensure it's fully flushed to the process
        process.stdin.write(prompt.encode())
        await process.stdin.drain()  # Wait until the data is written to the underlying transport
        
        # Close stdin to signal we're done sending input
        # This is critical - Claude CLI won't start processing until stdin is closed
        process.stdin.close()
        
        logger.debug("Prompt written and stdin closed, Claude process should begin")
    except Exception as e:
        logger.error(f"Error writing to stdin: {str(e)}")
        
        # Record error in metrics
        duration_ms = (time.time() - start_time) * 1000
        await metrics.record_claude_completion(process_id, duration_ms, error=e, conversation_id=conversation_id)
        
        raise

    # Read the output
    try:
        # Buffer for collecting complete JSON objects
        buffer = ""
        output_tokens = 0
        streaming_complete = False
        last_chunk_time = time.time()
        
        # Read the output in chunks
        while True:
            try:
                # Use a timeout to detect stalled output
                chunk = await asyncio.wait_for(process.stdout.read(1024), timeout=2.0)
                if not chunk:
                    # End of stream reached
                    logger.debug("End of stream reached from Claude CLI")
                    break
                
                last_chunk_time = time.time()  # Update last chunk time
                buffer += chunk.decode('utf-8')
                
                # Try to find and parse complete JSON objects in the buffer
                # We're looking for matching braces to identify complete objects
                open_braces = 0
                start_pos = None
                
                for i, char in enumerate(buffer):
                    if char == '{':
                        if open_braces == 0:
                            start_pos = i
                        open_braces += 1
                    elif char == '}':
                        open_braces -= 1
                        if open_braces == 0 and start_pos is not None:
                            # We've found a complete JSON object
                            json_str = buffer[start_pos:i+1]
                            try:
                                json_obj = json.loads(json_str)
                                logger.debug(f"Parsed complete JSON object: {str(json_obj)[:200]}...")
                                
                                # Process the JSON object based on its structure
                                if "type" in json_obj and json_obj["type"] == "message":
                                    if "content" in json_obj and isinstance(json_obj["content"], list):
                                        for item in json_obj["content"]:
                                            if item.get("type") == "text" and "text" in item:
                                                content = item["text"]
                                                logger.info(f"Extracted text content: {content[:50]}...")
                                                output_tokens += len(content.split()) / 0.75  # Rough token count estimate
                                                yield {"content": content}
                                elif "stop_reason" in json_obj:
                                    # End of message - this is an explicit completion signal
                                    logger.info("Received explicit stop_reason from Claude")
                                    streaming_complete = True
                                    
                                    # Record completion in metrics
                                    duration_ms = (time.time() - start_time) * 1000
                                    await metrics.record_claude_completion(
                                        process_id, 
                                        duration_ms, 
                                        output_tokens=int(output_tokens),
                                        conversation_id=conversation_id
                                    )
                                    
                                    # Immediately send completion signal
                                    yield {"done": True}
                                    
                                    # Break out of the main loop after completion
                                    break
                                # Additional handling for system messages with cost info
                                elif "role" in json_obj and json_obj["role"] == "system":
                                    # This is a system message with additional info
                                    if "cost_usd" in json_obj or "duration_ms" in json_obj:
                                        # Extract execution duration if available
                                        duration_ms = json_obj.get("duration_ms", (time.time() - start_time) * 1000)
                                        streaming_complete = True
                                        
                                        # Record completion in metrics
                                        await metrics.record_claude_completion(
                                            process_id, 
                                            duration_ms, 
                                            output_tokens=int(output_tokens),
                                            conversation_id=conversation_id
                                        )
                                        
                                        # Immediately send completion signal
                                        logger.info("Received system message with completion info from Claude")
                                        yield {"done": True}
                                        
                                        # Break out of the main loop after completion
                                        break
                                
                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse JSON: {e}")
                            
                            # Remove the processed object from the buffer
                            buffer = buffer[i+1:]
                            # Reset to scan the buffer from the beginning
                            break
                            
                # Check if we've received a completion signal to break the main loop
                if streaming_complete:
                    break
                
            except asyncio.TimeoutError:
                # No data received within timeout - check if process is still running
                if process.returncode is not None:
                    logger.info(f"Claude process completed with return code {process.returncode}")
                    break
                
                # Check if we've gone too long without output
                if time.time() - last_chunk_time > 5.0:
                    logger.warning("No output from Claude for 5 seconds, checking if process is still active")
                    
                    # Try to get process status without killing it
                    try:
                        # On Unix systems we can check if the process exists without affecting it
                        if os.name != 'nt':  # Not Windows
                            try:
                                os.kill(process.pid, 0)  # This just checks if the process exists
                                logger.info(f"Claude process {process.pid} still exists, continuing to wait")
                            except ProcessLookupError:
                                logger.info(f"Claude process {process.pid} no longer exists")
                                break
                        
                        # If we're still waiting, check if it's been too long since the last chunk
                        if time.time() - last_chunk_time > 15.0:  # 15 seconds total waiting time
                            logger.warning("No output for 15 seconds, assuming Claude is finished")
                            break
                    except Exception as e:
                        logger.warning(f"Error checking process status: {e}")
                        # Continue waiting
        
        # If any data remains in the buffer, try to parse it
        if buffer and not streaming_complete:
            try:
                json_obj = json.loads(buffer)
                logger.debug(f"Parsed final JSON object from buffer: {str(json_obj)[:200]}...")
                
                if "type" in json_obj and json_obj["type"] == "message":
                    if "content" in json_obj and isinstance(json_obj["content"], list):
                        for item in json_obj["content"]:
                            if item.get("type") == "text" and "text" in item:
                                content = item["text"]
                                logger.info(f"Extracted final text content: {content[:50]}...")
                                output_tokens += len(content.split()) / 0.75  # Rough token count estimate
                                yield {"content": content}
                elif "stop_reason" in json_obj:
                    streaming_complete = True
                    yield {"done": True}
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse final buffer: {buffer[:200]}...")
        
        # Check for any errors
        stderr_data = await process.stderr.read()
        if stderr_data:
            stderr_str = stderr_data.decode('utf-8').strip()
            if stderr_str:
                logger.error(f"Error from Claude: {stderr_str}")
                
                # Record error in metrics
                duration_ms = (time.time() - start_time) * 1000
                await metrics.record_claude_completion(process_id, duration_ms, error=stderr_str, conversation_id=conversation_id)
                
                yield {"error": stderr_str}
                return
        
        # Ensure we've read all available output before completing
        # This is important to prevent any buffered data from being lost
        try:
            # Set a short timeout to get any remaining data without blocking too long
            remaining_output = await asyncio.wait_for(process.stdout.read(), timeout=0.5)
            if remaining_output:
                logger.info(f"Found additional {len(remaining_output)} bytes in stdout before completion")
                buffer = remaining_output.decode('utf-8')
                
                # Try to parse any complete JSON objects in the buffer
                try:
                    json_obj = json.loads(buffer)
                    logger.debug(f"Parsed final JSON object: {str(json_obj)[:200]}...")
                    
                    # Check for content or completion messages
                    if ("type" in json_obj and json_obj["type"] == "message" and 
                        "content" in json_obj and isinstance(json_obj["content"], list)):
                        for item in json_obj["content"]:
                            if item.get("type") == "text" and "text" in item:
                                content = item["text"]
                                logger.info(f"Extracted final buffered content: {content[:50]}...")
                                yield {"content": content}
                    elif "stop_reason" in json_obj:
                        streaming_complete = True
                        yield {"done": True}
                        return
                except json.JSONDecodeError:
                    # Not a complete JSON object, see if we can extract any text
                    logger.warning(f"Final buffer couldn't be parsed as JSON: {buffer[:100]}...")
                    # Just continue to completion 
        except asyncio.TimeoutError:
            logger.debug("No additional output available in stdout buffer")
        except Exception as e:
            logger.warning(f"Error reading final output buffer: {e}")
        
        # If we didn't see an explicit completion, record it now and send completion
        if not streaming_complete:
            logger.info("No explicit completion signal received, sending completion")
            duration_ms = (time.time() - start_time) * 1000
            await metrics.record_claude_completion(process_id, duration_ms, output_tokens=int(output_tokens), conversation_id=conversation_id)
            yield {"done": True}
                
    except Exception as e:
        logger.error(f"Error processing Claude output stream: {str(e)}", exc_info=True)
        
        # Record error in metrics
        duration_ms = (time.time() - start_time) * 1000
        await metrics.record_claude_completion(process_id, duration_ms, error=e, conversation_id=conversation_id)
        
        yield {"error": str(e)}
        
    finally:
        # Wait for process to complete normally if it hasn't already
        if process:
            if process.returncode is None:
                try:
                    # First try to wait for graceful completion (may already be done)
                    logger.debug("Waiting for Claude process to complete...")
                    await asyncio.wait_for(process.wait(), timeout=1.0)
                    logger.debug(f"Claude process completed with return code {process.returncode}")
                except asyncio.TimeoutError:
                    # If still running, try to terminate gracefully
                    logger.info("Claude process still running, sending SIGTERM")
                    try:
                        process.terminate()
                        await asyncio.wait_for(process.wait(), timeout=1.0)
                        logger.debug("Claude process terminated cleanly")
                    except (asyncio.TimeoutError, ProcessLookupError):
                        # If still doesn't exit, force kill it
                        logger.warning("Claude process didn't terminate, forcing kill")
                        try:
                            process.kill()
                            logger.debug("Claude process killed")
                        except ProcessLookupError:
                            logger.debug("Claude process already gone")
                            pass
            else:
                logger.debug(f"Claude process already completed with return code {process.returncode}")
                
            # Store the output and then untrack the process
            if process.pid:
                try:
                    # For streaming processes, collect output differently
                    # We'll collect the response from stdout and any errors from stderr
                    stdout_text = "Streaming process - output sent directly to client"
                    
                    # Try to read any stderr data
                    stderr_data = b""
                    try:
                        stderr_data = await asyncio.wait_for(process.stderr.read(), timeout=0.1)
                    except (asyncio.TimeoutError, AttributeError):
                        pass
                    
                    stderr_text = stderr_data.decode() if stderr_data else ""
                    
                    # Store what we have
                    store_process_output(
                        str(process.pid),
                        stdout_text,
                        stderr_text,
                        cmd,
                        prompt,
                        "Streaming response - output sent directly to client",
                        {
                            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": "Streaming response - content sent directly to client"
                                    },
                                    "finish_reason": "stop"
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                                "total_tokens": 0
                            },
                            "note": "This was a streaming response where content was sent directly to the client."
                        }
                    )
                except Exception as e:
                    logger.error(f"Error storing streaming process output: {e}")
                
                # Now untrack the process
                untrack_claude_process(str(process.pid))

# Message formatting functions

def format_messages_for_claude(request: ChatRequest) -> str:
    """Format messages from Ollama request into a prompt for Claude Code CLI."""
    prompt = ""

    # Add system message if present
    if request.system:
        prompt += f"System: {request.system}\n\n"

    # Add all messages in conversation
    for msg in request.messages:
        if msg.role == "user":
            prompt += f"Human: {msg.content}\n\n"
        elif msg.role == "assistant":
            prompt += f"Assistant: {msg.content}\n\n"
        elif msg.role == "system" and not request.system:
            prompt += f"System: {msg.content}\n\n"

    return prompt

def format_openai_to_claude(request: OpenAIChatRequest) -> str:
    """Format messages from OpenAI request into a prompt for Claude Code CLI."""
    prompt = ""

    # Process all messages in conversation
    for msg in request.messages:
        if msg.role == "user":
            prompt += f"Human: {msg.content}\n\n"
        elif msg.role == "assistant":
            prompt += f"Assistant: {msg.content}\n\n"
        elif msg.role == "system":
            prompt += f"System: {msg.content}\n\n"

    return prompt

def format_to_openai_chat_completion(claude_response, model, request_id=None):
    """
    Convert Claude CLI response to OpenAI-compatible Chat Completion format.
    
    OpenAI Chat Completion format:
    {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-3.5-turbo-0613",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "..."
            },
            "logprobs": null,
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 12,
            "total_tokens": 21
        }
    }
    """
    import time
    logger.debug(f"Converting Claude response to OpenAI format: {str(claude_response)[:200]}...")
    
    # Get current timestamp
    timestamp = int(time.time())
    
    # Extract content - handle differently based on response type
    content = ""
    duration_ms = 0
    prompt_tokens = 0
    completion_tokens = 0
    
    if isinstance(claude_response, dict):
        # Extract duration if available
        duration_ms = claude_response.get("duration_ms", 0)
        duration_api_ms = claude_response.get("duration_api_ms", duration_ms)
        
        # Extract token counts if available
        if "usage" in claude_response:
            prompt_tokens = claude_response["usage"].get("prompt_tokens", 0)
            completion_tokens = claude_response["usage"].get("completion_tokens", 0)
        
        # Check if it's our parsed response format
        if "role" in claude_response and claude_response["role"] == "assistant":
            # Check if we have parsed JSON
            if claude_response.get("parsed_json", False):
                # Use the pre-parsed content
                parsed_content = claude_response.get("content", {})
                # For OpenAI, serialize back to JSON string
                content = json.dumps(parsed_content)
            else:
                # Use the raw content
                content = claude_response.get("content", "")
        
        # Check for standard Claude system response with result (most common format)
        elif "role" in claude_response and claude_response["role"] == "system" and "result" in claude_response:
            # This is the Claude CLI JSON response format
            raw_result = claude_response["result"]
            
            # Handle the case where result is a string that looks like a Python dict
            # (which happens with tool_calls and other structured outputs)
            if isinstance(raw_result, str) and raw_result.startswith("{") and raw_result.endswith("}"):
                try:
                    # Try to parse it properly as JSON if it's a string that looks like JSON
                    if "'" in raw_result and not '"' in raw_result:
                        # Convert Python single quotes to JSON double quotes
                        quoted_result = raw_result.replace("'", '"')
                        parsed_result = json.loads(quoted_result)
                        # For structured content like tool_calls, we want to pass it through properly
                        # rather than as a string representation
                        
                        # Check if this is a tools response (OpenWebUI compatibility)
                        if "tools" in parsed_result:
                            logger.info("Detected tools array in response, modifying for OpenWebUI compatibility")
                            # Instead of returning the raw JSON with tools, return a text message
                            # that will make OpenWebUI continue the conversation
                            content = "I need to use a tool to answer your question. Please continue with the tool execution."
                        else:
                            content = parsed_result
                    else:
                        # Otherwise try to parse as-is
                        parsed_json = json.loads(raw_result)
                        
                        # Check if this is a tools response (OpenWebUI compatibility)
                        if isinstance(parsed_json, dict) and "tools" in parsed_json:
                            logger.info("Detected tools array in response, modifying for OpenWebUI compatibility")
                            # Instead of returning the raw JSON with tools, return a text message
                            # that will make OpenWebUI continue the conversation
                            content = "I need to use a tool to answer your question. Please continue with the tool execution."
                        else:
                            content = parsed_json
                except json.JSONDecodeError:
                    # If parsing fails, use the raw string
                    content = raw_result
                    logger.warning(f"Failed to parse result as JSON: {raw_result[:100]}...")
            else:
                content = raw_result
            
            # If cost_usd is available, log it
            if "cost_usd" in claude_response:
                logger.info(f"Claude request cost: ${claude_response['cost_usd']:.6f} USD")
                
        # Check for structured content array (sometimes used in Claude responses)
        elif "content" in claude_response and isinstance(claude_response["content"], list):
            for item in claude_response["content"]:
                if item.get("type") == "text" and "text" in item:
                    content += item["text"]
            
        # Fallback to result field
        elif "result" in claude_response:
            content = claude_response["result"]
            
    else:
        # Not a dict, use as string
        content = str(claude_response)
    
    # Use provided request_id or get from response or generate a new one
    message_id = request_id or (claude_response.get("id") if isinstance(claude_response, dict) else None)
    if not message_id:
        message_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    
    # If token counts aren't available, estimate them
    if not prompt_tokens:
        prompt_tokens = int((duration_ms / 10) if duration_ms else 100) 
    if not completion_tokens:
        completion_tokens = int((duration_ms / 20) if duration_ms else 50)
    
    # Format the response in OpenAI chat completion format
    openai_response = {
        "id": message_id,
        "object": "chat.completion",
        "created": timestamp,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content if isinstance(content, str) else json.dumps(content)
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }
    
    logger.debug(f"Converted to OpenAI format: {str(openai_response)[:200]}...")
    return openai_response

# Stream response functions for both API formats

async def stream_openai_response(claude_prompt: str, model_name: str, conversation_id: str = None, request_id: str = None):
    """
    Stream responses from Claude in OpenAI-compatible format.
    Uses the request_id from the client if provided.
    """
    import time
    
    # Use the request_id if provided, otherwise generate one
    message_id = request_id if request_id else f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    is_first_chunk = True
    full_response = ""
    completion_sent = False

    try:
        # Use Claude's streaming JSON output
        async for chunk in stream_claude_output(claude_prompt, conversation_id):
            logger.debug(f"Processing OpenAI stream chunk: {chunk}")
            
            # Check for errors
            if "error" in chunk:
                logger.error(f"Error in OpenAI stream: {chunk['error']}")
                error_response = {
                    "error": {
                        "message": f"Error: {chunk['error']}",
                        "type": "server_error",
                        "code": 500
                    }
                }
                # Send error response
                yield f"data: {json.dumps(error_response)}\n\n"
                # Immediately follow with DONE marker
                yield "data: [DONE]\n\n"
                completion_sent = True
                return
                
            # Extract content based on chunk format
            content = ""
            
            if "content" in chunk:
                content = chunk["content"]
                
                # Skip empty content
                if not content:
                    continue
                
                # Check if the content is a JSON string containing a tools array
                if isinstance(content, str):
                    try:
                        # Check if it's a JSON string starting and ending with braces
                        if content.strip().startswith('{') and content.strip().endswith('}'):
                            # Try to parse as JSON
                            parsed_content = json.loads(content)
                            
                            # Check if it contains a tools array
                            if isinstance(parsed_content, dict) and "tools" in parsed_content:
                                logger.info("Detected tools array in streaming response, modifying for OpenWebUI compatibility")
                                # Replace the JSON with a text message to make OpenWebUI continue the conversation
                                content = "I need to use a tool to answer your question. Please continue with the tool execution."
                    except json.JSONDecodeError:
                        # Not valid JSON, proceed with original content
                        pass
                
                # Accumulate full response for logging
                full_response += content
                
                # For the first real content, log it specially
                if content and is_first_chunk:
                    logger.info(f"First OpenAI content chunk received: {content[:50]}...")
                    
                # Format in OpenAI delta format
                response = {
                    "id": message_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                # Only include role in the first chunk
                                "role": "assistant" if is_first_chunk else "",
                                "content": content
                            },
                            "finish_reason": None
                        }
                    ]
                }
                
                if is_first_chunk:
                    is_first_chunk = False
                    
                # Send the SSE chunk
                sse_data = f"data: {json.dumps(response)}\n\n"
                logger.debug(f"Sending OpenAI SSE chunk: {sse_data[:100]}...")
                yield sse_data
                
            elif "done" in chunk and chunk["done"]:
                # This is the completion signal - send final chunk with finish_reason
                # CRITICAL: This is a completion marker from Claude
                logger.info("Received completion signal from Claude, sending final chunks")
                
                # Send the completion message with finish_reason
                final_response = {
                    "id": message_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},  # Empty delta for the final chunk
                            "finish_reason": "stop"
                        }
                    ]
                }
                yield f"data: {json.dumps(final_response)}\n\n"
                
                # Immediately follow with the DONE marker
                # This is critical - clients wait for this marker
                yield "data: [DONE]\n\n"
                completion_sent = True
                
                # Log complete response summary
                logger.info(f"Complete OpenAI response length: {len(full_response)} chars")
                if len(full_response) < 500:
                    logger.debug(f"Complete OpenAI response: {full_response}")
                
                return  # Exit the generator after sending completion
            else:
                # Log unrecognized format but don't interrupt the stream
                logger.warning(f"Unrecognized OpenAI chunk format: {chunk}")
                continue
        
        # If we reached here, the stream ended without a formal "done" marker
        # This is a safeguard - we should always properly terminate the stream
        if not completion_sent:
            logger.warning("Stream ended without completion signal, sending final markers")
            
            # Send the completion message with finish_reason
            final_response = {
                "id": message_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},  # Empty delta for the final chunk
                        "finish_reason": "stop"
                    }
                ]
            }
            yield f"data: {json.dumps(final_response)}\n\n"
            
            # Immediately follow with the DONE marker
            yield "data: [DONE]\n\n"
            
            # Log complete response summary
            logger.info(f"Complete OpenAI response length: {len(full_response)} chars")
            if len(full_response) < 500:
                logger.debug(f"Complete OpenAI response: {full_response}")
            
    except Exception as e:
        logger.error(f"Error streaming from Claude with OpenAI format: {str(e)}", exc_info=True)
        
        # Send error response
        error_response = {
            "error": {
                "message": f"Streaming error: {str(e)}",
                "type": "server_error",
                "code": 500
            }
        }
        yield f"data: {json.dumps(error_response)}\n\n"
        
        # Immediately follow with DONE marker (critical)
        yield "data: [DONE]\n\n"

# Unified function to parse request body
async def parse_request_body(request: Request):
    """Parse the request body and return the data as a dict."""
    try:
        # Try to parse based on content type
        if request.headers.get("content-type") == "application/json":
            # Parse JSON request
            data = await request.json()
        else:
            # Try to parse as form data
            form_data = await request.form()
            if form_data:
                # Convert form data to dict
                data = {}
                for k, v in form_data.items():
                    try:
                        # Try to parse as JSON if possible
                        data[k] = json.loads(v)
                    except:
                        data[k] = v
            else:
                # Fallback to trying to parse as JSON
                data = await request.json()
        
        return data
    except Exception as e:
        logger.error(f"Error parsing request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Error parsing request: {str(e)}")

# API endpoints - Chat completions endpoint

@app.post("/chat/completions")
async def chat(request_body: Request):
    """
    Chat completion API endpoint compatible with OpenAI's chat completions.
    """
    # Log the raw request
    logger.debug(f"Received chat completion request with content type: {request_body.headers.get('content-type', 'unknown')}")

    try:
        # Parse the request body based on content type
        if request_body.headers.get("content-type") == "application/json":
            # Parse JSON request
            data = await request_body.json()
        else:
            # Try to parse as form data
            form_data = await request_body.form()
            if form_data:
                # Convert form data to dict
                data = {}
                for k, v in form_data.items():
                    try:
                        # Try to parse as JSON if possible
                        data[k] = json.loads(v)
                    except:
                        data[k] = v
            else:
                # Fallback to trying to parse as JSON
                data = await request_body.json()
        
        logger.debug(f"Parsed request data: {data}")
        
        # Convert to ChatRequest model
        request = ChatRequest(**data)
    except Exception as e:
        logger.error(f"Error parsing chat request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Error parsing request: {str(e)}")
        
    # Log the incoming request
    logger.info(f"Received chat request for model: {request.model}")

    # Get or create conversation ID
    conversation_id = None
    
    # First check if the client supplied a conversation_id in the request JSON
    if hasattr(request, 'conversation_id') and request.conversation_id:
        conversation_id = request.conversation_id
        logger.info(f"Using client-provided conversation_id: {conversation_id}")
    # Otherwise use the request ID for tracking conversations
    elif request.id:
        # Check if we have this conversation in our cache
        conversation_id = get_conversation_id(request.id)
        if conversation_id is None and len(request.messages) > 0:
            # This is a new conversation with an ID
            # Generate a unique conversation ID for Claude (use the request ID itself)
            conversation_id = request.id
            set_conversation_id(request.id, conversation_id)
            logger.info(f"Created new conversation with ID: {conversation_id}")
    # As a fallback, always ensure we have a conversation ID for metrics tracking
    if conversation_id is None:
        conversation_id = f"conv-{uuid.uuid4().hex[:8]}"
        logger.info(f"Generated fallback conversation ID: {conversation_id}")

    # Format the messages for Claude Code CLI
    claude_prompt = format_messages_for_claude(request)

    # Handle streaming vs. non-streaming responses
    if request.stream:
        # Use standard FastAPI StreamingResponse with our generator
        logger.info("Using standard StreamingResponse for streaming OpenAI compatibility")
        return StreamingResponse(
            stream_openai_response(claude_prompt, request.model, conversation_id, request.id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "X-Accel-Buffering": "no"  # Prevents buffering in Nginx, which helps with streaming
            }
        )
    else:
        try:
            # Start timing the non-streaming request
            start_time = time.time()
            request_id = f"req-{uuid.uuid4().hex[:8]}"
            
            logger.info(f"Processing non-streaming request {request_id} for model: {request.model}")
            
            # For non-streaming, get the full response at once
            claude_response = await run_claude_command(claude_prompt, conversation_id=conversation_id)
            
            # Log the raw Claude response at debug level
            if isinstance(claude_response, dict):
                logger.debug(f"Raw Claude response: {json.dumps(claude_response)[:500]}...")
            else:
                logger.debug(f"Raw Claude response: {str(claude_response)[:500]}...")
                
            # Log information about the client
            user_agent = request_body.headers.get("user-agent", "Unknown")
            referer = request_body.headers.get("referer", "Unknown")
            origin = request_body.headers.get("origin", "Unknown")
            host = request_body.headers.get("host", "Unknown")
            
            # Detect OpenWebUI specifically
            is_openwebui = (
                "OpenWebUI" in user_agent or 
                "openwebui" in str(referer).lower() or 
                "openwebui" in str(origin).lower()
            )
            
            logger.info(f"Client info for {request_id}: UA={user_agent}, Referer={referer}, Origin={origin}, Host={host}")
            if is_openwebui:
                logger.info(f"Detected OpenWebUI client for request {request_id}")
            
            # Format as OpenAI chat completion
            openai_response = format_to_openai_chat_completion(claude_response, request.model, request.id)
            
            # Log the formatted response
            logger.info(f"Formatted OpenAI response for request {request_id}")
            logger.debug(f"OpenAI-format response: {json.dumps(openai_response)[:500]}...")
            
            # Create minimal set of headers that are compatible with all clients
            headers = {
                # Essential CORS headers
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
                
                # Basic cache control
                "Cache-Control": "no-cache",
                
                # Add a unique request ID for tracking
                "X-Request-ID": request_id
            }
            
            # Log if we detected OpenWebUI
            if is_openwebui:
                logger.info(f"Detected OpenWebUI client for request {request_id}")
            
            # Create the JSONResponse with the headers we just built
            response = JSONResponse(
                content=openai_response,
                media_type="application/json",
                headers=headers
            )
            
            # Log the response being returned
            duration_ms = int((time.time() - start_time) * 1000)
            logger.info(f"Returning non-streaming response for {request_id} (duration: {duration_ms}ms)")
            
            # Log conversation tracking info
            if conversation_id:
                logger.info(f"Tracked in conversation: {conversation_id}")
                logger.info(f"Active conversations: {len(metrics.active_conversations)}")
                logger.info(f"Unique conversations: {len(metrics.unique_conversations)}")
            
            # No artificial delay - let FastAPI/Starlette handle the request normally
            # This avoids potential issues with event loop and response handling
            
            # Log detailed timing information
            logger.info(f"Request {request_id} timing: total={duration_ms}ms")
            logger.info(f"Sending response for {request_id} to client: approx {len(json.dumps(openai_response))} bytes")
            
            # Return the response directly
            return response
        except Exception as e:
            # Enhanced error handling with better client response
            error_id = f"err-{uuid.uuid4().hex[:8]}"
            logger.error(f"Error processing non-streaming request {error_id}: {str(e)}", exc_info=True)
            logger.error(traceback.format_exc())
            
            # Create a proper OpenAI-compatible error response
            error_response = {
                "error": {
                    "message": f"Error processing request: {str(e)}",
                    "type": "server_error",
                    "param": None,
                    "code": "internal_error",
                    "error_id": error_id
                }
            }
            
            # Use minimal error headers for better compatibility
            error_headers = {
                # Essential CORS headers
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
                
                # Basic tracking and content info
                "X-Request-ID": error_id,
                "Content-Type": "application/json"
            }
            
            # Detect OpenWebUI for logging purposes
            try:
                user_agent = request_body.headers.get("user-agent", "Unknown")
                origin = request_body.headers.get("origin", "Unknown")
                
                is_openwebui = (
                    "OpenWebUI" in user_agent or 
                    "openwebui" in str(request_body.headers.get("referer", "")).lower() or 
                    "openwebui" in str(origin).lower()
                )
                
                if is_openwebui:
                    logger.info(f"Detected OpenWebUI client for error response {error_id}")
            except Exception as header_err:
                logger.warning(f"Could not check for OpenWebUI in error response: {header_err}")
            
            # Return error with proper headers
            return JSONResponse(
                status_code=500,
                content=error_response,
                headers=error_headers
            )

@app.post("/v1/chat/completions")
async def openai_chat_completions(request_body: Request):
    """
    OpenAI-compatible chat completions endpoint for v1 API.
    Simply forwards to our main chat completion endpoint.
    """
    return await chat(request_body)

@app.get("/models")
async def list_models():
    """List available models (OpenAI-compatible models endpoint)."""
    import time
    current_time = int(time.time())

    # Static list of Claude models in OpenAI format
    claude_models = [
        {
            "id": "anthropic/claude-3.7-sonnet",
            "object": "model",
            "created": current_time,
            "owned_by": "anthropic"
        },
    ]

    return {"object": "list", "data": claude_models}

@app.get("/v1/models")
async def openai_list_models():
    """List available models in OpenAI format (v1 API)."""
    # Reuse the same implementation as the main models endpoint
    return await list_models()

# Add explicit OPTIONS handlers for critical endpoints to ensure CORS is working
@app.options("/chat/completions")
@app.options("/v1/chat/completions")
@app.options("/openwebui_test")
@app.options("/test_openwebui")
async def options_chat():
    """Handle OPTIONS requests for API endpoints."""
    return JSONResponse(
        content={"status": "ok"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",  # Allow all headers for maximum compatibility
            "Access-Control-Max-Age": "86400",
        }
    )

@app.get("/version")
async def get_version():
    """Get API version info (OpenAI-compatible version endpoint)."""
    return {
        "version": "0.1.0",
        "build": "claude-ollama-server"
    }

@app.post("/test_openwebui")
async def test_openwebui(request: Request):
    """Test endpoint for OpenWebUI compatibility, especially for tool/function calling."""
    body = await request.json()
    response_type = body.get("response_type", "tool")  # tool, text, or error
    
    logger.info(f"Testing tool response with type: {response_type}")
    
    if response_type == "tool":
        # Simulate a tool response
        tool_response = {
            "id": f"test-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "anthropic/claude-3.7-sonnet",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "tools": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": json.dumps({
                                            "location": "San Francisco, CA",
                                            "unit": "celsius"
                                        })
                                    }
                                }
                            ]
                        })
                    },
                    "finish_reason": "tool_calls"
                }
            ],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 20,
                "total_tokens": 70
            }
        }
        
        # Our enhanced handler should detect the tools array and replace it with text
        response = format_to_openai_chat_completion(
            {"role": "system", "result": json.dumps(tool_response["choices"][0]["message"]["content"])},
            "anthropic/claude-3.7-sonnet"
        )
        
        return JSONResponse(content=response)
    elif response_type == "text":
        # Return a simple text response
        return JSONResponse(content={
            "id": f"test-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "anthropic/claude-3.7-sonnet",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "This is a test text response."
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 20,
                "total_tokens": 70
            }
        })
    else:
        # Simulate an error response
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "This is a test error response.",
                    "type": "test_error",
                    "code": "test_error_code"
                }
            }
        )

@app.post("/openwebui_test")
async def openwebui_test(request: Request):
    """Special endpoint for diagnosing OpenWebUI compatibility issues."""
    body = await request.json()
    test_message = body.get("message", "This is a test response for OpenWebUI")
    
    # Log information about the client
    user_agent = request.headers.get("user-agent", "Unknown")
    referer = request.headers.get("referer", "Unknown")
    origin = request.headers.get("origin", "Unknown")
    host = request.headers.get("host", "Unknown")
    
    # Detect if request is from OpenWebUI
    is_openwebui = (
        "OpenWebUI" in user_agent or 
        "openwebui" in str(referer).lower() or 
        "openwebui" in str(origin).lower()
    )
    
    logger.info(f"OpenWebUI test request from: UA={user_agent}, Referer={referer}, Origin={origin}, Host={host}")
    if is_openwebui:
        logger.info("Confirmed request is from OpenWebUI")
    
    # Create a simple test response
    test_response = {
        "id": f"test-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "anthropic/claude-3.7-sonnet",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": test_message
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }
    
    # Minimal headers for maximum compatibility
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Content-Type": "application/json"
    }
    
    logger.info(f"Sending test response: {json.dumps(test_response)[:100]}...")
    return JSONResponse(content=test_response, headers=headers)

# Dictionary to track proxy-launched Claude processes
proxy_launched_processes = {}
# Dictionary to store outputs from recent processes (limited to last 20)
process_outputs = collections.OrderedDict()
MAX_STORED_OUTPUTS = 20

# Self-test function for basic health checks
def run_self_test():
    """Run quick self-tests to verify functionality."""
    print("Running Claude Ollama API self-tests...")
    
    # Test 1: Check metrics initialization
    print("Test 1: Checking metrics initialization...")
    try:
        # Just check if metrics exists and has basic attributes
        if not hasattr(metrics, 'total_invocations'):
            print("FAIL: metrics does not have total_invocations attribute")
            return False
        if not hasattr(metrics, 'active_conversations'):
            print("FAIL: metrics does not have active_conversations attribute")
            return False
        print("PASS: Metrics initialized correctly")
    except Exception as e:
        print(f"FAIL: Error checking metrics: {e}")
        return False
    
    # Test 2: Check dashboard HTML generation
    print("Test 2: Checking dashboard HTML generation...")
    try:
        html = generate_dashboard_html()
        if not html or not isinstance(html, str):
            print("FAIL: generate_dashboard_html() did not return a string")
            return False
        if not "<!DOCTYPE html>" in html:
            print("FAIL: Dashboard HTML does not contain DOCTYPE declaration")
            return False
        if not "Claude Usage" in html:
            print("FAIL: Dashboard HTML does not contain 'Claude Usage' section")
            return False
        print("PASS: Dashboard HTML generated correctly")
    except Exception as e:
        print(f"FAIL: Error generating dashboard HTML: {e}")
        return False
    
    # Test 3: Check conversation ID tracking
    print("Test 3: Checking conversation ID tracking...")
    try:
        test_req_id = "test-request-id"
        test_conv_id = "test-conv-id"
        
        # Store a conversation ID
        set_conversation_id(test_req_id, test_conv_id)
        
        # Retrieve it
        retrieved_id = get_conversation_id(test_req_id)
        
        if retrieved_id != test_conv_id:
            print(f"FAIL: Retrieved conversation ID '{retrieved_id}' does not match stored ID '{test_conv_id}'")
            return False
        
        print("PASS: Conversation ID tracking works correctly")
    except Exception as e:
        print(f"FAIL: Error testing conversation ID tracking: {e}")
        return False
    
    print("All self-tests passed successfully!")
    return True

def track_claude_process(pid, command):
    """Track a Claude process launched by this proxy server"""
    import time
    proxy_launched_processes[pid] = {
        "pid": pid,
        "command": command,
        "start_time": time.time(),
        "status": "running"
    }
    logger.info(f"Tracking new Claude process with PID {pid}")

def untrack_claude_process(pid):
    """Remove a Claude process from tracking"""
    if pid in proxy_launched_processes:
        proxy_launched_processes.pop(pid, None)
        logger.info(f"Untracked Claude process with PID {pid}")

def store_process_output(pid, stdout, stderr, command, prompt, response, converted_response=None, model=DEFAULT_MODEL):
    """Store the output from a Claude process"""
    global process_outputs
    timestamp = datetime.datetime.now().isoformat()
    
    # Create a new entry for this process
    entry = {
        "pid": pid,
        "timestamp": timestamp,
        "command": command,
        "prompt": prompt,
        "stdout": stdout,
        "stderr": stderr,
        "response": response
    }
    
    # If we have a converted response, add it
    if converted_response:
        entry["converted_response"] = converted_response
    # Otherwise, try to convert it now
    elif response and isinstance(response, (dict, str)):
        try:
            converted = format_to_openai_chat_completion(response, model)
            entry["converted_response"] = converted
        except Exception as e:
            logger.error(f"Failed to convert response for storage: {e}")
    
    # Store the entry
    process_outputs[pid] = entry
    
    # Limit the number of stored outputs
    while len(process_outputs) > MAX_STORED_OUTPUTS:
        process_outputs.popitem(last=False)  # Remove oldest item (FIFO)
    
    logger.info(f"Stored output for process {pid}")

def get_process_output(pid):
    """Get the stored output for a process"""
    return process_outputs.get(pid, None)

def get_running_claude_processes():
    """Get information about currently running Claude processes that were launched by this proxy"""
    try:
        import subprocess
        import re
        import time
        import psutil
        
        # List of processes to return
        active_processes = []
        
        # Check each tracked process to see if it's still running
        pids_to_remove = []
        for pid, process_info in proxy_launched_processes.items():
            try:
                # Check if process still exists
                process = psutil.Process(int(pid))
                
                # Get process info
                with process.oneshot():
                    user = process.username()
                    cpu = f"{process.cpu_percent(interval=0.1):.1f}"
                    mem = f"{process.memory_percent():.1f}"
                    started = time.strftime("%H:%M:%S", time.localtime(process.create_time()))
                    command = " ".join(process.cmdline())[:80] + ("..." if len(" ".join(process.cmdline())) > 80 else "")
                    
                    # Calculate runtime
                    runtime_secs = int(time.time() - process_info['start_time'])
                    hours, remainder = divmod(runtime_secs, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    runtime = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                    
                    active_processes.append({
                        "user": user,
                        "pid": pid,
                        "cpu": cpu,
                        "memory": mem,
                        "started": started,
                        "runtime": runtime,
                        "command": command
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Process no longer exists or can't be accessed
                pids_to_remove.append(pid)
                
        # Clean up processes that no longer exist
        for pid in pids_to_remove:
            untrack_claude_process(pid)
        
        return active_processes
    except Exception as e:
        logger.error(f"Error getting Claude processes: {e}")
        return []

def generate_dashboard_html():
    """Generate HTML for the dashboard page"""
    # Get current metrics
    metrics_data = metrics.get_metrics()
    
    # Get currently running Claude processes
    running_processes = get_running_claude_processes()
    
    # Format metrics for display - convert from ms to minutes and seconds
    def format_time_ms(time_ms):
        if not time_ms:
            return "N/A"
        total_seconds = time_ms / 1000
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60
        return f"{minutes}m {seconds:.1f}s"
    
    avg_execution = format_time_ms(metrics_data['performance']['avg_execution_time_ms'])
    median_execution = format_time_ms(metrics_data['performance']['median_execution_time_ms'])
    
    # Memory metrics formatting
    avg_memory = f"{metrics_data['resources']['avg_memory_mb']:.2f}" if metrics_data['resources']['avg_memory_mb'] else "N/A"
    peak_memory = f"{metrics_data['resources']['peak_memory_mb']:.2f}" if metrics_data['resources']['peak_memory_mb'] else "N/A"
    avg_cpu = f"{metrics_data['resources']['avg_cpu_percent']:.2f}%" if metrics_data['resources']['avg_cpu_percent'] else "N/A"
    
    # Create the HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Claude Ollama API Dashboard</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f7;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1, h2, h3 {{
            color: #000;
        }}
        h1 {{
            margin-bottom: 10px;
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        .metric {{
            margin-bottom: 15px;
        }}
        .metric-name {{
            font-weight: bold;
            margin-bottom: 5px;
            color: #555;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: 500;
        }}
        .metric-unit {{
            font-size: 14px;
            color: #777;
        }}
        .error-card {{
            background-color: #fff8f8;
            border-left: 4px solid #e53935;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: 600;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .reload {{
            display: inline-block;
            background-color: #0071e3;
            color: white;
            padding: 8px 16px;
            border-radius: 6px;
            text-decoration: none;
            margin-top: 10px;
            transition: background-color 0.2s;
        }}
        .reload:hover {{
            background-color: #0077ed;
        }}
        .button {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            text-decoration: none;
            font-size: 13px;
            cursor: pointer;
            border: none;
        }}
        .alert {{
            background-color: #f44336;
            color: white;
        }}
        .alert:hover {{
            background-color: #d32f2f;
        }}
        .timestamp {{
            color: #777;
            font-size: 14px;
            margin-bottom: 20px;
        }}
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 500;
            margin-left: 8px;
        }}
        .badge-success {{
            background-color: #e3f2fd;
            color: #0277bd;
        }}
        .badge-primary {{
            background-color: #e8f5e9;
            color: #2e7d32;
        }}
        .auto-refresh {{
            font-size: 14px;
            color: #555;
            margin-top: 5px;
        }}
        .refresh-toggle {{
            margin-left: 10px;
            color: #0071e3;
            text-decoration: none;
            font-size: 12px;
            padding: 2px 6px;
            border-radius: 4px;
            background-color: #f0f7ff;
        }}
        .refresh-toggle:hover {{
            background-color: #e1f0ff;
        }}
        .chart-container {{
            height: 200px;
            margin-top: 15px;
        }}
        .highlight {{
            background-color: #f3f9ff;
            border-left: 4px solid #0077ed;
        }}
        /* Modal styles */
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            overflow: auto;
            background-color: rgba(0,0,0,0.4);
        }}
        .modal-content {{
            background-color: #fefefe;
            margin: 5% auto;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
            width: 80%;
            max-width: 1000px;
            max-height: 80vh;
            overflow-y: auto;
            position: relative;
        }}
        .modal-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
            margin-bottom: 15px;
        }}
        .modal-header h3 {{
            margin: 0;
        }}
        .modal-footer {{
            margin-top: 15px;
            padding-top: 10px;
            border-top: 1px solid #eee;
            text-align: right;
        }}
        .close-button {{
            color: #aaa;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
        }}
        .close-button:hover {{
            color: black;
        }}
        pre {{
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
            white-space: pre-wrap;
            max-height: 300px;
            overflow-y: auto;
        }}
        pre.error {{
            background-color: #fff0f0;
            border-left: 3px solid #ff6b6b;
        }}
        .output-table {{
            width: 100%;
        }}
    </style>
    <script>
        // Store scroll position before refresh
        function saveScrollPosition() {{
            sessionStorage.setItem('scrollPosition', window.scrollY);
        }}
        
        // Restore scroll position after refresh
        function restoreScrollPosition() {{
            const scrollPosition = sessionStorage.getItem('scrollPosition');
            if (scrollPosition) {{
                window.scrollTo(0, parseInt(scrollPosition));
            }}
        }}
        
        // Variables to control auto-refresh
        let refreshTimeout;
        let refreshEnabled = true;
        const REFRESH_INTERVAL = 10000;  // 10 seconds
        
        // Function to schedule next refresh
        function scheduleRefresh() {{
            // Clear any existing timeout
            if (refreshTimeout) {{
                clearTimeout(refreshTimeout);
            }}
            
            // Only schedule if refresh is enabled
            if (refreshEnabled) {{
                refreshTimeout = setTimeout(function() {{
                    // Only reload if no modal is open
                    if (document.querySelector('.modal[style*="display: block"]') === null) {{
                        saveScrollPosition();
                        window.location.reload();
                    }} else {{
                        // If modal is open, reschedule the refresh
                        scheduleRefresh();
                    }}
                }}, REFRESH_INTERVAL);
            }}
        }}
        
        // Function to toggle refresh on/off
        function toggleRefresh() {{
            refreshEnabled = !refreshEnabled;
            
            // Update status text
            const statusElement = document.getElementById('refresh-status');
            if (statusElement) {{
                statusElement.textContent = refreshEnabled ? 'Auto-refresh' : 'Refresh paused';
                statusElement.style.color = refreshEnabled ? '' : '#ff6b6b';
            }}
            
            // If turning refresh back on, schedule next refresh
            if (refreshEnabled) {{
                scheduleRefresh();
            }} else if (refreshTimeout) {{
                clearTimeout(refreshTimeout);
            }}
        }}
        
        // Start the refresh cycle
        scheduleRefresh();
        
        // Countdown timer for refresh and restore scroll position
        window.onload = function() {{
            // Restore scroll position on page load
            restoreScrollPosition();
            
            // Setup countdown timer
            let timeLeft = 10;
            const timerElement = document.getElementById('refresh-timer');
            const statusElement = document.getElementById('refresh-status');
            
            setInterval(function() {{
                // Only countdown if refresh is enabled
                if (refreshEnabled) {{
                    timeLeft -= 1;
                    if (timeLeft >= 0) {{
                        timerElement.textContent = timeLeft;
                    }} else {{
                        timeLeft = 10;
                    }}
                }} else {{
                    // If refresh is paused, keep showing status
                    statusElement.textContent = 'Refresh paused';
                    statusElement.style.color = '#ff6b6b';
                }}
            }}, 1000);
            
            // Make manual refresh button preserve scroll position
            document.querySelectorAll('.reload').forEach(function(button) {{
                button.addEventListener('click', function(e) {{
                    e.preventDefault();
                    saveScrollPosition();
                    window.location.reload();
                }});
            }});
        }};
    </script>
</head>
<body>
    <div class="container">
        <h1>Claude API Dashboard</h1>
        <div class="timestamp">
            Last updated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            <span class="auto-refresh">
                (<span id="refresh-status">Auto-refresh</span> in <span id="refresh-timer">10</span>s)
                <a href="javascript:void(0)" onclick="toggleRefresh()" class="refresh-toggle">Pause/Resume</a>
            </span>
        </div>

        <div class="stats-grid">
            <div class="card">
                <h2>Server Status</h2>
                <div class="metric">
                    <div class="metric-name">Uptime</div>
                    <div class="metric-value">{metrics_data['uptime']['formatted']}</div>
                </div>
                <div class="metric">
                    <div class="metric-name">Server Started</div>
                    <div class="metric-value" style="font-size: 16px;">{metrics_data['uptime']['start_time'].replace('T', ' ').split('.')[0]}</div>
                </div>
            </div>

            <div class="card highlight">
                <h2>Claude Invocations</h2>
                <div class="metric">
                    <div class="metric-name">Total Claude Processes</div>
                    <div class="metric-value">{metrics_data['claude_invocations']['total']}</div>
                </div>
                <div class="metric">
                    <div class="metric-name">Currently Running</div>
                    <div class="metric-value">{metrics_data['claude_invocations']['current_running']}</div>
                </div>
                <div class="metric">
                    <div class="metric-name">Max Concurrent</div>
                    <div class="metric-value">{metrics_data['claude_invocations']['max_concurrent']}</div>
                </div>
            </div>

            <div class="card">
                <h2>Claude Usage</h2>
                <div class="metric">
                    <div class="metric-name">Invocations Per Minute</div>
                    <div class="metric-value">{metrics_data['claude_invocations']['per_minute']:.2f}</div>
                </div>
                <div class="metric">
                    <div class="metric-name">Invocations Per Hour</div>
                    <div class="metric-value">{metrics_data['claude_invocations']['per_hour']:.2f}</div>
                </div>
                <div class="metric">
                    <div class="metric-name">Active Conversations</div>
                    <div class="metric-value">{len(metrics.active_conversations)}</div>
                </div>
                <div class="metric">
                    <div class="metric-name">Total Conversations</div>
                    <div class="metric-value">{len(metrics.unique_conversations)}</div>
                </div>
                <div class="metric">
                    <div class="metric-name">Total Output Tokens</div>
                    <div class="metric-value">{metrics_data['tokens']['total_completion']}</div>
                </div>
            </div>

            <div class="card highlight">
                <h2>Performance</h2>
                <div class="metric">
                    <div class="metric-name">Average Execution Time</div>
                    <div class="metric-value">{avg_execution}</div>
                </div>
                <div class="metric">
                    <div class="metric-name">Median Execution Time</div>
                    <div class="metric-value">{median_execution}</div>
                </div>
            </div>

            <div class="card">
                <h2>System Resources</h2>
                <div class="metric">
                    <div class="metric-name">Average Memory</div>
                    <div class="metric-value">{avg_memory} <span class="metric-unit">MB</span></div>
                </div>
                <div class="metric">
                    <div class="metric-name">Peak Memory</div>
                    <div class="metric-value">{peak_memory} <span class="metric-unit">MB</span></div>
                </div>
                <div class="metric">
                    <div class="metric-name">Average CPU</div>
                    <div class="metric-value">{avg_cpu}</div>
                </div>
            </div>

            <div class="card">
                <h2>Conversations</h2>
                <div class="metric">
                    <div class="metric-name">Unique Conversations</div>
                    <div class="metric-value">{metrics_data['conversations']['unique_count']}</div>
                </div>
                <div class="metric">
                    <div class="metric-name">Active Conversations</div>
                    <div class="metric-value">{metrics_data['conversations']['active_count']}</div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Timeline</h2>
            <table>
                <tr>
                    <th>Event</th>
                    <th>Timestamp</th>
                </tr>
                <tr>
                    <td>First Claude Invocation</td>
                    <td>{metrics_data['timestamps']['first_invocation'].replace('T', ' ').split('.')[0] if metrics_data['timestamps']['first_invocation'] else 'N/A'}</td>
                </tr>
                <tr>
                    <td>Last Claude Invocation</td>
                    <td>{metrics_data['timestamps']['last_invocation'].replace('T', ' ').split('.')[0] if metrics_data['timestamps']['last_invocation'] else 'N/A'}</td>
                </tr>
                <tr>
                    <td>Last Claude Completion</td>
                    <td>{metrics_data['timestamps']['last_completion'].replace('T', ' ').split('.')[0] if metrics_data['timestamps']['last_completion'] else 'N/A'}</td>
                </tr>
            </table>
        </div>

        <div class="card">
            <h2>Invocations by Model</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Count</th>
                </tr>
                {''.join([f"<tr><td>{model}</td><td>{count}</td></tr>" for model, count in metrics_data['distribution']['by_model'].items()])}
            </table>
        </div>

        <div class="card">
            <h2>Invocations by Conversation</h2>
            <table>
                <tr>
                    <th>Conversation ID</th>
                    <th>Count</th>
                </tr>
                {''.join([f"<tr><td>{conv_id}</td><td>{count}</td></tr>" for conv_id, count in list(metrics_data['distribution']['by_conversation'].items())[:25]])}
                {f"<tr><td colspan='2'>...and {len(metrics_data['distribution']['by_conversation']) - 25} more conversations</td></tr>" if len(metrics_data['distribution']['by_conversation']) > 25 else ""}
            </table>
        </div>

        <div id="processes" class="card highlight">
            <h2>Proxy-Launched Claude Processes</h2>
            <p>This section shows Claude processes launched by this proxy server. <a href="#process-outputs" class="button">View Recent Process Outputs</a></p>
            <table>
                <tr>
                    <th>PID</th>
                    <th>User</th>
                    <th>CPU %</th>
                    <th>Memory %</th>
                    <th>Started</th>
                    <th>Runtime</th>
                    <th>Command</th>
                    <th>Actions</th>
                </tr>
"""
    # Add process rows dynamically
    process_rows = ""
    if running_processes:
        for process in running_processes:
            process_rows += f"""                <tr>
                    <td>{process['pid']}</td>
                    <td>{process['user']}</td>
                    <td>{process['cpu']}%</td>
                    <td>{process['memory']}%</td>
                    <td>{process['started']}</td>
                    <td>{process['runtime']}</td>
                    <td><code>{process['command']}</code></td>
                    <td>
                        <a href="javascript:void(0)" onclick="if(confirm('Are you sure you want to terminate process {process['pid']}?')) fetch('/terminate_process/{process['pid']}', {{method: 'POST'}}).then(() => window.location.reload());" class="button alert">Terminate</a>
                    </td>
                </tr>"""
    else:
        process_rows = "<tr><td colspan='8'>No proxy-launched Claude processes currently running</td></tr>"
    
    html += process_rows
    
    html += f"""            </table>
            <p><em>Last updated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</em></p>
            <p><a href="#processes" class="reload">Refresh</a></p>
        </div>"""

    # Add error card if there are errors
    if metrics_data['errors']['count'] > 0:
        error_rows = ""
        for error in metrics_data['errors']['recent']:
            error_time = error['time'].replace('T', ' ').split('.')[0]
            error_msg = error['error']
            error_rows += f"<tr><td>{error_time}</td><td>{error_msg}</td></tr>"
            
        html += f"""
        <div class="card error-card">
            <h2>Recent Errors</h2>
            <div class="metric">
                <div class="metric-name">Total Errors</div>
                <div class="metric-value">{metrics_data['errors']['count']}</div>
            </div>
            <table>
                <tr>
                    <th>Time</th>
                    <th>Error</th>
                </tr>
                {error_rows}
            </table>
        </div>
        """
        
    # Add section for process outputs
    html += """
        <div id="process-outputs" class="card">
            <h2>Recent Process Outputs</h2>
            <p>This section shows outputs from recently completed Claude processes (up to 20):</p>
            <div id="process-output-list">Loading process outputs...</div>
            
            <script>
                // Fetch process outputs on page load
                async function fetchProcessOutputs() {
                    try {
                        const response = await fetch('/process_outputs');
                        const data = await response.json();
                        displayProcessOutputs(data.outputs);
                    } catch (error) {
                        console.error('Error fetching process outputs:', error);
                        document.getElementById('process-output-list').textContent = 'Error loading process outputs';
                    }
                }
                
                // Display process outputs in a table
                function displayProcessOutputs(outputs) {
                    const container = document.getElementById('process-output-list');
                    
                    if (!outputs || outputs.length === 0) {
                        container.innerHTML = '<p>No process outputs available yet.</p>';
                        return;
                    }
                    
                    let html = `
                    <table class="output-table">
                        <tr>
                            <th>PID</th>
                            <th>Timestamp</th>
                            <th>Command</th>
                            <th>Actions</th>
                        </tr>
                    `;
                    
                    outputs.forEach(output => {
                        // Format timestamp
                        const timestamp = new Date(output.timestamp).toLocaleString();
                        
                        html += `
                        <tr>
                            <td>${output.pid}</td>
                            <td>${timestamp}</td>
                            <td><code>${output.command}</code></td>
                            <td>
                                <button class="button" onclick="viewProcessOutput('${output.pid}')">View Details</button>
                            </td>
                        </tr>
                        `;
                    });
                    
                    html += '</table>';
                    container.innerHTML = html;
                }
                
                // Fetch and display a specific process output
                async function viewProcessOutput(pid) {
                    try {
                        // Pause auto-refresh while viewing output
                        refreshEnabled = false;
                        
                        const response = await fetch(`/process_output/${pid}`);
                        const data = await response.json();
                        
                        // Create modal window
                        const modal = document.createElement('div');
                        modal.classList.add('modal');
                        modal.setAttribute('data-pid', pid);
                        
                        // Format the timestamp
                        const timestamp = new Date(data.timestamp).toLocaleString();
                        
                        // Format the content
                        const promptHtml = `<pre>${escapeHtml(data.prompt)}</pre>`;
                        const stdoutHtml = `<pre>${escapeHtml(data.stdout)}</pre>`;
                        const stderrHtml = data.stderr ? `<pre class="error">${escapeHtml(data.stderr)}</pre>` : '<p>No errors</p>';
                        
                        // Format the response (original and converted)
                        let responseHtml = '';
                        
                        // Check if the response is an object or string
                        if (typeof data.response === 'object') {
                            responseHtml = `<pre>${escapeHtml(JSON.stringify(data.response, null, 2))}</pre>`;
                        } else {
                            responseHtml = `<pre>${escapeHtml(data.response)}</pre>`;
                        }
                        
                        // Format the converted response if available
                        let convertedHtml = '';
                        if (data.converted_response) {
                            convertedHtml = `
                                <h4>Converted OpenAI Response (Sent to OpenWebUI)</h4>
                                <pre>${escapeHtml(JSON.stringify(data.converted_response, null, 2))}</pre>
                            `;
                        }
                        
                        modal.innerHTML = `
                            <div class="modal-content">
                                <div class="modal-header">
                                    <h3>Process Output: PID ${data.pid}</h3>
                                    <span class="close-button" onclick="closeModal(this)">&times;</span>
                                </div>
                                <p><strong>Timestamp:</strong> ${timestamp}</p>
                                <p><strong>Command:</strong> <code>${escapeHtml(data.command)}</code></p>
                                
                                <h4>Prompt</h4>
                                ${promptHtml}
                                
                                <h4>Standard Output</h4>
                                ${stdoutHtml}
                                
                                <h4>Standard Error</h4>
                                ${stderrHtml}
                                
                                <h4>Original Claude Response</h4>
                                ${responseHtml}
                                
                                ${convertedHtml}
                                
                                <div class="modal-footer">
                                    <button class="button" onclick="closeModal(this.closest('.modal-content').querySelector('.close-button'))">Close</button>
                                </div>
                            </div>
                        `;
                        
                        document.body.appendChild(modal);
                        modal.style.display = 'block';
                        
                        // Add event listener for the escape key
                        document.addEventListener('keydown', handleEscapeKey);
                        
                    } catch (error) {
                        console.error('Error fetching process output:', error);
                        alert('Error loading process output for PID ' + pid);
                        
                        // Re-enable auto-refresh if there was an error
                        refreshEnabled = true;
                        scheduleRefresh();
                    }
                }
                
                // Handle escape key for closing modal
                function handleEscapeKey(e) {
                    if (e.key === 'Escape') {
                        const modal = document.querySelector('.modal[style*="display: block"]');
                        if (modal) {
                            closeModal(modal.querySelector('.close-button'));
                        }
                    }
                }
                
                // Helper function to escape HTML special characters
                function escapeHtml(text) {
                    if (!text) return '';
                    return text
                        .replace(/&/g, "&amp;")
                        .replace(/</g, "&lt;")
                        .replace(/>/g, "&gt;")
                        .replace(/"/g, "&quot;")
                        .replace(/'/g, "&#039;");
                }
                
                // Close the modal when the close button is clicked
                function closeModal(button) {
                    const modal = button.closest('.modal');
                    modal.style.display = 'none';
                    modal.remove();
                    
                    // Remove escape key listener
                    document.removeEventListener('keydown', handleEscapeKey);
                    
                    // Re-enable auto-refresh
                    refreshEnabled = true;
                    scheduleRefresh();
                }
                
                // Fetch outputs on page load
                document.addEventListener('DOMContentLoaded', fetchProcessOutputs);
            </script>
        </div>
    """
    
    # Close the outer containers
    html += """
    </div>
</body>
</html>
"""
    return html

@app.get("/", response_class=HTMLResponse)
async def root():
    """Dashboard endpoint that displays API metrics."""
    return generate_dashboard_html()

@app.get("/status", response_class=PlainTextResponse)
async def status():
    """Simple status endpoint that mimics Ollama's root response."""
    return "Ollama is running"

@app.get("/metrics", response_class=JSONResponse)
async def get_metrics():
    """JSON endpoint for metrics data"""
    return metrics.get_metrics()

@app.get("/process_outputs")
async def list_process_outputs():
    """List all stored process outputs"""
    # Convert OrderedDict to list for JSON serialization, newest first
    outputs_list = []
    for pid, output in reversed(process_outputs.items()):
        # Create a summary without the full output text
        summary = {
            "pid": output["pid"],
            "timestamp": output["timestamp"],
            "command": output["command"][:100] + ("..." if len(output["command"]) > 100 else ""),
            "prompt_preview": output["prompt"][:100] + ("..." if len(output["prompt"]) > 100 else ""),
            "has_stdout": bool(output["stdout"]),
            "has_stderr": bool(output["stderr"]),
            "has_response": bool(output["response"])
        }
        outputs_list.append(summary)
    
    return {"outputs": outputs_list}

@app.get("/process_output/{pid}")
async def get_single_process_output(pid: str):
    """Get the output for a specific process"""
    output = get_process_output(pid)
    if output:
        return output
    else:
        raise HTTPException(status_code=404, detail=f"No output found for process {pid}")

@app.post("/terminate_process/{pid}")
async def terminate_process(pid: str):
    """
    Terminate a specific process by PID.
    This is used for the process management UI.
    """
    try:
        pid = int(pid)
        import os
        import signal
        import psutil
        
        # Safety check - only terminate Claude-related processes
        process = psutil.Process(pid)
        if "claude" in process.name().lower() or any("claude" in arg.lower() for arg in process.cmdline()):
            # First try SIGTERM for clean shutdown
            logger.info(f"Attempting to terminate Claude process {pid} with SIGTERM")
            process.terminate()
            
            # Wait up to 3 seconds for process to terminate
            try:
                process.wait(timeout=3)
                return {"status": "success", "message": f"Process {pid} terminated successfully"}
            except psutil.TimeoutExpired:
                # If process doesn't terminate, use SIGKILL
                logger.warning(f"Process {pid} did not terminate with SIGTERM, using SIGKILL")
                process.kill()
                return {"status": "success", "message": f"Process {pid} forcibly terminated with SIGKILL"}
        else:
            logger.warning(f"Attempted to terminate non-Claude process {pid}, request denied")
            return JSONResponse(
                status_code=403,
                content={"status": "error", "message": "Can only terminate Claude-related processes"}
            )
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Invalid PID format"}
        )
    except psutil.NoSuchProcess:
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": f"Process {pid} not found"}
        )
    except Exception as e:
        logger.error(f"Error terminating process {pid}: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Error terminating process: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    import sys
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Claude Ollama API Server")
    parser.add_argument("--test", action="store_true", help="Run self-tests and exit")
    parser.add_argument("--host", type=str, help="Host to bind to (default: from env var or 0.0.0.0)")
    parser.add_argument("--port", type=int, help="Port to run on (default: from env var or 22434)")
    parser.add_argument("port_pos", nargs="?", type=int, help="Port (legacy positional argument)")
    
    args = parser.parse_args()
    
    # Run self-tests if requested
    if args.test:
        print("Running self-tests...")
        success = run_self_test()
        sys.exit(0 if success else 1)
    
    # Setup server config
    host = args.host or os.environ.get("HOST", "0.0.0.0")
    
    # Determine port (positional arg takes precedence, then --port, then env var, then default)
    if args.port_pos is not None:
        port = args.port_pos
    elif args.port is not None:
        port = args.port
    else:
        port = int(os.environ.get("PORT", 22434))  # Custom port to avoid conflicts with Ollama

    print(f"Starting Claude API Server on http://{host}:{port}")
    print("Ensure Claude Code is installed and configured.")
    print("This server will use your existing Claude Code CLI installation.")

    # Start server
    uvicorn.run(app, host=host, port=port)