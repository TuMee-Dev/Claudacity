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

DEFAULT_MODEL = "claude-3-7-sonnet-latest"  # Default Claude model to report
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
            self.unique_conversations.add(conversation_id)
            self.active_conversations.add(conversation_id)
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
    
    async def record_claude_completion(self, process_id, duration_ms, output_tokens=None, memory_mb=None, error=None):
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
    
    def prune_old_data(self):
        """Remove old data from time-based metrics"""
        now = datetime.datetime.now()
        
        # Remove old conversation data (older than 1 hour)
        one_hour_ago = now - datetime.timedelta(hours=1)
        
        # Prune old active conversations
        old_conversation_ids = set()
        for conversation_id in self.active_conversations:
            # In a real implementation, we would have per-conversation timestamp tracking
            old_conversation_ids.add(conversation_id)
        
        for conversation_id in old_conversation_ids:
            self.active_conversations.discard(conversation_id)
    
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

# Custom direct streaming response class that doesn't buffer

class DirectStreamResponse(Response):
    """
    Custom response class that writes directly to the connection without buffering.
    Enables immediate flushing of SSE responses to the client.
    """
    def __init__(self, generate_content, media_type="text/event-stream"):
        super().__init__(background=BackgroundTask(self._cleanup), media_type=media_type)
        self.generate_content = generate_content
        self.started = False
        self.conn_closed = False
        
    async def _cleanup(self):
        """Clean up resources when the connection closes"""
        self.conn_closed = True
        
    async def stream_response(self, scope, receive, send):
        """Stream the response to the client directly"""
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [
                [b"content-type", b"text/event-stream"],
                [b"cache-control", b"no-cache"],
                [b"connection", b"keep-alive"],
                [b"transfer-encoding", b"chunked"],
            ],
        })
        
        self.started = True
        
        try:
            # Get generator for content
            async for chunk in self.generate_content:
                if self.conn_closed:
                    logger.info("Connection closed by client, stopping stream")
                    break
                    
                # Send chunk immediately
                await send({
                    "type": "http.response.body",
                    "body": chunk.encode('utf-8'),
                    "more_body": True,
                })
                # Short pause to allow client processing
                await asyncio.sleep(0.001)
                
            # Send final empty chunk to close the stream
            await send({
                "type": "http.response.body",
                "body": b"",
                "more_body": False,
            })
            
        except Exception as e:
            logger.error(f"Error streaming response: {str(e)}", exc_info=True)
            if not self.started:
                await send({
                    "type": "http.response.start",
                    "status": 500,
                    "headers": [
                        [b"content-type", b"text/plain"],
                    ],
                })
            
            error_msg = f"Error streaming response: {str(e)}"
            await send({
                "type": "http.response.body",
                "body": error_msg.encode("utf-8"),
                "more_body": False,
            })
            
    def __call__(self, scope, receive, send):
        """Process the ASGI request"""
        asyncio.create_task(self.stream_response(scope, receive, send))
        return self

# Initialize FastAPI app

app = FastAPI(
    title="Claude Ollama API",
    description="OpenAI-compatible API server for Claude Code",
    version="0.1.0",
)

# Add CORS middleware to allow requests from any origin

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
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
    
    # Log request details
    logger.info(f"Request {request_id}: {method} {url}")

    try:
        # Process the request
        response = await call_next(request)
        
        # Calculate request duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Log response code
        logger.info(f"Response {request_id}: {response.status_code} ({int(duration_ms)}ms)")
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
    id: Optional[str] = None  # Conversation ID

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
    id: Optional[str] = None  # Conversation ID
    user: Optional[str] = None  # OpenAI field, can also be used for conversation ID if id is not provided

# Utility functions for working with Claude Code CLI

async def run_claude_command(prompt: str, conversation_id: str = None) -> str:
    """Run a Claude Code command and return the output."""
    # Base command
    base_cmd = f"{CLAUDE_CMD} -p"

    # Add conversation flag if there's a conversation ID
    if conversation_id:
        base_cmd += f" -c {conversation_id}"

    # Use regular JSON output format
    cmd = f"{base_cmd} --output-format json"

    logger.info(f"Running command: {cmd}")
    
    # Generate a unique process ID for tracking
    process_id = f"claude-process-{uuid.uuid4().hex[:8]}"
    start_time = time.time()
    model = DEFAULT_MODEL
    
    # Record the Claude process start in metrics
    await metrics.record_claude_start(process_id, model, conversation_id)

    try:
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Send the prompt to Claude
        stdout, stderr = await process.communicate(input=prompt.encode())
        
        # Calculate execution duration
        duration_ms = (time.time() - start_time) * 1000
        
        if process.returncode != 0:
            error_msg = stderr.decode()
            logger.error(f"Claude command failed: {error_msg}")
            
            # Record completion with error
            await metrics.record_claude_completion(process_id, duration_ms, error=error_msg)
            
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
                await metrics.record_claude_completion(process_id, response_duration_ms, output_tokens)
                
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
            await metrics.record_claude_completion(process_id, duration_ms, output_tokens)
            
            # Return the response as-is if it doesn't match expected format
            return response
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response, returning raw output")
            
            # Record completion metrics
            await metrics.record_claude_completion(process_id, duration_ms)
            
            return output
            
    except Exception as e:
        # Record error in metrics
        await metrics.record_claude_completion(process_id, duration_ms, error=e)
        
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
        await metrics.record_claude_completion(process_id, duration_ms, error=e)
        
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
                                        output_tokens=int(output_tokens)
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
                                            output_tokens=int(output_tokens)
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
                await metrics.record_claude_completion(process_id, duration_ms, error=stderr_str)
                
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
            await metrics.record_claude_completion(process_id, duration_ms, output_tokens=int(output_tokens))
            yield {"done": True}
                
    except Exception as e:
        logger.error(f"Error processing Claude output stream: {str(e)}", exc_info=True)
        
        # Record error in metrics
        duration_ms = (time.time() - start_time) * 1000
        await metrics.record_claude_completion(process_id, duration_ms, error=e)
        
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
    
    # Get current timestamp
    timestamp = int(time.time())
    
    # Extract content - handle differently based on response type
    content = ""
    duration_ms = 0
    
    if isinstance(claude_response, dict):
        # Check if it's our parsed response format
        if "role" in claude_response and claude_response["role"] == "assistant":
            duration_ms = claude_response.get("duration_ms", 0)
            
            # Check if we have parsed JSON
            if claude_response.get("parsed_json", False):
                # Use the pre-parsed content
                parsed_content = claude_response.get("content", {})
                # For OpenAI, serialize back to JSON string
                content = json.dumps(parsed_content)
            else:
                # Use the raw content
                content = claude_response.get("content", "")
                
        # Check for structured content array
        elif "content" in claude_response and isinstance(claude_response["content"], list):
            for item in claude_response["content"]:
                if item.get("type") == "text" and "text" in item:
                    content += item["text"]
            duration_ms = claude_response.get("duration_ms", 0)
            
        # Check for system response with result
        elif "role" in claude_response and claude_response["role"] == "system" and "result" in claude_response:
            content = claude_response["result"]
            duration_ms = claude_response.get("duration_ms", 0)
            
        # Fallback to result field
        elif "result" in claude_response:
            content = claude_response["result"]
            duration_ms = claude_response.get("duration_ms", 0)
    else:
        # Not a dict, use as string
        content = str(claude_response)
    
    # Use provided request_id or get from response or generate a new one
    message_id = request_id or (claude_response.get("id") if isinstance(claude_response, dict) else None)
    if not message_id:
        message_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    
    # Calculate token counts (rough estimation)
    prompt_tokens = int((duration_ms / 10) if duration_ms else 100)
    completion_tokens = int((duration_ms / 20) if duration_ms else 50)
    
    # Format the response in OpenAI chat completion format
    return {
        "id": message_id,
        "object": "chat.completion",
        "created": timestamp,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
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
    if request.id:
        # Check if we have this conversation in our cache
        conversation_id = get_conversation_id(request.id)
        if conversation_id is None and len(request.messages) > 0:
            # This is a new conversation with an ID
            # Generate a unique conversation ID for Claude (use the request ID itself)
            conversation_id = request.id
            set_conversation_id(request.id, conversation_id)

    # Format the messages for Claude Code CLI
    claude_prompt = format_messages_for_claude(request)

    # Handle streaming vs. non-streaming responses
    if request.stream:
        # We'll use our direct streaming response to avoid buffering
        async def direct_openai_stream():
            """Direct streaming generator with immediate flushing"""
            # Use the request_id if provided, otherwise generate one
            message_id = request.id if request.id else f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())
            is_first_chunk = True
            full_response = ""
            completion_sent = False
            
            try:
                # Set up process to run Claude CLI
                # Use stream-json format for true streaming
                base_cmd = f"{CLAUDE_CMD} -p"
                if conversation_id:
                    base_cmd += f" -c {conversation_id}"
                cmd = f"{base_cmd} --output-format stream-json"
                
                logger.info(f"Running direct stream command: {cmd}")
                
                # Generate a unique process ID for tracking
                process_id = f"claude-process-{uuid.uuid4().hex[:8]}"
                start_time = time.time()
                model = request.model
                
                # Record the Claude process start in metrics
                await metrics.record_claude_start(process_id, model, conversation_id)
                
                # Start the Claude process
                process = await asyncio.create_subprocess_shell(
                    cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                # Write the prompt to stdin and close it immediately
                try:
                    process.stdin.write(claude_prompt.encode())
                    await process.stdin.drain()
                    process.stdin.close()
                    logger.debug("Prompt sent to Claude process")
                except Exception as e:
                    error_msg = f"Error writing to Claude CLI: {str(e)}"
                    logger.error(error_msg)
                    
                    # Send error as SSE
                    error_json = json.dumps({
                        "error": {"message": error_msg, "type": "server_error", "code": 500}
                    })
                    yield f"data: {error_json}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                
                # Process Claude output directly
                buffer = ""
                output_tokens = 0
                
                # Read output until EOF
                while True:
                    # Read a chunk with a short timeout
                    try:
                        chunk = await asyncio.wait_for(process.stdout.read(4096), timeout=0.1)
                        if not chunk:  # EOF
                            break
                        
                        buffer += chunk.decode('utf-8')
                        
                        # Find complete JSON objects
                        while True:
                            # Look for a complete JSON object by checking matched braces
                            # This is a simple implementation that works for clean JSON
                            open_idx = buffer.find("{")
                            if open_idx == -1:
                                break
                                
                            # Find the matching closing brace by counting
                            level = 0
                            close_idx = -1
                            
                            for i in range(open_idx, len(buffer)):
                                if buffer[i] == "{":
                                    level += 1
                                elif buffer[i] == "}":
                                    level -= 1
                                    if level == 0:
                                        close_idx = i + 1
                                        break
                            
                            if close_idx == -1:  # No complete object found
                                break
                                
                            # Extract the JSON object
                            try:
                                json_obj = json.loads(buffer[open_idx:close_idx])
                                # Remove processed part from buffer
                                buffer = buffer[close_idx:]
                                
                                # Process the JSON object based on its type
                                if "type" in json_obj and json_obj["type"] == "message":
                                    # Content chunk
                                    if "content" in json_obj and isinstance(json_obj["content"], list):
                                        for item in json_obj["content"]:
                                            if item.get("type") == "text" and "text" in item:
                                                content = item["text"]
                                                full_response += content
                                                output_tokens += len(content.split()) / 0.75
                                                
                                                # Format as OpenAI compatible response
                                                sse_obj = {
                                                    "id": message_id,
                                                    "object": "chat.completion.chunk",
                                                    "created": created,
                                                    "model": model,
                                                    "choices": [{
                                                        "index": 0,
                                                        "delta": {
                                                            # Only include role in first chunk
                                                            "role": "assistant" if is_first_chunk else "",
                                                            "content": content
                                                        },
                                                        "finish_reason": None
                                                    }]
                                                }
                                                
                                                if is_first_chunk:
                                                    is_first_chunk = False
                                                
                                                # Directly yield the SSE formatted data
                                                sse_chunk = f"data: {json.dumps(sse_obj)}\n\n"
                                                yield sse_chunk
                                                
                                elif "stop_reason" in json_obj:
                                    # End of message - send completion
                                    if not completion_sent:
                                        logger.info("Sending direct completion signal")
                                        
                                        # Record metrics
                                        duration_ms = (time.time() - start_time) * 1000
                                        await metrics.record_claude_completion(
                                            process_id, 
                                            duration_ms,
                                            output_tokens=int(output_tokens)
                                        )
                                        
                                        # Send finish reason
                                        completion_obj = {
                                            "id": message_id,
                                            "object": "chat.completion.chunk",
                                            "created": created,
                                            "model": model,
                                            "choices": [{
                                                "index": 0,
                                                "delta": {},
                                                "finish_reason": "stop"
                                            }]
                                        }
                                        
                                        # Send completion - critical!
                                        yield f"data: {json.dumps(completion_obj)}\n\n"
                                        yield "data: [DONE]\n\n"
                                        completion_sent = True
                                        break
                                        
                                elif ("role" in json_obj and json_obj["role"] == "system" and 
                                      ("cost_usd" in json_obj or "duration_ms" in json_obj)):
                                    # System info - indicates completion
                                    if not completion_sent:
                                        # Record completion metrics
                                        duration_ms = json_obj.get("duration_ms", (time.time() - start_time) * 1000)
                                        await metrics.record_claude_completion(
                                            process_id, 
                                            duration_ms,
                                            output_tokens=int(output_tokens)
                                        )
                                        
                                        # Send completion signal
                                        logger.info("Sending system completion signal")
                                        
                                        # Send finish reason
                                        completion_obj = {
                                            "id": message_id,
                                            "object": "chat.completion.chunk",
                                            "created": created,
                                            "model": model,
                                            "choices": [{
                                                "index": 0,
                                                "delta": {},
                                                "finish_reason": "stop"
                                            }]
                                        }
                                        
                                        # Send completion - critical!
                                        yield f"data: {json.dumps(completion_obj)}\n\n"
                                        yield "data: [DONE]\n\n"
                                        completion_sent = True
                                        break
                                
                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse JSON chunk: {e}")
                                # Skip this malformed object and try to find next valid one
                                buffer = buffer[1:]  # Skip first char and retry
                            except Exception as e:
                                logger.error(f"Error processing chunk: {e}")
                                buffer = buffer[close_idx:]  # Skip to next chunk
                    
                    except asyncio.TimeoutError:
                        # No data right now, check if process is still alive
                        if process.returncode is not None:
                            # Process has exited
                            logger.info(f"Claude process exited with code {process.returncode}")
                            break
                
                # Final error check
                if process.returncode is not None and process.returncode != 0:
                    # Process failed
                    stderr_data = await process.stderr.read()
                    error_msg = stderr_data.decode('utf-8').strip() if stderr_data else f"Process failed with code {process.returncode}"
                    logger.error(f"Claude CLI error: {error_msg}")
                    
                    if not completion_sent:
                        error_obj = {"error": {"message": error_msg, "type": "server_error", "code": 500}}
                        yield f"data: {json.dumps(error_obj)}\n\n"
                        yield "data: [DONE]\n\n"
                        completion_sent = True
                
                # Ensure we always send completion
                if not completion_sent:
                    logger.info("Sending final completion signal")
                    
                    # Record completion 
                    duration_ms = (time.time() - start_time) * 1000
                    await metrics.record_claude_completion(
                        process_id, 
                        duration_ms,
                        output_tokens=int(output_tokens) if output_tokens > 0 else 0
                    )
                    
                    # Send finish reason
                    completion_obj = {
                        "id": message_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }]
                    }
                    
                    # Send completion - critical!
                    yield f"data: {json.dumps(completion_obj)}\n\n"
                    yield "data: [DONE]\n\n"
                
            except Exception as e:
                logger.error(f"Error in direct streaming: {str(e)}", exc_info=True)
                
                if not completion_sent:
                    error_obj = {"error": {"message": str(e), "type": "server_error", "code": 500}}
                    yield f"data: {json.dumps(error_obj)}\n\n"
                    yield "data: [DONE]\n\n"
        
        # Return our direct streaming response with no buffering
        return DirectStreamResponse(direct_openai_stream())
    else:
        try:
            # For non-streaming, get the full response at once
            claude_response = await run_claude_command(claude_prompt, conversation_id=conversation_id)
            logger.debug(f"Full claude_response from run_claude_command: {claude_response}")
            
            # Format as OpenAI chat completion
            openai_response = format_to_openai_chat_completion(claude_response, request.model, request.id)
            
            logger.debug(f"Final OpenAI-format response: {json.dumps(openai_response)}")
            
            # Return explicitly as JSONResponse with application/json content type
            return JSONResponse(
                content=openai_response,
                media_type="application/json",
                headers={"Content-Type": "application/json"}
            )
        except Exception as e:
            logger.error(f"Error calling Claude: {str(e)}", exc_info=True)
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error calling Claude: {str(e)}")

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

@app.get("/version")
async def get_version():
    """Get API version info (OpenAI-compatible version endpoint)."""
    return {
        "version": "0.1.0",
        "build": "claude-ollama-server"
    }

def generate_dashboard_html():
    """Generate HTML for the dashboard page"""
    # Get current metrics
    metrics_data = metrics.get_metrics()
    
    # Format metrics for display
    avg_execution = f"{metrics_data['performance']['avg_execution_time_ms']:.2f}" if metrics_data['performance']['avg_execution_time_ms'] else "N/A"
    median_execution = f"{metrics_data['performance']['median_execution_time_ms']:.2f}" if metrics_data['performance']['median_execution_time_ms'] else "N/A"
    
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
        .chart-container {{
            height: 200px;
            margin-top: 15px;
        }}
        .highlight {{
            background-color: #f3f9ff;
            border-left: 4px solid #0077ed;
        }}
    </style>
    <script>
        // Auto-refresh the page every 10 seconds
        setTimeout(function() {{
            window.location.reload();
        }}, 10000);
        
        // Countdown timer for refresh
        window.onload = function() {{
            let timeLeft = 10;
            const timerElement = document.getElementById('refresh-timer');
            
            setInterval(function() {{
                timeLeft -= 1;
                if (timeLeft >= 0) {{
                    timerElement.textContent = timeLeft;
                }}
            }}, 1000);
        }};
    </script>
</head>
<body>
    <div class="container">
        <h1>Claude API Dashboard</h1>
        <div class="timestamp">
            Last updated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            <span class="auto-refresh">(Auto-refreshes in <span id="refresh-timer">10</span>s)</span>
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
                    <div class="metric-name">Total Output Tokens</div>
                    <div class="metric-value">{metrics_data['tokens']['total_completion']}</div>
                </div>
            </div>

            <div class="card highlight">
                <h2>Performance</h2>
                <div class="metric">
                    <div class="metric-name">Average Execution Time</div>
                    <div class="metric-value">{avg_execution} <span class="metric-unit">ms</span></div>
                </div>
                <div class="metric">
                    <div class="metric-name">Median Execution Time</div>
                    <div class="metric-value">{median_execution} <span class="metric-unit">ms</span></div>
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

        {'''
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
                {''.join([f"<tr><td>{error['time'].replace('T', ' ').split('.')[0]}</td><td>{error['error']}</td></tr>" for error in metrics_data['errors']['recent']])}
            </table>
        </div>
        ''' if metrics_data['errors']['count'] > 0 else ''}
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

if __name__ == "__main__":
    import uvicorn
    import sys

    # Setup server config
    host = os.environ.get("HOST", "0.0.0.0")  # Listen on all network interfaces by default

    # Check if port is provided as command-line argument
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port: {sys.argv[1]}")
            sys.exit(1)
    else:
        port = int(os.environ.get("PORT", 22434))  # Custom port to avoid conflicts with Ollama

    print(f"Starting Claude API Server on http://{host}:{port}")
    print("Ensure Claude Code is installed and configured.")
    print("This server will use your existing Claude Code CLI installation.")

    # Start server
    uvicorn.run(app, host=host, port=port)