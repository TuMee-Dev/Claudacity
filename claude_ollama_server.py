"""
Claude-compatible Ollama API Server.

This FastAPI application provides an OpenAI-compatible API server that interfaces with
a locally running Claude Code process. It implements both Ollama API and OpenAI API format.
"""

import argparse
import asyncio
import datetime
import json
import logging
import os
import pprint
import psutil # type: ignore
import shlex
import shutil
import sys
import tempfile
import time
import timezone
import traceback
import uuid
import uvicorn # type: ignore
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any, Union, Deque
from pydantic import BaseModel, Field # type: ignore
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks # type: ignore
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse, HTMLResponse, Response # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from starlette.background import BackgroundTask # type: ignore
from claude_metrics import ClaudeMetrics
import dashboard
import formatters
import process_tracking

# Configuration

# Debug mode - set to False in production
DEBUG = False

# Process and streaming response timeouts
CLAUDE_STREAM_CHUNK_TIMEOUT = 18.0  # Seconds without output before checking process status (was 10, originally 5)
CLAUDE_STREAM_MAX_SILENCE = 180.0  # Maximum seconds to wait with no output before assuming process hung (was 60, originally 15)

# Version information
API_VERSION = "0.1.0"
BUILD_NAME = "claude-ollama-server"
GIT_SHA = ""  # Could be populated dynamically if needed
BUILD_DATE = ""  # Could be populated dynamically if needed

DEFAULT_MODEL = "anthropic/claude-3.7-sonnet"  # Default Claude model to report (must match /models endpoint)
DEFAULT_MAX_TOKENS = 1000000  # 1m context length
CONVERSATION_CACHE_TTL = 3600 * 3  # 3 hours in seconds

# Define available models (currently just one, but in a list for future expansion)
AVAILABLE_MODELS = [
    {
        "name": "claude-3.7-sonnet",  # Model tag
        "modified_at": "2025-05-05T19:53:25.564072",  # Use actual timestamp if needed
        "size": 0,  # Set to 0 if not applicable
        "digest": "anthropic_claude_3_7_sonnet_20250505",  # Unique identifier for the model
        "details": {
            "model": "claude-3.7-sonnet",  # Repeats the tag
            "parent_model": "",  # Match Ollama format
            "format": "api",  # Using "api" instead of "gguf" since this is an API model
            "family": "anthropic",
            "families": ["anthropic", "claude"],  # Array of family names
            "parameter_size": "13B",
            "quantization_level": "none"
        }
    }
]

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Set up file handler for logging to logs directory
log_file_path = os.path.join("logs", "server.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S'))

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,  # Use DEBUG flag to control log verbosity
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Get logger and add file handler explicitly
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)

# Metrics tracking class

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

# Background task for cleanup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run background tasks when the server starts."""
    # Start background cleanup task
    periodic_cleanup_task = asyncio.create_task(periodic_cleanup())
    yield
    # Stop background cleanup task
    #&? Impliment


# Initialize FastAPI app

app = FastAPI(
    title="Claude Ollama API",
    description="OpenAI-compatible API server for Claude Code",
    version=API_VERSION,
    lifespan=lifespan
)


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
    logger.debug(f"Request {request_id}: {method} {url}")
    
    # Enhanced logging for chat completions
    if is_chat_completion:
        try:
            # Log headers for debugging
            headers = dict(request.headers.items())
            user_agent = headers.get("user-agent", "Unknown")
            origin = headers.get("origin", "Unknown")
            
            logger.debug(f"Chat completion request from: UA={user_agent}, Origin={origin}")
            
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
                    logger.debug(f"Chat completion details: model={model}, stream={stream}")
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
        logger.debug(f"Response {request_id}: {response.status_code} ({int(duration_ms)}ms)")
        
        # Enhanced logging for chat completions
        if is_chat_completion:
            # Log headers for debugging
            headers = dict(response.headers.items())
            content_type = headers.get("content-type", "Unknown")
            logger.debug(f"Chat completion response: Content-Type={content_type}")
        
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
    return datetime.now(timezone.utc).isoformat("T", "milliseconds") + "Z"

# Command to interact with Claude Code CLI


CLAUDE_CMD = process_tracking.find_claude_command()

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
    tools: Optional[List[Dict[str, Any]]] = None  # Add tools field for function/tool calling

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
    ollama_client: Optional[bool] = False  # Flag for Ollama client

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

async def run_claude_command(prompt: str, conversation_id: str = None, original_request=None, timeout: float = CLAUDE_STREAM_MAX_SILENCE) -> str:
    """Run a Claude Code command and return the output."""
    # Log the request details for debugging
    logger.debug(f"run_claude_command called with:")
    logger.debug(f"  - prompt length: {len(prompt)}")
    logger.debug(f"  - conversation_id: {conversation_id}")
    logger.debug(f"  - timeout: {timeout}s")
    
    # Start building the base command
    base_cmd = f"{CLAUDE_CMD}"

    # Log if tools are present for debugging purposes only
    if original_request and isinstance(original_request, dict) and original_request.get('tools'):
        logger.debug(f"[TOOLS] Detected tools in original_request for conversation: {conversation_id}")
        logger.debug(f"[TOOLS] Tools are detected but will NOT be passed to Claude directly")
    
    # Always use conversation ID if provided (regardless of tools or message count)
    if conversation_id:
        logger.debug(f"[TOOLS] Using conversation ID: {conversation_id}")
        base_cmd += f" -c {conversation_id}"
        
        # Check if we're in test mode (only create temp dirs in production)
        is_test = 'unittest' in sys.modules or os.environ.get('TESTING') == '1'
        
        if not is_test:
            # Create or get a temporary directory for this conversation
            temp_dir = get_conversation_temp_dir(conversation_id)
            
            # Set the current working directory for this conversation
            # We'll use environment variable to pass it to the Claude CLI
            os.environ["CLAUDE_CWD"] = temp_dir
    else:
        logger.debug(f"[TOOLS] No conversation ID provided")

    # Add the -p flag AFTER the conversation flag
    # The prompt needs to be directly after -p, not piped via stdin
    quoted_prompt = shlex.quote(prompt)
    cmd = f"{base_cmd} -p {quoted_prompt} --output-format json"

    logger.debug(f"Running command: {cmd}")
    
    # Generate a unique process ID for tracking
    process_id = f"claude-process-{uuid.uuid4().hex[:8]}"
    start_time = time.time()
    model = DEFAULT_MODEL
    
    # Record the Claude process start in metrics
    await metrics.record_claude_start(process_id, model, conversation_id)

    # Track this process with the original request data
    process_tracking.proxy_launched_processes[process_id] = {
        "command": cmd,
        "start_time": time.time(),
        "current_request": original_request,
        "status": "running"  # Explicitly mark as running
    }
    
    # Set appropriate timeout parameters based on prompt complexity
    # The longer/more complex the prompt, the more time Claude might need
    current_chunk_timeout = CLAUDE_STREAM_CHUNK_TIMEOUT
    current_max_silence = CLAUDE_STREAM_MAX_SILENCE
    
    # Adjust timeout based on prompt length or complexity if needed
    process_tracking.proxy_launched_processes[process_id]['chunk_timeout'] = current_chunk_timeout
    process_tracking.proxy_launched_processes[process_id]['max_silence'] = current_max_silence
    
    logger.info(f"Tracking new Claude process with PID {process_id}")

    process = None
    try:
        # No need for stdin=PIPE since we're passing the prompt via command line now
        logger.debug(f"[TOOLS] About to start Claude process with command: {cmd}")
        try:
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdin=None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            if process and process.pid:
                logger.info(f"[TOOLS] Successfully started Claude process with PID: {process.pid}")
            else:
                logger.error(f"[TOOLS] Failed to start Claude process, no PID assigned")
        except Exception as e:
            logger.error(f"[TOOLS] Exception while starting Claude process: {str(e)}")
            raise
        
        # Track this process
        if process and process.pid:
            # Transfer the original request from our process ID to the actual process ID
            if original_request and process_id in process_tracking.proxy_launched_processes:
                original_request_data = process_tracking.proxy_launched_processes[process_id].get("current_request")
            else:
                original_request_data = original_request
                
            process_tracking.track_claude_process(str(process.pid), cmd, original_request_data)
        
        # Wait for Claude to process the command (prompt is already in the command line)
        # Add timeout handling for non-streaming mode
        try:
            # Get process-specific timeout if available (or use the default)
            process_info = process_tracking.proxy_launched_processes.get(str(process.pid), {})
            max_silence = process_info.get('max_silence', timeout)
            logger.info(f"Using timeout of {max_silence} seconds for non-streaming request")
            logger.debug(f"Process ID: {process.pid}, Command: {cmd}")
            
            # Use asyncio.wait_for to add a timeout to communicate()
            logger.info(f"Starting process.communicate() with timeout={max_silence}s")
            try:
                logger.info(f"Non-streaming pid is: {process.pid}")
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=max_silence)
                logger.info("process.communicate() completed successfully")
            except Exception as comm_error:
                logger.error(f"Error in process.communicate(): {comm_error}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise
            
            # Log any stderr output for debugging
            if stderr:
                stderr_text = stderr.decode() if stderr else ""
                if stderr_text:
                    logger.error(f"[TOOLS] Claude process stderr output: {stderr_text}")
                    
                    # Check for authentication issues in stderr
                    if "log in" in stderr_text.lower() or "authenticate" in stderr_text.lower() or "not authenticated" in stderr_text.lower() or "login" in stderr_text.lower() or "session expired" in stderr_text.lower():
                        logger.warning("Authentication issue detected in Claude CLI")
                        # Create an auth error response that can be propagated back to the client
                        auth_error = {
                            "error": {
                                "message": "Claude CLI authentication required. Please log in using the Claude CLI.",
                                "type": "auth_error",
                                "param": None,
                                "code": "claude_auth_required"
                            },
                            "auth_required": True,
                            "user_message": "Authentication required: Please log in using the Claude CLI on your server."
                        }
                        return json.dumps(auth_error)
            else:
                logger.info("No stderr output from Claude process")
                
            if stdout:
                stdout_text = stdout.decode() if stdout else ""
                logger.info(f"stdout length: {len(stdout_text)} bytes")
                logger.debug(f"stdout preview: {stdout_text[:100]}...")
            else:
                logger.warning("No stdout output from Claude process!")
                
        except asyncio.TimeoutError:
            # Process is taking too long - likely hung
            logger.warning(f"Non-streaming Claude process {process.pid} timed out after {max_silence} seconds")
            logger.warning(f"Command that may have hung: {cmd}")
            logger.warning(f"Process state: {process}")
            # Try to get process info
            try:
                p = psutil.Process(process.pid)
                logger.warning(f"Process status: {p.status()}")
                logger.warning(f"Process creation time: {p.create_time()}")
                logger.warning(f"Process CPU times: {p.cpu_times()}")
            except Exception as psutil_error:
                logger.warning(f"Could not get psutil info for process: {psutil_error}")
            
            # Try to terminate the process
            try:
                logger.warning(f"Attempting to terminate hung Claude process {process.pid}")
                process.kill()
                # If successful, untrack this process
                process_tracking.untrack_claude_process(str(process.pid))
            except Exception as kill_error:
                logger.error(f"Failed to kill hung process: {kill_error}")
            
            # Record timeout error in metrics
            duration_ms = (time.time() - start_time) * 1000
            await metrics.record_claude_completion(process_id, duration_ms, error="Process timeout", conversation_id=conversation_id)
            
            # Raise a specific timeout exception
            raise Exception(f"Claude command timed out after {timeout} seconds")
        
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
                    openai_response = formatters.format_to_openai_chat_completion(response_obj or stdout_text, model)
                except Exception as e:
                    logger.error(f"Failed to convert to OpenAI format: {e}")
                
                # Get the original request for storing
                original_request = None
                if "current_request" in process_tracking.proxy_launched_processes.get(str(process.pid), {}):
                    original_request = process_tracking.proxy_launched_processes[str(process.pid)]["current_request"]
                
                process_tracking.store_process_output(
                    str(process.pid),
                    stdout_text,
                    stderr_text,
                    cmd,
                    prompt,
                    response_obj or stdout_text,  # Original response
                    openai_response,  # Converted response
                    model,
                    original_request
                )
            except Exception as e:
                logger.error(f"Error storing process output: {e}")
            
            # Now untrack the process
            process_tracking.untrack_claude_process(str(process.pid))
        
        if process.returncode != 0:
            stderr_text = stderr.decode() if stderr else ""
            stdout_text = stdout.decode() if stdout else ""
            
            # Check if we have an API error in stdout (this can happen with certain Claude CLI errors)
            if stdout_text and "API Error:" in stdout_text:
                error_msg = stdout_text
            elif stderr_text:
                error_msg = stderr_text
            else:
                error_msg = "Unknown error"
                
            # Log the detailed error
            logger.error(f"Claude command failed with return code {process.returncode}: {error_msg}")
            
            # Record completion with error
            await metrics.record_claude_completion(process_id, duration_ms, error=error_msg, conversation_id=conversation_id)
            
            raise Exception(f"Claude command failed: {error_msg}")
        
        output = stdout.decode()
        logger.debug(f"Raw Claude response: {output}")
        
        # Parse JSON response
        try:
            response = json.loads(output)
            logger.debug(f"Parsed Claude response: {response}")
            
            # Check for authentication error in the exact format provided
            if (isinstance(response, dict) and 
                response.get("role") == "system" and 
                "result" in response and 
                isinstance(response["result"], str) and 
                ("Invalid API key" in response["result"] or 
                 "Please run /login" in response["result"])):
                
                logger.warning(f"Authentication error detected in Claude response: {response['result']}")
                # Create an auth error response that can be propagated back to the client
                auth_error = {
                    "error": {
                        "message": response["result"],
                        "type": "auth_error",
                        "param": None,
                        "code": "claude_auth_required"
                    },
                    "auth_required": True,
                    "user_message": "Authentication required: Please log in using the Claude CLI on your server."
                }
                return json.dumps(auth_error)
            
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
            process_tracking.untrack_claude_process(str(process.pid))
            
        logger.error(f"Error running Claude command: {str(e)}")
        raise

async def stream_claude_output(prompt: str, conversation_id: str = None, original_request=None):
    """
    Run Claude with streaming JSON output and extract the content.
    Processes multiline JSON objects in the stream.
    """
    # Start building the base command
    base_cmd = f"{CLAUDE_CMD}"
    
    # We don't need to extract model - Claude command only takes -p and --output-format
    logger.info(f"[TOOLS] Preparing to run Claude in streaming mode with prompt length: {len(prompt)}")

    # Log if tools are present for debugging purposes only
    if original_request and isinstance(original_request, dict) and original_request.get('tools'):
        logger.info(f"[TOOLS] Detected tools in original_request for streaming conversation: {conversation_id}")
        logger.info(f"[TOOLS] Tools are detected but will NOT be passed to Claude directly for streaming")
            
    # Always use the conversation ID if provided (regardless of tools or message count)
    if conversation_id:
        logger.info(f"[TOOLS] Using conversation ID for streaming: {conversation_id}")
        base_cmd += f" -c {conversation_id}"
        
        # Check if we're in test mode (only create temp dirs in production)
        is_test = 'unittest' in sys.modules or os.environ.get('TESTING') == '1'
        
        if not is_test:
            # Create or get a temporary directory for this conversation
            temp_dir = get_conversation_temp_dir(conversation_id)
            
            # Set the current working directory for this conversation
            os.environ["CLAUDE_CWD"] = temp_dir
    else:
        logger.info(f"[TOOLS] No conversation ID provided for streaming")

    # Add the -p flag AFTER the conversation flag
    # The prompt needs to be directly after -p, not piped via stdin
    quoted_prompt = shlex.quote(prompt)
    cmd = f"{base_cmd} -p {quoted_prompt} --output-format stream-json"

    logger.info(f"Running command for streaming: {cmd}")
    
    # Generate a unique process ID for tracking
    process_id = f"claude-process-{uuid.uuid4().hex[:8]}"
    start_time = time.time()
    model = DEFAULT_MODEL
    
    # Record the Claude process start in metrics
    await metrics.record_claude_start(process_id, model, conversation_id)
    
    # Track this process with the original request data
    process_tracking.proxy_launched_processes[process_id] = {
        "command": cmd,
        "start_time": time.time(),
        "current_request": original_request,
        "status": "running"  # Explicitly mark as running
    }
    logger.info(f"Tracking new Claude process with PID {process_id}")

    logger.debug(f"[TOOLS] About to start Claude process with command: {cmd}")
    try:
        # Set appropriate timeout parameters based on prompt complexity
        # The longer/more complex the prompt, the more time Claude might need
        current_chunk_timeout = CLAUDE_STREAM_CHUNK_TIMEOUT
        current_max_silence = CLAUDE_STREAM_MAX_SILENCE
        
        # Store these in the process info for reference during streaming
        process_info = process_tracking.proxy_launched_processes.get(process_id, {})
        if process_info:
            process_info['chunk_timeout'] = current_chunk_timeout
            process_info['max_silence'] = current_max_silence
        
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdin=None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        if process and process.pid:
            logger.info(f"[TOOLS] Successfully started Claude process with PID: {process.pid}")
        else:
            logger.error(f"[TOOLS] Failed to start Claude process, no PID assigned")
    except Exception as e:
        logger.error(f"[TOOLS] Exception while starting Claude process: {str(e)}")
        raise
    
    # Track this process
    if process and process.pid:
        # Transfer the original request from our process ID to the actual process ID
        if original_request and process_id in process_tracking.proxy_launched_processes:
            original_request_data = process_tracking.proxy_launched_processes[process_id].get("current_request")
        else:
            original_request_data = original_request
            
        process_tracking.track_claude_process(str(process.pid), cmd, original_request_data)

    # No need to write to stdin - the prompt is in the command line
    # Just log that we're ready to process
    logger.debug("Claude process started with prompt in command line, ready to read output")

    # Check for any auth errors in stderr first
    stderr_data = b""
    try:
        # Try to read any early stderr output - non-blocking
        stderr_data = await asyncio.wait_for(process.stderr.read(1024), timeout=0.5)
    except asyncio.TimeoutError:
        # No stderr output yet, which is fine
        pass
    except Exception as e:
        logger.debug(f"Error reading initial stderr: {str(e)}")
    
    # If we got stderr data, check for auth issues
    if stderr_data:
        stderr_text = stderr_data.decode('utf-8')
        logger.warning(f"Early stderr from Claude: {stderr_text}")
        
        # Check for auth-related errors
        if any(term in stderr_text.lower() for term in ["log in", "authenticate", "not authenticated", "login", "session expired"]):
            logger.warning("Authentication issue detected in Claude CLI")
            # Return an auth error that can be passed back to client
            yield {
                "error": {
                    "message": "Claude CLI authentication required. Please log in using the Claude CLI.",
                    "type": "auth_error",
                    "code": "claude_auth_required"
                }
            }
            # Close the process
            process.terminate()
            return
    
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
                                                # Return plain dict object, let the higher-level formatters handle it
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
                                    # Return plain dict object, let the higher-level formatters handle it
                                    yield {"done": True}
                                    
                                    # Break out of the main loop after completion
                                    break
                                # Additional handling for system messages with cost info
                                elif "role" in json_obj and json_obj["role"] == "system":
                                    # Check for authentication errors in system message
                                    if ("result" in json_obj and 
                                        isinstance(json_obj["result"], str) and 
                                        ("Invalid API key" in json_obj["result"] or 
                                         "Please run /login" in json_obj["result"])):
                                        
                                        logger.warning(f"Authentication error detected in streaming response: {json_obj['result']}")
                                        # Return an auth error that can be passed back to client
                                        yield {
                                            "error": {
                                                "message": json_obj["result"],
                                                "type": "auth_error",
                                                "code": "claude_auth_required"
                                            }
                                        }
                                        # Mark as complete
                                        streaming_complete = True
                                        break
                                    
                                    # This is a system message with additional info
                                    elif "cost_usd" in json_obj or "duration_ms" in json_obj:
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
                                        # Return plain dict object, let the higher-level formatters handle it
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
                
                # Get process-specific timeout if available (or use the default)
                process_info = process_tracking.proxy_launched_processes.get(process.pid, {})
                chunk_timeout = process_info.get('chunk_timeout', CLAUDE_STREAM_CHUNK_TIMEOUT)
                
                # Check if we've gone too long without output - using the process-specific timeout
                if time.time() - last_chunk_time > chunk_timeout:
                    logger.warning(f"No output from Claude for {chunk_timeout} seconds, checking if process is still active")
                    
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
                        # For complex prompts, Claude may need more time between chunks
                        if time.time() - last_chunk_time > CLAUDE_STREAM_MAX_SILENCE:
                            logger.warning(f"No output for {CLAUDE_STREAM_MAX_SILENCE} seconds, assuming Claude is finished or hung")
                            # Log the command to help diagnose hanging issues
                            cmd_to_log = cmd if 'cmd' in locals() else "Unknown command"
                            logger.warning(f"Command that may have hung: {cmd_to_log}")
                            
                            # Try to terminate the process if it's hung
                            try:
                                logger.warning(f"Attempting to terminate hung Claude process {process.pid}")
                                process.kill()
                                # If successful, untrack this process
                                process_tracking.untrack_claude_process(process.pid)
                            except Exception as kill_error:
                                logger.error(f"Failed to kill hung process: {kill_error}")
                            
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
                                # Return plain dict object, let the higher-level formatters handle it
                                yield {"content": content}
                elif "stop_reason" in json_obj:
                    streaming_complete = True
                    # Return plain dict object, let the higher-level formatters handle it
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
                
                # Return plain dict object, let the higher-level formatters handle it
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
                                # Return plain dict object, let the higher-level formatters handle it
                                yield {"content": content}
                    elif "stop_reason" in json_obj:
                        streaming_complete = True
                        # Return plain dict object, let the higher-level formatters handle it
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
            # Return plain dict object, let the higher-level formatters handle it
            yield {"done": True}
                
    except Exception as e:
        logger.error(f"Error processing Claude output stream: {str(e)}", exc_info=True)
        
        # Record error in metrics
        duration_ms = (time.time() - start_time) * 1000
        await metrics.record_claude_completion(process_id, duration_ms, error=e, conversation_id=conversation_id)
        
        # Return plain dict object, let the higher-level formatters handle it
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
                    # Get the original request for storing
                    original_request = None
                    if "current_request" in process_tracking.proxy_launched_processes.get(str(process.pid), {}):
                        original_request = process_tracking.proxy_launched_processes[str(process.pid)]["current_request"]
                    
                    process_tracking.store_process_output(
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
                        },
                        model,
                        original_request
                    )
                except Exception as e:
                    logger.error(f"Error storing streaming process output: {e}")
                
                # Now untrack the process
                process_tracking.untrack_claude_process(str(process.pid))

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


# Stream response functions for both API formats

def get_real_process_id(request_info):
    """
    Get the real process ID from either the request info or the global process dict.
    Handles both temporary string IDs and numeric process IDs.
    """
    if not request_info:
        return None
    
    # Try to get PID from request info
    pid = None
    if isinstance(request_info, dict) and "pid" in request_info:
        pid = request_info["pid"]
    elif hasattr(request_info, "pid"):
        pid = request_info.pid
    
    if not pid:
        return None
    
    # Check if it's a temporary ID that maps to a real PID
    with process_tracking.process_lock:
        if isinstance(pid, str) and pid.startswith("claude-process-") and pid in process_tracking.proxy_launched_processes:
            # Get the real PID if it exists
            real_pid = process_tracking.proxy_launched_processes[pid].get("real_pid")
            if real_pid:
                return real_pid
    
    # Otherwise return the pid directly
    return pid

async def stream_openai_response(claude_prompt: str, model_name: str, conversation_id: str = None, request_id: str = None, original_request=None):
    """
    Stream responses from Claude in the appropriate format based on the client.
    Handles both OpenAI-compatible and Ollama-compatible formats.
    Uses the request_id from the client if provided.
    """
    # Check if this is an Ollama client request
    is_ollama_client = False
    if original_request:
        if isinstance(original_request, dict):
            is_ollama_client = original_request.get('ollama_client', False)
        elif hasattr(original_request, 'ollama_client'):
            is_ollama_client = original_request.ollama_client
    
    # Log based on client type
    if is_ollama_client:
        logger.info(f"Starting unified streaming with Ollama format, request_id={request_id}")
    else:
        logger.info(f"Starting unified streaming with OpenAI format, request_id={request_id}")
    
    # Use the request_id if provided, otherwise generate one
    message_id = request_id if request_id else f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    is_first_chunk = True
    full_response = ""
    tool_calls_collection = []  # Store all detected tool calls for dashboard
    completion_sent = False
    has_tools = False  # Initialize the flag to track if tools are detected
    start_time = time.time()

    try:
        # Use Claude's streaming JSON output
        async for chunk in stream_claude_output(claude_prompt, conversation_id, original_request):
            # Add detailed logging for tools troubleshooting
            logger.info(f"[TOOLS] Received chunk: {str(chunk)[:100]}...")
            logger.debug(f"Processing OpenAI stream chunk: {chunk}")
            
            # Check for errors or auth issues
            if "error" in chunk:
                # Check if this is an auth error
                if isinstance(chunk["error"], dict) and chunk["error"].get("type") == "auth_error":
                    logger.warning("Authentication error detected in stream, passing to client")
                    
                    # Format as OpenAI error response for streaming
                    auth_error_msg = {
                        "id": message_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": "Authentication required: " + chunk["error"].get("message", "Please log in with Claude CLI")
                            },
                            "finish_reason": "error"
                        }]
                    }
                    
                    # Send the error as a regular chunk
                    yield f"data: {json.dumps(auth_error_msg)}\n\n"
                    
                    # Send completion marker
                    yield "data: [DONE]\n\n"
                    
                    # Mark as completed
                    completion_sent = True
                    return
                logger.error(f"Error in OpenAI stream: {chunk['error']}")
                error_response = {
                    "error": {
                        "message": f"Error: {chunk['error']}",
                        "type": "server_error",
                        "code": 500
                    }
                }
                # Format error response based on client type
                if is_ollama_client:
                    # Ollama format error
                    ollama_model = get_ollama_model_name(model_name)
                    ollama_error = {
                        "model": ollama_model,
                        "created_at": datetime.datetime.now().isoformat() + "Z",
                        "error": error_response["error"]["message"],
                        "done": False
                    }
                    yield f"{json.dumps(ollama_error)}\n"
                else:
                    # OpenAI format error (SSE)
                    yield f"data: {json.dumps(error_response)}\n\n"
                # Format done marker based on client type
                if is_ollama_client:
                    # Ollama expects a final message with done=true
                    ollama_done = {
                        "model": model_name,
                        "created_at": created,
                        "done": True
                    }
                    yield f"{json.dumps(ollama_done)}\n"
                else:
                    # OpenAI clients expect "data: [DONE]" marker
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
                # Note: We don't reset has_tools here - it should persist across chunks
                if isinstance(content, str):
                    try:
                        # Check if it's a JSON string starting and ending with braces
                        if content.strip().startswith('{') and content.strip().endswith('}'):
                            # Try to parse as JSON
                            parsed_content = json.loads(content)
                            
                            # Handle empty object response for tools (high priority)
                            if parsed_content == {} or (isinstance(parsed_content, dict) and len(parsed_content) == 0):
                                logger.info("[TOOLS] Empty object in streaming response, formatting as empty tool_calls")
                                content = json.dumps({"tool_calls": []})
                            # Handle existing tool_calls array
                            elif "tool_calls" in parsed_content:
                                # Already in correct format, just check if tools being used
                                has_tools = len(parsed_content["tool_calls"]) > 0
                                logger.info(f"[TOOLS] Tool calls array found in stream: has_tools={has_tools}")
                                
                            # Check if it contains a tools array
                            elif isinstance(parsed_content, dict) and "tools" in parsed_content:
                                logger.info("[TOOLS] Detected tools array in streaming response, modifying for OpenWebUI compatibility")
                                # Convert Claude's "tools" format to OpenWebUI's expected "tool_calls" format
                                tool_calls = []
                                for tool in parsed_content.get("tools", []):
                                    if "function" in tool and isinstance(tool["function"], dict):
                                        tool_call = {
                                            "name": tool["function"].get("name", "unknown_tool"),
                                            "parameters": {}
                                        }
                                        # Parse the arguments if they exist
                                        arguments = tool["function"].get("arguments", "{}")
                                        if isinstance(arguments, str):
                                            try:
                                                tool_call["parameters"] = json.loads(arguments)
                                            except json.JSONDecodeError:
                                                tool_call["parameters"] = {"raw_arguments": arguments}
                                        elif isinstance(arguments, dict):
                                            tool_call["parameters"] = arguments
                                        
                                        tool_calls.append(tool_call)
                                
                                # Mark that we have tools for finish_reason - IMPORTANT: for the entire session
                                if len(tool_calls) > 0:
                                    has_tools = True
                                    logger.info(f"[TOOLS] Detected tools in content: has_tools={has_tools}")
                                    logger.info("Setting has_tools=True for the entire streaming session")
                                
                                # Create the OpenWebUI expected format
                                content = json.dumps({"tool_calls": tool_calls})
                    except json.JSONDecodeError:
                        # Not valid JSON, proceed with original content
                        pass
                
                # Accumulate full response for logging
                full_response += content
                
                # Update the streaming buffer directly - no locking needed
                try:
                    # Store in the global buffer using a common key
                    if request_id:
                        # Store in both formats - the ID and the claude-process-ID format
                        streaming_content_buffer[request_id] = full_response
                        streaming_content_buffer[f"claude-process-{request_id}"] = full_response
                        
                        # Also update any process outputs entries that match this request ID
                        for pid, output in process_tracking.process_outputs.items():
                            # Update any entries with matching request ID or similar formats
                            if (pid == request_id or 
                                pid == f"claude-process-{request_id}" or
                                output.get("request_id") == request_id):
                                
                                # Update multiple fields to ensure we capture it
                                output["stream_content"] = full_response
                                output["stream_buffer"] = full_response
                                output["streaming_content"] = full_response
                                
                                # If it was a placeholder, also update the main response field
                                if output.get("response") == "Streaming response - output sent directly to client":
                                    output["response"] = full_response
                                    
                                logger.debug(f"Updated process outputs for {pid} with streaming content")
                        
                        logger.debug(f"Updated streaming buffers for request {request_id} with content length {len(full_response)}")
                except Exception as e:
                    logger.error(f"Error updating streaming buffer: {e}")
                
                # For the first real content, log it specially
                if content and is_first_chunk:
                    logger.info(f"First OpenAI content chunk received: {content[:50]}...")
                    
                # Format in OpenAI delta format
                # Check if content might be a tool call
                is_tool_call = False
                tool_call_content = None
                
                # Handle tool calls for streaming
                if isinstance(content, str) and "tool_calls" in content and content.strip().startswith("{") and content.strip().endswith("}"):
                    try:
                        tool_data = json.loads(content)
                        if "tool_calls" in tool_data and isinstance(tool_data["tool_calls"], list) and tool_data["tool_calls"]:
                            # We have tool calls - format them properly
                            is_tool_call = True
                            
                            # Set up delta for tool calls
                            delta = {
                                # Only include role in the first chunk
                                "role": "assistant" if is_first_chunk else ""
                            }
                            
                            # If this is the first chunk with tools, include the tool calls array
                            if is_first_chunk or not has_tools:
                                # Extract the tool calls
                                raw_tool_calls = tool_data["tool_calls"]
                                
                                # Format according to OpenAI's function calling spec
                                tool_calls_array = []
                                for i, tool_call in enumerate(raw_tool_calls):
                                    formatted_tool_call = {
                                        "id": f"call_{uuid.uuid4().hex[:8]}",
                                        "type": "function",
                                        "function": {
                                            "name": tool_call.get("name", "unknown_tool"),
                                            "arguments": json.dumps(tool_call.get("parameters", {}))
                                        }
                                    }
                                    tool_calls_array.append(formatted_tool_call)
                                
                                # Set tool_calls in delta and make content null
                                delta["content"] = None
                                delta["tool_calls"] = tool_calls_array
                                
                                # Store the tool calls for the dashboard
                                tool_calls_collection.extend(tool_calls_array)
                                
                                # Mark that we have detected tools
                                has_tools = True
                                logger.info(f"[TOOLS] Detected and formatted {len(tool_calls_array)} tool calls for streaming")
                    except Exception as e:
                        logger.error(f"Error formatting tool calls in streaming: {str(e)}")
                        # Fall back to regular content
                        is_tool_call = False
                
                # Format the actual response
                if is_tool_call:
                    response = {
                        "id": message_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": delta,
                                "finish_reason": None
                            }
                        ]
                    }
                else:
                    # Regular content format
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
                    
                # Format data based on client type
                if is_ollama_client:
                    # Ollama uses NDJSON format (one JSON object per line, no SSE formatting)
                    # Convert to Ollama model name format if needed
                    ollama_model = get_ollama_model_name(model_name)
                    ollama_response = {
                        "model": ollama_model,
                        "created_at": datetime.datetime.now().isoformat() + "Z",
                        "message": {"role": "assistant", "content": content},
                        "done": False
                    }
                    ndjson_data = f"{json.dumps(ollama_response)}\n"
                    logger.debug(f"Sending Ollama NDJSON chunk: {ndjson_data[:100]}...")
                    yield ndjson_data
                else:
                    # OpenAI uses SSE format (data: prefix and double newlines)
                    sse_data = f"data: {json.dumps(response)}\n\n"
                    logger.debug(f"Sending OpenAI SSE chunk: {sse_data[:100]}...")
                    yield sse_data
                
            elif "done" in chunk and chunk["done"]:
                # This is the completion signal - send final chunk with finish_reason
                # CRITICAL: This is a completion marker from Claude
                logger.info("Received completion signal from Claude, sending final chunks")
                
                # Check if we had tools in the response to set proper finish_reason
                finish_reason = "tool_calls" if has_tools else "stop"
                logger.info(f"[TOOLS] Sending completion with finish_reason={finish_reason}")
                logger.info(f"Setting finish_reason to: {finish_reason}")
                
                # Add detailed log about tool call status
                if has_tools:
                    logger.info(f"[TOOLS] Completing a response with tool calls - this should trigger tool execution in the client")
                
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
                            "finish_reason": finish_reason
                        }
                    ]
                }
                yield f"data: {json.dumps(final_response)}\n\n"
                
                # Format done marker based on client type
                if is_ollama_client:
                    # Ollama expects a final message with done=true and more detailed stats
                    total_duration = int((time.time() - start_time) * 1000000)  # microseconds
                    ollama_model = get_ollama_model_name(model_name)
                    ollama_done = {
                        "model": ollama_model,
                        "created_at": datetime.datetime.now().isoformat() + "Z",
                        "message": {"role": "assistant", "content": ""},
                        "done_reason": finish_reason,
                        "done": True,
                        "total_duration": total_duration,
                        "load_duration": int(total_duration * 0.1),
                        "prompt_eval_count": len(claude_prompt),
                        "prompt_eval_duration": int(total_duration * 0.3),
                        "eval_count": len(full_response),
                        "eval_duration": int(total_duration * 0.6)
                    }
                    yield f"{json.dumps(ollama_done)}\n"
                else:
                    # OpenAI clients expect "data: [DONE]" marker
                    yield "data: [DONE]\n\n"
                completion_sent = True
                
                # Log complete response summary
                logger.info(f"Complete OpenAI response length: {len(full_response)} chars")
                if len(full_response) < 500:
                    logger.debug(f"Complete OpenAI response: {full_response}")
                
                # Call our new function to update streaming process outputs
                try:
                    # Call the dedicated function to update streaming outputs
                    if has_tools and tool_calls_collection:
                        # For tool calls, create a proper response
                        tool_calls_response = {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": tool_calls_collection
                        }
                        # Record in a format that would make sense as output
                        tool_calls_formatted = json.dumps(tool_calls_response, indent=2)
                        with process_tracking.process_lock:
                            # Try to update all recent processes in case we missed the right one
                            for pid, info in list(process_tracking.proxy_launched_processes.items()):
                                if (time.time() - info.get("start_time", 0) < 300):  # 5 minutes
                                    update_streaming_process_output(str(pid), tool_calls_formatted)
                    else:
                        # For regular responses, update with the text content
                        with process_tracking.process_lock:
                            # Try to update all recent processes in case we missed the right one
                            for pid, info in list(process_tracking.proxy_launched_processes.items()):
                                if (time.time() - info.get("start_time", 0) < 300):  # 5 minutes
                                    update_streaming_process_output(str(pid), full_response)
                    
                    logger.info(f"Updated streaming processes with content length: {len(full_response)}")
                except Exception as e:
                    logger.error(f"Error updating streaming process output: {e}")
                
                # Store the complete response for the dashboard, including tool calls
                if has_tools and tool_calls_collection:
                    # Create a complete message that includes all tool calls
                    complete_message = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls_collection
                    }
                    
                    # Log the tool calls for debugging
                    logger.info(f"[TOOLS] Have {len(tool_calls_collection)} tool calls to store")
                    for i, tc in enumerate(tool_calls_collection):
                        if "function" in tc:
                            logger.info(f"[TOOLS] Tool call {i+1}: {tc['function'].get('name', 'unknown')} with args: {tc['function'].get('arguments', '{}')[:100]}")
                    
                    # Find all running Claude processes and store the tool calls in all of them
                    # This ensures we don't miss the process due to ID mapping issues
                    with process_tracking.process_lock:
                        stored = False
                        for pid, process_info in process_tracking.proxy_launched_processes.items():
                            # Only update running processes started within the last few minutes
                            if (process_info.get("status") == "running" and 
                                time.time() - process_info.get("start_time", 0) < 300):  # 5 minutes
                                
                                # Store the complete response
                                process_info["complete_response"] = {
                                    "message": complete_message,
                                    "finish_reason": "tool_calls"
                                }
                                logger.info(f"[TOOLS] Stored complete tool response for process {pid} with {len(tool_calls_collection)} tool calls")
                                stored = True
                        
                        if not stored:
                            logger.warning(f"[TOOLS] Failed to find a running Claude process to store tool calls")
                
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
            # Use the existing has_tools flag to determine finish_reason
            finish_reason = "tool_calls" if has_tools else "stop"
            logger.info(f"[TOOLS] Sending completion with finish_reason={finish_reason}")
            logger.info(f"Fallback finish_reason: {finish_reason}")
            
            final_response = {
                "id": message_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},  # Empty delta for the final chunk
                        "finish_reason": finish_reason
                    }
                ]
            }
            yield f"data: {json.dumps(final_response)}\n\n"
            
            # Format done marker based on client type
            if is_ollama_client:
                # Ollama expects a final message with done=true and more detailed stats
                total_duration = int((time.time() - start_time) * 1000000)  # microseconds
                ollama_model = get_ollama_model_name(model_name)
                ollama_done = {
                    "model": ollama_model,
                    "created_at": datetime.datetime.now().isoformat() + "Z",
                    "message": {"role": "assistant", "content": ""},
                    "done_reason": finish_reason,
                    "done": True,
                    "total_duration": total_duration,
                    "load_duration": int(total_duration * 0.1),
                    "prompt_eval_count": len(claude_prompt),
                    "prompt_eval_duration": int(total_duration * 0.3),
                    "eval_count": len(full_response),
                    "eval_duration": int(total_duration * 0.6)
                }
                yield f"{json.dumps(ollama_done)}\n"
            else:
                # OpenAI clients expect "data: [DONE]" marker
                yield "data: [DONE]\n\n"
            
            # Log complete response summary
            logger.info(f"Complete OpenAI response length: {len(full_response)} chars")
            if len(full_response) < 500:
                logger.debug(f"Complete OpenAI response: {full_response}")
                
            # Call our new function to update streaming process outputs (fallback case)
            try:
                # Call the dedicated function to update streaming outputs
                if has_tools and tool_calls_collection:
                    # For tool calls, create a proper response
                    tool_calls_response = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls_collection
                    }
                    # Record in a format that would make sense as output
                    tool_calls_formatted = json.dumps(tool_calls_response, indent=2)
                    with process_tracking.process_lock:
                        # Try to update all recent processes in case we missed the right one
                        for pid, info in list(process_tracking.proxy_launched_processes.items()):
                            if (time.time() - info.get("start_time", 0) < 300):  # 5 minutes
                                update_streaming_process_output(str(pid), tool_calls_formatted)
                else:
                    # For regular responses, update with the text content
                    with process_tracking.process_lock:
                        # Try to update all recent processes in case we missed the right one
                        for pid, info in list(process_tracking.proxy_launched_processes.items()):
                            if (time.time() - info.get("start_time", 0) < 300):  # 5 minutes
                                update_streaming_process_output(str(pid), full_response)
                
                logger.info(f"Updated streaming processes with content length: {len(full_response)} (fallback)")
            except Exception as e:
                logger.error(f"Error updating streaming process output (fallback): {e}")
            
    except Exception as e:
        logger.error(f"Error streaming from Claude with OpenAI format: {str(e)}", exc_info=True)
        
        # Send error response based on client type
        error_message = f"Streaming error: {str(e)}"
        
        if is_ollama_client:
            # Ollama format error
            ollama_error = {
                "model": model_name,
                "created_at": created,
                "error": error_message,
                "done": False
            }
            yield f"{json.dumps(ollama_error)}\n"
        else:
            # OpenAI format error
            error_response = {
                "error": {
                    "message": error_message,
                    "type": "server_error",
                    "code": 500
                }
            }
            yield f"data: {json.dumps(error_response)}\n\n"
        
        # Format done marker based on client type
        if is_ollama_client:
            # For Ollama errors, return a properly formatted error message and done marker
            ollama_model = get_ollama_model_name(model_name)
            error_obj = {
                "model": ollama_model,
                "created_at": datetime.datetime.now().isoformat() + "Z",
                "message": {
                    "role": "assistant",
                    "content": f"Error: {str(e)}"
                },
                "done_reason": "error",
                "done": True
            }
            yield f"{json.dumps(error_obj)}\n"
        else:
            # OpenAI clients expect "data: [DONE]" marker
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
    # IMPORTANT: Only use the conversation ID provided by OpenWebUI/client
    # We should NOT generate our own random conversation IDs as they leak into the user's prompt
    # Metrics can work without a conversation ID, so we'll only use what's provided

    # Format the messages for Claude Code CLI
    claude_prompt = format_messages_for_claude(request)

    # Handle streaming vs. non-streaming responses
    if request.stream:
        # Convert the request to a dict for storage
        request_dict = {
            "model": request.model,
            "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
            "stream": request.stream,
            "conversation_id": conversation_id
        }
        
        # Add system if present
        if hasattr(request, 'system') and request.system:
            request_dict["system"] = request.system
            
        # Add tools if present
        if hasattr(request, 'tools') and request.tools:
            request_dict["tools"] = request.tools
            logger.info(f"[TOOLS] Request includes tools: {len(request.tools)} tool(s)")
            
        # Check if this request came from an Ollama client (via our api/chat endpoint)
        # Can be detected in multiple ways
        is_ollama_client = False
        
        # 1. Check the data dictionary for the flag
        if 'ollama_client' in data:
            is_ollama_client = data.get('ollama_client', False)
            logger.info("Request identified as coming from Ollama client (via flag)")
            
        # 2. Check the URL path for Ollama endpoints
        try:
            if hasattr(request_body, 'scope') and 'path' in request_body.scope:
                path = request_body.scope['path']
                if path in ['/api/chat', '/api/generate']:
                    is_ollama_client = True
                    logger.info(f"Request identified as coming from Ollama client (via path: {path})")
        except Exception as e:
            logger.warning(f"Error checking request path: {e}")
        
        # Configure response format based on client type
        if is_ollama_client:
            logger.info("Using NDJSON format for Ollama streaming compatibility")
            media_type = "application/x-ndjson"
        else:
            logger.info("Using SSE format for OpenAI streaming compatibility")
            media_type = "text/event-stream"
            
        # Define headers for streaming response
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": media_type,
            "Access-Control-Allow-Origin": "*",
            "X-Accel-Buffering": "no"  # Prevents buffering in Nginx, which helps with streaming
        }
        
        # Log headers for debugging
        logger.info(f"Sending response with media_type={media_type}")
        
        # Use the same streaming function regardless of client type
        return StreamingResponse(
            stream_openai_response(claude_prompt, request.model, conversation_id, request.id, request_dict),
            media_type=media_type,
            headers=headers
        )
    else:
        try:
            # Start timing the non-streaming request
            start_time = time.time()
            request_id = f"req-{uuid.uuid4().hex[:8]}"
            
            logger.info(f"Processing non-streaming request {request_id} for model: {request.model}")
            
            # Convert the request to a dict for storage
            request_dict = {
                "model": request.model,
                "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
                "stream": request.stream,
                "conversation_id": conversation_id
            }
            
            # Add system if present
            if hasattr(request, 'system') and request.system:
                request_dict["system"] = request.system
                
            # Add tools if present
            if hasattr(request, 'tools') and request.tools:
                request_dict["tools"] = request.tools
                logger.info(f"[TOOLS] Request includes tools: {len(request.tools)} tool(s)")
            
            # For non-streaming, get the full response
            try:
                logger.info(f"Starting non-streaming request via run_claude_command for prompt of length {len(claude_prompt)}")
                
                # Log that we're about to run the command
                logger.info(f"About to run Claude in non-streaming mode with prompt length: {len(claude_prompt)}")
                
                # Actually run the command
                claude_response = await run_claude_command(claude_prompt, conversation_id=conversation_id, original_request=request_dict)
                
                # Check if response is a string that looks like JSON
                if isinstance(claude_response, str) and claude_response.strip().startswith('{') and claude_response.strip().endswith('}'):
                    try:
                        response_obj = json.loads(claude_response)
                        
                        # Check for authentication error in our format
                        if "error" in response_obj and isinstance(response_obj["error"], dict) and response_obj["error"].get("type") == "auth_error":
                            logger.warning("Authentication error detected in run_claude_command response")
                            # Return a friendly error response to the client
                            error_message = response_obj["error"].get("message", "Authentication required")
                            user_message = response_obj.get("user_message", "Please log in using the Claude CLI on your server")
                            
                            # Return error response in OpenAI format
                            return JSONResponse(
                                status_code=401,  # Unauthorized
                                content={
                                    "error": {
                                        "message": error_message,
                                        "type": "auth_error",
                                        "param": None,
                                        "code": "claude_auth_required"
                                    }
                                },
                                headers={"WWW-Authenticate": "Basic realm=\"Claude CLI Authentication Required\""}
                            )
                        
                        # Check for the exact Claude CLI auth error format
                        elif (response_obj.get("role") == "system" and 
                              "result" in response_obj and 
                              isinstance(response_obj["result"], str) and 
                              ("Invalid API key" in response_obj["result"] or 
                               "Please run /login" in response_obj["result"])):
                            
                            logger.warning(f"Claude CLI authentication error detected: {response_obj['result']}")
                            
                            # Return error response in OpenAI format
                            return JSONResponse(
                                status_code=401,  # Unauthorized
                                content={
                                    "error": {
                                        "message": response_obj["result"],
                                        "type": "auth_error",
                                        "param": None,
                                        "code": "claude_auth_required"
                                    }
                                },
                                headers={"WWW-Authenticate": "Basic realm=\"Claude CLI Authentication Required\""}
                            )
                    except json.JSONDecodeError:
                        # Not valid JSON, proceed as normal
                        pass
                
                # Log the raw Claude response at debug level
                if isinstance(claude_response, dict):
                    logger.debug(f"Raw Claude response: {json.dumps(claude_response)[:500]}...")
                else:
                    logger.debug(f"Raw Claude response: {str(claude_response)[:500]}...")
            except asyncio.TimeoutError as e:
                logger.error(f"Timeout error in non-streaming request: {str(e)}")
                raise Exception(f"Request timed out: The response took too long to generate. This may happen with complex prompts. Please try using streaming mode for this prompt.") 
            except Exception as e:
                logger.error(f"Error in run_claude_command: {str(e)}")
                raise
                
            # Log information about the client
            user_agent = request_body.headers.get("user-agent", "Unknown")
            referer = request_body.headers.get("referer", "Unknown")
            origin = request_body.headers.get("origin", "Unknown")
            host = request_body.headers.get("host", "Unknown")
            
            # Detect OpenWebUI specifically with detailed logging
            is_openwebui = (
                "OpenWebUI" in user_agent or 
                "openwebui" in str(referer).lower() or 
                "openwebui" in str(origin).lower()
            )
            
            logger.info(f"Client info for {request_id}: UA={user_agent}, Referer={referer}, Origin={origin}, Host={host}")
            if is_openwebui:
                logger.info(f"Detected OpenWebUI client for request {request_id}")
                logger.info(f"OpenWebUI User-Agent: {user_agent}")
                logger.info(f"OpenWebUI Referer: {referer}")
                logger.info(f"OpenWebUI Origin: {origin}")
            
            # Format as OpenAI chat completion
            # IMPORTANT: Don't pass conversation_id as the request_id to prevent leakage
            # Only use the request ID for this
            openai_response = formatters.format_to_openai_chat_completion(claude_response, request.model, request.id)
            
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
            
            # Log headers for debugging tools handshaking
            logger.info(f"[TOOLS] Sending response headers: {headers}")
            
            # Log if we detected OpenWebUI
            if is_openwebui:
                logger.info(f"Detected OpenWebUI client for request {request_id}")
            
            # Add debug for OpenWebUI specific response
            if is_openwebui:
                logger.info(f"OpenWebUI response content (first 200 chars): {json.dumps(openai_response)[:200]}")
                if "choices" in openai_response and openai_response["choices"] and "content" in openai_response["choices"][0]["message"]:
                    message_content = openai_response["choices"][0]["message"]["content"]
                    logger.info(f"OpenWebUI message.content (first 200 chars): {message_content[:200]}")
                    
                    # Check if this contains tool_calls
                    if isinstance(message_content, str) and "tool_calls" in message_content:
                        logger.info("OpenWebUI response contains tool_calls - checking the format")
                        try:
                            tool_content = json.loads(message_content)
                            if "tool_calls" in tool_content:
                                logger.info(f"Tool calls found: {json.dumps(tool_content['tool_calls'])[:200]}")
                        except:
                            logger.warning("Failed to parse tool_calls content as JSON")
            
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
            
            # Store in process outputs directly for debugging in dashboard
            for process_id, process_info in list(process_tracking.proxy_launched_processes.items()):
                if "pid" in process_info:
                    # Create a dict representation of the request
                    request_dict = {
                        "model": request.model,
                        "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
                        "stream": request.stream,
                        "conversation_id": conversation_id
                    }
                    
                    # Add system if present
                    if hasattr(request, 'system') and request.system:
                        request_dict["system"] = request.system
                    
                    # Update the process info with the original request
                    process_info["current_request"] = request_dict
                    
                    # Check if we have output already stored
                    pid_str = str(process_info.get("pid"))
                    if pid_str in process_tracking.process_outputs:
                        # Update the existing output with the request data
                        process_tracking.process_outputs[pid_str]["original_request"] = request_dict
                        logger.info(f"Updated output with request data for process {pid_str}")
            
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
@app.options("/test_client_info")
@app.options("/test_tool_calling")
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
        "version": API_VERSION,
        "build": BUILD_NAME
    }

@app.get("/api/version")
async def get_api_version():
    """Get API version info (Ollama-compatible API version endpoint)."""
    # Return simplified format to match Ollama's format exactly
    return {
        "version": API_VERSION
    }

@app.get("/api/tags")
async def get_tags():
    """Get list of available models (Ollama-compatible tags endpoint)."""
    models = []
    # Add all available models to the response
    for model in AVAILABLE_MODELS:
        # Use helper function to format model name with appropriate tag
        model_name = model['name']
        ollama_model_name = get_ollama_model_name(model_name)
        
        model_entry = {
            "name": ollama_model_name,  # Use name with appropriate tag
            "model": ollama_model_name,  # Adding "model" field at root level to match Ollama format
            "modified_at": model.get("modified_at", datetime.datetime.now().isoformat()),
            "size": model.get("size", 0),
            "digest": model.get("digest", ""),
            "details": {
                "parent_model": model["details"].get("parent_model", ""),
                "format": model["details"].get("format", "api"),
                "model": ollama_model_name,  # Also update model name in details
                "family": model["details"]["family"],
                "families": model["details"].get("families", [model["details"]["family"]]),
                "parameter_size": model["details"]["parameter_size"],
                "quantization_level": model["details"]["quantization_level"]
            }
        }
        models.append(model_entry)
    
    return {"models": models}

# Ollama API Chat Request Model
class OllamaChatMessage(BaseModel):
    role: str
    content: str

class OllamaChatRequest(BaseModel):
    model: str
    messages: List[OllamaChatMessage]
    stream: bool = True
    options: Optional[Dict[str, Any]] = None
    format: Optional[str] = None

# Helper function to handle Ollama model name convention
def get_ollama_model_name(model_name: str) -> str:
    """
    Apply Ollama model naming conventions:
    - Extract base model name for internal processing by stripping tags (everything after ":")
    - Add ":latest" tag if no tag is present in the model name
    
    Args:
        model_name: Original model name, may include a tag (e.g. "model:tag")
        
    Returns:
        Ollama-formatted model name with appropriate tag
    """
    if ":" not in model_name:
        return f"{model_name}:latest"
    return model_name

# Helper function to extract base model name
def get_base_model_name(model_name: str) -> str:
    """
    Extract the base model name by removing any tags.
    
    Args:
        model_name: Original model name, may include a tag (e.g. "model:tag")
        
    Returns:
        Base model name without tags
    """
    if ":" in model_name:
        return model_name.split(":")[0]
    return model_name

# No longer using separate Ollama streaming function
# We now use a unified approach through the OpenAI chat endpoint

@app.post("/api/chat")
async def ollama_chat(request: OllamaChatRequest):
    """
    Ollama-compatible chat API endpoint.
    """
    logger.info(f"Received Ollama chat request for model: {request.model}")
    
    # Convert to Claude model name
    model = get_base_model_name(request.model)
    
    # Format for Claude CLI
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    claude_prompt = "\n\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    
    # Explicitly use Ollama format for responses
    is_ollama_client = True
    
    # Create metadata for streaming
    request_id = f"ollama-req-{uuid.uuid4().hex[:8]}"
    request_dict = {
        "model": model,
        "messages": messages,
        "stream": request.stream,
        "ollama_client": True
    }
    
    # Configure media type for stream format
    media_type = "application/x-ndjson"  # Ollama uses NDJSON
    
    # Define headers for streaming response
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Content-Type": media_type,
        "Access-Control-Allow-Origin": "*",
        "X-Accel-Buffering": "no"  # Prevents buffering in Nginx, which helps with streaming
    }
    
    # If streaming, return the streaming response
    if request.stream:
        logger.info(f"Using NDJSON format for Ollama streaming")
        
        # Create an Ollama-specific streaming function
        async def stream_ollama_response():
            async for chunk in stream_claude_output(claude_prompt, None, request_dict):
                if "content" in chunk:
                    content = chunk["content"]
                    # Format in Ollama format
                    ollama_model = get_ollama_model_name(model)
                    response = {
                        "model": ollama_model,
                        "created_at": datetime.datetime.now().isoformat() + "Z",
                        "message": {
                            "role": "assistant",
                            "content": content
                        },
                        "done": False
                    }
                    yield json.dumps(response) + "\n"
                elif "done" in chunk and chunk["done"]:
                    # Final message with done flag
                    ollama_model = get_ollama_model_name(model)
                    done_msg = {
                        "model": ollama_model,
                        "created_at": datetime.datetime.now().isoformat() + "Z",
                        "done": True
                    }
                    yield json.dumps(done_msg) + "\n"
        
        return StreamingResponse(
            stream_ollama_response(),
            media_type=media_type,
            headers=headers
        )
    else:
        # For non-streaming, get the full response
        try:
            logger.info(f"Starting non-streaming request via run_claude_command for prompt of length {len(claude_prompt)}")
            
            # Log the actual command that will be run (important for debugging)
            # Start building the base command
            base_cmd = f"{CLAUDE_CMD}"
            if model:
                base_cmd += f" --model {model}"
            quoted_prompt = shlex.quote(claude_prompt)
            cmd = f"{base_cmd} -p {quoted_prompt} --output-format json"
            logger.debug(f"Non-streaming command: {cmd}")
            
            # Actually run the command
            claude_response = await run_claude_command(claude_prompt, conversation_id=None, original_request=request_dict)
            
            # Log the successful response
            logger.info(f"Non-streaming request completed successfully, response type: {type(claude_response)}")
            if isinstance(claude_response, dict):
                logger.info(f"Response keys: {list(claude_response.keys())}")
            else:
                logger.info(f"Response (not a dict): {str(claude_response)[:100]}...")
            
            # Format as Ollama response
            ollama_model = get_ollama_model_name(model)
            ollama_response = {
                "model": ollama_model,
                "created_at": datetime.datetime.now().isoformat() + "Z",
                "message": {
                    "role": "assistant",
                    "content": claude_response.get("content", "")
                },
                "done": True
            }
        except Exception as e:
            # Return a graceful error response with detailed info
            logger.error(f"ERROR in non-streaming response: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            ollama_model = get_ollama_model_name(model)
            ollama_response = {
                "model": ollama_model,
                "created_at": datetime.datetime.now().isoformat() + "Z",
                "message": {
                    "role": "assistant", 
                    "content": f"Error: {str(e)}"
                },
                "done": True,
                "error": str(e)
            }
        
        return JSONResponse(
            content=ollama_response,
            media_type="application/json",
            headers={"Access-Control-Allow-Origin": "*"}
        )

# Ollama generate endpoint model
class OllamaGenerateRequest(BaseModel):
    model: str
    prompt: str
    stream: bool = True
    options: Optional[Dict[str, Any]] = None
    format: Optional[str] = None
    
@app.post("/api/generate")
async def ollama_generate(request: OllamaGenerateRequest):
    """
    Ollama-compatible completion/generate API endpoint.
    Similar to chat but takes a single prompt instead of messages array.
    """
    logger.info(f"Received Ollama generate request for model: {request.model}")
    
    # Convert to Claude model name
    model = get_base_model_name(request.model)
    
    # Format as a user prompt
    claude_prompt = f"user: {request.prompt}"
    
    # Create metadata for streaming
    request_id = f"ollama-gen-{uuid.uuid4().hex[:8]}"
    request_dict = {
        "model": model,
        "messages": [{"role": "user", "content": request.prompt}],
        "stream": request.stream,
        "ollama_client": True
    }
    
    # Configure media type for stream format
    media_type = "application/x-ndjson"  # Ollama uses NDJSON
    
    # Define headers for streaming response
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Content-Type": media_type,
        "Access-Control-Allow-Origin": "*",
        "X-Accel-Buffering": "no"  # Prevents buffering in Nginx, which helps with streaming
    }
    
    # If streaming, return the streaming response
    if request.stream:
        logger.info(f"Using NDJSON format for Ollama streaming")
        
        # Create an Ollama-specific streaming function
        async def stream_ollama_response():
            async for chunk in stream_claude_output(claude_prompt, None, request_dict):
                if "content" in chunk:
                    content = chunk["content"]
                    # Format in Ollama format
                    ollama_model = get_ollama_model_name(model)
                    response = {
                        "model": ollama_model,
                        "created_at": datetime.datetime.now().isoformat() + "Z",
                        "response": content,
                        "done": False
                    }
                    yield json.dumps(response) + "\n"
                elif "done" in chunk and chunk["done"]:
                    # Final message with done flag
                    ollama_model = get_ollama_model_name(model)
                    done_msg = {
                        "model": ollama_model,
                        "created_at": datetime.datetime.now().isoformat() + "Z",
                        "done": True
                    }
                    yield json.dumps(done_msg) + "\n"
        
        return StreamingResponse(
            stream_ollama_response(),
            media_type=media_type,
            headers=headers
        )
    else:
        # For non-streaming, get the full response
        claude_response = await run_claude_command(claude_prompt, conversation_id=None, original_request=request_dict)
        
        # Format as Ollama response
        ollama_model = get_ollama_model_name(model)
        ollama_response = {
            "model": ollama_model,
            "created_at": datetime.datetime.now().isoformat() + "Z",
            "response": claude_response.get("content", ""),
            "done": True
        }
        
        return JSONResponse(
            content=ollama_response,
            media_type="application/json",
            headers={"Access-Control-Allow-Origin": "*"}
        )

@app.post("/test_tool_calling")
async def test_tool_calling(request: Request):
    """Test endpoint for OpenWebUI compatibility, especially for tool/function calling."""
    # Only accessible in debug mode
    if not DEBUG:
        raise HTTPException(status_code=404, detail="Endpoint not available in production mode")
    body = await request.json()
    response_type = body.get("response_type", "tool")  # tool, text, or error
    
    logger.debug(f"Testing tool response with type: {response_type}")
    
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
        response = formatters.format_to_openai_chat_completion(
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

@app.post("/test_client_info")
async def test_client_info(request: Request):
    """Special endpoint for diagnosing client compatibility issues."""
    # Only accessible in debug mode
    if not DEBUG:
        raise HTTPException(status_code=404, detail="Endpoint not available in production mode")
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
    
    # Test 2: Check metrics data generation
    print("Test 2: Checking metrics data generation...")
    try:
        metrics_data = metrics.get_metrics()
        if not isinstance(metrics_data, dict):
            print("FAIL: metrics.get_metrics() did not return a dictionary")
            return False
        if 'uptime' not in metrics_data or 'claude_invocations' not in metrics_data:
            print(f"FAIL: Metrics data is missing required fields")
            return False
        print("PASS: Metrics data generation works correctly")
    except Exception as e:
        print(f"FAIL: Error generating metrics data: {e}")
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



def update_streaming_process_output(pid, content):
    """Update a streaming process output with the full response content"""
    global streaming_content_buffer
    logger.info(f"Updating streaming process output for PID {pid} with content length: {len(content)}")
    
    # Create global buffer for streaming content if it doesn't exist
    if 'streaming_content_buffer' not in globals():
        global streaming_content_buffer
        streaming_content_buffer = {}
    
    # Always store the streaming content in a global buffer for easy access
    streaming_content_buffer[pid] = content
        
    # Update in process_tracking.process_outputs
    if pid in process_tracking.process_outputs:
        # Set the actual response content directly - overwrite the placeholder
        process_tracking.process_outputs[pid]["response"] = content
        # Also store it in special fields for dashboard access
        process_tracking.process_outputs[pid]["final_output"] = content
        process_tracking.process_outputs[pid]["stream_buffer"] = content  # Add dedicated field for streaming buffer
        process_tracking.process_outputs[pid]["stream_data"] = content    # Alternative name in case code looks for this
        logger.info(f"Updated multiple content fields for process {pid}")
        
        # Also update the message content in the converted response
        if "converted_response" in process_tracking.process_outputs[pid]:
            try:
                conv_resp = process_tracking.process_outputs[pid]["converted_response"]
                if isinstance(conv_resp, dict) and "choices" in conv_resp:
                    for choice in conv_resp["choices"]:
                        if "message" in choice:
                            # Replace the placeholder with actual content
                            choice["message"]["content"] = content
                            logger.info(f"Updated converted response message content for {pid}")
                
                # Also set status to finished if present
                if isinstance(conv_resp, dict) and "status" in conv_resp:
                    conv_resp["status"] = "finished"
                    logger.info(f"Updated status to finished in converted response for {pid}")
                    
                # Store updated converted_response back in process_tracking.process_outputs
                process_tracking.process_outputs[pid]["converted_response"] = conv_resp
            except Exception as e:
                logger.error(f"Error updating converted response: {e}")
        
        # Also update process status
        if pid in process_tracking.proxy_launched_processes:
            # Update the status to finished
            process_tracking.proxy_launched_processes[pid]["status"] = "finished"
            logger.info(f"Marked process {pid} as finished")
            
            # Store content in multiple places to make it accessible
            process_tracking.proxy_launched_processes[pid]["content"] = content
            process_tracking.proxy_launched_processes[pid]["stream_buffer"] = content
            process_tracking.proxy_launched_processes[pid]["streaming_content"] = content
            
            # Add complete_response for the dashboard to display
            complete_message = {
                "role": "assistant",
                "content": content
            }
            process_tracking.proxy_launched_processes[pid]["complete_response"] = {
                "message": complete_message,
                "finish_reason": "stop"
            }
            logger.info(f"Added content in multiple fields for process {pid}")
        
        # Log a sample of the content for debugging
        content_sample = content[:50] + "..." if len(content) > 50 else content
        logger.info(f"Final streaming content for {pid}: {content_sample}")


# Initialize dashboard if the module is available
# Import dashboard at the end to avoid circular imports
try:
    # Initialize dashboard with required dependencies
    dashboard.init_dashboard(
        app=app,
        claude_metrics=metrics,
    )
    logger.info("Dashboard module initialized")
except (ImportError, AttributeError) as e:
    logger.error(f"Dashboard module not found or error initializing: {e}, embedded dashboard will be used", exc_info=True)

if __name__ == "__main__":
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