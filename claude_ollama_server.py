"""
Claude-compatible Ollama API Server.

This FastAPI application provides an OpenAI-compatible API server that interfaces with
a locally running Claude Code process. It implements both Ollama API and OpenAI API format.
"""

import argparse
import asyncio
import datetime
from datetime import timezone
import json
import logging
import os
import psutil # type: ignore
import shlex
import sys
import tempfile
import time
import traceback
import uuid
import uvicorn # type: ignore
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field # type: ignore
from fastapi import FastAPI, Request, HTTPException # type: ignore
from fastapi.responses import StreamingResponse, JSONResponse # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from claude_metrics import ClaudeMetrics
import dashboard
import formatters
import process_tracking
import streaming
import models
# Configuration
import config
from config import AVAILABLE_MODELS, DEFAULT_MAX_TOKENS, CONVERSATION_CACHE_TTL
from models import ChatRequest

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Set up file handler for logging to logs directory
log_file_path = os.path.join("logs", "server.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S'))

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG if config.DEBUG else logging.INFO,  # Use config.DEBUG flag to control log verbosity
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Get logger and add file handler explicitly
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)

# Metrics tracking class

# Initialize metrics
metrics = ClaudeMetrics()

# Conversation cache to track active conversations
# Keys are conversation IDs, values are (timestamp, conversation_id) tuples
conversation_cache = {}

# Background task for cleanup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run background tasks when the server starts."""
    # Start background cleanup task
    logger.info("Application startup: Starting periodic cleanup task.")
    periodic_cleanup_task = asyncio.create_task(periodic_cleanup())
    yield
    # Stop background cleanup task
    if periodic_cleanup_task:
        logger.info("Application shutdown: Stopping periodic cleanup task.") 
        periodic_cleanup_task.cancel()
        try:
            await periodic_cleanup_task
        except asyncio.CancelledError:
            logger.info("Periodic cleanup task was successfully cancelled and has exited.")
        except Exception as e: # Catch any other potential error during task shutdown
            logger.error(f"Error during periodic cleanup task shutdown: {e}", exc_info=True)

# Initialize FastAPI app

app = FastAPI(
    title="Claude Ollama API",
    description="OpenAI-compatible API server for Claude Code",
    version=config.API_VERSION,
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
                for conv_id in list(models.conversation_temp_dirs.keys()):
                    if conv_id not in active_conversations:
                        models.cleanup_conversation_temp_dir(conv_id)
                        logger.debug(f"Cleaned up temp directory for inactive conversation: {conv_id}")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Error cleaning up conversation temp dirs: {e}")
        except asyncio.CancelledError: # <<< --- ADDED THIS EXCEPTION HANDLER
            logger.info("Periodic cleanup task is stopping.")
            break # Exit the loop to terminate the task
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


# Utility functions for working with Claude Code CLI


async def run_claude_command(prompt: str, conversation_id: str = None, original_request=None, timeout: float = streaming.CLAUDE_STREAM_MAX_SILENCE) -> str:
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
        base_cmd += f" -r {conversation_id}"
        
        # Check if we're in test mode (only create temp dirs in production)
        is_test = 'unittest' in sys.modules or os.environ.get('TESTING') == '1'
        
        if not is_test:
            # Create or get a temporary directory for this conversation
            temp_dir = models.get_conversation_temp_dir(conversation_id)
            
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
    model = models.DEFAULT_MODEL
    
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
    current_chunk_timeout = streaming.CLAUDE_STREAM_CHUNK_TIMEOUT
    current_max_silence = streaming.CLAUDE_STREAM_MAX_SILENCE
    
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
                        auth_error = _create_auth_error_response("Claude CLI authentication required. Please log in using the Claude CLI.")
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
                auth_error = _create_auth_error_response(response["result"])
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

def _create_auth_error_response(error_message):
    """Create a standardized Claude authentication error response"""
    return {
        "error": {
            "message": error_message,
            "type": "auth_error",
            "param": None,
            "code": "claude_auth_required"
        },
        "auth_required": True,
        "user_message": "Authentication required: Please log in using the Claude CLI."
    }

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
            logger.debug("Using NDJSON format for Ollama streaming compatibility")
            media_type = "application/x-ndjson"
        else:
            logger.debug("Using SSE format for OpenAI streaming compatibility")
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
            streaming.stream_openai_response(metrics, claude_prompt, request.model, conversation_id, request.id, request_dict),
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
                                content=_create_auth_error_response(error_message),
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
                                content=_create_auth_error_response(response_obj["result"]),
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
        "version": config.API_VERSION,
        "build": config.BUILD_NAME
    }

@app.get("/api/version")
async def get_api_version():
    """Get API version info (Ollama-compatible API version endpoint)."""
    # Return simplified format to match Ollama's format exactly
    return {
        "version": config.API_VERSION
    }

@app.get("/api/tags")
async def get_tags():
    """Get list of available models (Ollama-compatible tags endpoint)."""
    models = []
    # Add all available models to the response
    for model in config.AVAILABLE_MODELS:
        # Use helper function to format model name with appropriate tag
        model_name = model['name']
        ollama_model_name = models.get_ollama_model_name(model_name)
        
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
        logger.debug(f"Using NDJSON format for Ollama streaming")
        
        # Create an Ollama-specific streaming function
        async def stream_ollama_response():
            async for chunk in streaming.stream_claude_output(metrics, CLAUDE_CMD, claude_prompt, None, request_dict):
                if "content" in chunk:
                    content = chunk["content"]
                    # Format in Ollama format
                    ollama_model = models.get_ollama_model_name(model)
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
                    ollama_model = models.get_ollama_model_name(model)
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
                content_str = claude_response.get("content", "")
            else:
                logger.info(f"Response (not a dict): {str(claude_response)[:100]}...")
                content_str = str(claude_response)
                
            # Format as Ollama response
            ollama_model = models.get_ollama_model_name(model)
            ollama_response = {
                "model": ollama_model,
                "created_at": datetime.datetime.now().isoformat() + "Z",
                "message": {
                    "role": "assistant",
                    "content": content_str
                },
                "done": True
            }
        except Exception as e:
            # Return a graceful error response with detailed info
            logger.error(f"ERROR in non-streaming response: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            ollama_model = models.get_ollama_model_name(model)
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
        logger.debug(f"Using NDJSON format for Ollama streaming")
        
        # Create an Ollama-specific streaming function
        async def stream_ollama_response():
            async for chunk in streaming.stream_claude_output(metrics, CLAUDE_CMD, claude_prompt, None, request_dict):
                if "content" in chunk:
                    content = chunk["content"]
                    # Format in Ollama format
                    ollama_model = models.get_ollama_model_name(model)
                    response = {
                        "model": ollama_model,
                        "created_at": datetime.datetime.now().isoformat() + "Z",
                        "response": content,
                        "done": False
                    }
                    yield json.dumps(response) + "\n"
                elif "done" in chunk and chunk["done"]:
                    # Final message with done flag
                    ollama_model = models.get_ollama_model_name(model)
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
        ollama_model = models.get_ollama_model_name(model)
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
    if not config.DEBUG:
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
    if not config.DEBUG:
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
    logger.error(f"Dashboard module not found or error initializing: {e}", exc_info=True)

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