"""
Claude-compatible Ollama API Server.

This FastAPI application provides an OpenAI-compatible API server that interfaces with
a locally running Claude Code process. It implements both Ollama API and OpenAI API format.
"""

import argparse
import asyncio
import datetime
import json
import os
import shlex
import sys
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
import internal.claude_metrics as claude_metrics
from internal.claude_metrics import ClaudeMetrics
import internal.dashboard as dashboard
import internal.formatters as formatters
import internal.process_tracking as process_tracking
import internal.streaming as streaming
import internal.models as models
import internal.conversations as conversations
import internal.routes as routes
# Configuration
import config

from internal.logging_config import setup_logging
debug = False
logger = setup_logging(debug)

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
routes.register_routes(app)


async def periodic_cleanup():
    """Periodically clean up old data to prevent memory leaks."""
    while True:
        try:
            # Clean up every 5 minutes
            await asyncio.sleep(5 * 60)  # 5 minutes
            
            # Clean up old metrics data
            claude_metrics.global_metrics.prune_old_data()
            logger.debug("Performed periodic cleanup of old metrics data")
            conversations.conversation_cache_cleanup()
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

# Utility functions for working with Claude Code CLI



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
    model_list = []
    # Add all available models to the response
    for model in config.AVAILABLE_MODELS:
        # Use helper function to format model name with appropriate tag
        model_name = model['name']
        # Use the get_ollama_model_name function from the models module that's already imported
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
        model_list.append(model_entry)
    
    return {"models": model_list}

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
            async for chunk in streaming.stream_claude_output(claude_metrics.global_metrics, CLAUDE_CMD, claude_prompt, None, request_dict):
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
            claude_response = await process_tracking.run_claude_command(claude_prompt, conversation_id=None, original_request=request_dict)
            
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
            async for chunk in streaming.stream_claude_output(claude_metrics.global_metrics, CLAUDE_CMD, claude_prompt, None, request_dict):
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
        claude_response = await process_tracking.run_claude_command(claude_prompt, conversation_id=None, original_request=request_dict)
        
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
        if not hasattr(claude_metrics.global_metrics, 'total_invocations'):
            print("FAIL: metrics does not have total_invocations attribute")
            return False
        if not hasattr(claude_metrics.global_metrics, 'active_conversations'):
            print("FAIL: metrics does not have active_conversations attribute")
            return False
        print("PASS: Metrics initialized correctly")
    except Exception as e:
        print(f"FAIL: Error checking metrics: {e}")
        return False
    
    # Test 2: Check metrics data generation
    print("Test 2: Checking metrics data generation...")
    try:
        metrics_data = claude_metrics.global_metrics.get_metrics()
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
        conversations.set_conversation_id(test_req_id, test_conv_id)
        
        # Retrieve it
        retrieved_id = conversations.get_conversation_id(test_req_id)
        
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
        claude_metrics=claude_metrics.global_metrics,
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