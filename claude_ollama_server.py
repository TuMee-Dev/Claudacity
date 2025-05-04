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
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
import httpx
import asyncio
import fastapi
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

# Configuration

DEFAULT_MODEL = "claude-3-7-sonnet-latest"  # Default Claude model to report
DEFAULT_MAX_TOKENS = 128000  # 128k context length
CONVERSATION_CACHE_TTL = 3600 * 3  # 3 hours in seconds

# Configure logging with more details

logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG to get more detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

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

    # Log request details
    logger.info(f"Request {request_id}: {method} {url}")

    try:
        # Log request body for non-GET requests
        if method != "GET":
            try:
                body = await request.body()
                if body:
                    body_str = body.decode()
                    # Truncate long bodies to avoid log flood
                    if len(body_str) > 1000:
                        body_str = body_str[:1000] + "..."
                    logger.debug(f"Request {request_id} body: {body_str}")
            except Exception as e:
                logger.warning(f"Could not log request body: {str(e)}")
                
        # Process the request
        response = await call_next(request)
        
        # Log response code
        logger.info(f"Response {request_id}: {response.status_code}")
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

CLAUDE_CMD = "claude"  # Assumes claude is in PATH

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

    try:
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Send the prompt to Claude
        stdout, stderr = await process.communicate(input=prompt.encode())
        
        if process.returncode != 0:
            logger.error(f"Claude command failed: {stderr.decode()}")
            raise Exception(f"Claude command failed: {stderr.decode()}")
        
        output = stdout.decode()
        logger.debug(f"Raw Claude response: {output}")
        
        # Parse JSON response
        try:
            response = json.loads(output)
            logger.debug(f"Parsed Claude response: {response}")
            
            # If we have a system response with a result, use that content
            if isinstance(response, dict) and "role" in response and response["role"] == "system" and "result" in response:
                duration_ms = response.get("duration_ms", 0)
                cost_usd = response.get("cost_usd", 0)
                result = response["result"]
                
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
                            "duration_ms": duration_ms,
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
                    "duration_ms": duration_ms,
                    "cost_usd": cost_usd
                }
            
            # Return the response as-is if it doesn't match expected format
            return response
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response, returning raw output")
            return output
            
    except Exception as e:
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

    process = await asyncio.create_subprocess_shell(
        cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    # Write the prompt to stdin
    try:
        process.stdin.write(prompt.encode())
        await process.stdin.drain()
        process.stdin.close()
    except Exception as e:
        logger.error(f"Error writing to stdin: {str(e)}")
        raise

    # Read the output
    try:
        # Buffer for collecting complete JSON objects
        buffer = ""
        
        # Read the output in chunks
        while True:
            chunk = await process.stdout.read(1024)
            if not chunk:
                break
                
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
                                            yield {"content": content}
                            elif "stop_reason" in json_obj:
                                # End of message
                                yield {"done": True}
                            # Additional handling for system messages with cost info
                            elif "role" in json_obj and json_obj["role"] == "system" and "cost_usd" in json_obj:
                                # This is a system message with cost info
                                yield {"done": True}
                            
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse JSON: {e}")
                        
                        # Remove the processed object from the buffer
                        buffer = buffer[i+1:]
                        # Reset to scan the buffer from the beginning
                        break
        
        # If any data remains in the buffer, try to parse it
        if buffer:
            try:
                json_obj = json.loads(buffer)
                logger.debug(f"Parsed final JSON object: {str(json_obj)[:200]}...")
                
                if "type" in json_obj and json_obj["type"] == "message":
                    if "content" in json_obj and isinstance(json_obj["content"], list):
                        for item in json_obj["content"]:
                            if item.get("type") == "text" and "text" in item:
                                content = item["text"]
                                logger.info(f"Extracted final text content: {content[:50]}...")
                                yield {"content": content}
                elif "stop_reason" in json_obj:
                    yield {"done": True}
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse final buffer: {buffer[:200]}...")
        
        # Check for any errors
        stderr_data = await process.stderr.read()
        if stderr_data:
            stderr_str = stderr_data.decode('utf-8').strip()
            if stderr_str:
                logger.error(f"Error from Claude: {stderr_str}")
                yield {"error": stderr_str}
                
    except Exception as e:
        logger.error(f"Error processing Claude output stream: {str(e)}", exc_info=True)
        yield {"error": str(e)}
        
    finally:
        # Ensure the process is terminated
        if process and process.returncode is None:
            try:
                process.terminate()
                await asyncio.wait_for(process.wait(), timeout=1.0)
            except (asyncio.TimeoutError, ProcessLookupError):
                logger.warning("Had to force kill Claude process")
                try:
                    process.kill()
                except ProcessLookupError:
                    pass

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
                yield f"data: {json.dumps(error_response)}\n\n"
                yield "data: [DONE]\n\n"
                return
                
            # Extract content based on chunk format
            content = ""
            is_final = False
            
            if "content" in chunk:
                content = chunk["content"]
            elif "done" in chunk and chunk["done"]:
                # Send final chunk with finish_reason
                logger.info("Sending final OpenAI chunk")
                final_response = {
                    "id": message_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }
                    ]
                }
                yield f"data: {json.dumps(final_response)}\n\n"
                yield "data: [DONE]\n\n"
                
                # Log complete response summary
                logger.info(f"Complete OpenAI response length: {len(full_response)} chars")
                if len(full_response) < 500:
                    logger.debug(f"Complete OpenAI response: {full_response}")
                return
            else:
                # Log unrecognized format but don't interrupt the stream
                logger.warning(f"Unrecognized OpenAI chunk format: {chunk}")
                continue
                
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
            
        # Ensure we send the final [DONE] marker
        yield "data: [DONE]\n\n"
            
    except Exception as e:
        logger.error(f"Error streaming from Claude with OpenAI format: {str(e)}", exc_info=True)
        error_response = {
            "error": {
                "message": f"Streaming error: {str(e)}",
                "type": "server_error",
                "code": 500
            }
        }
        yield f"data: {json.dumps(error_response)}\n\n"
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
        # For streaming, use the OpenAI streaming format
        return StreamingResponse(
            stream_openai_response(claude_prompt, request.model, conversation_id, request.id),
            media_type="text/event-stream"
        )
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

@app.get("/", response_class=PlainTextResponse)
async def root():
    """Root endpoint that mimics Ollama's root response."""
    return "Ollama is running"

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