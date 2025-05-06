"""
Test implementation of an Ollama-compatible chat API endpoint.
This is a standalone test script to show how the chat endpoint would work.
"""

import asyncio
import json
import datetime
import time
import uvicorn
from typing import Dict, List, Optional, Any, Union
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

app = FastAPI()

# Sample model data
AVAILABLE_MODELS = [
    {
        "name": "claude-3.7-sonnet",
        "modified_at": "2025-05-05T19:53:25.564072",
        "size": 0,
        "digest": "anthropic_claude_3_7_sonnet_20250505",
        "details": {
            "model": "claude-3.7-sonnet",
            "parent_model": "",
            "format": "api",
            "family": "anthropic",
            "families": ["anthropic", "claude"],
            "parameter_size": "13B",
            "quantization_level": "none"
        }
    }
]

# Define chat request model
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: bool = True
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None

# Claude API wrapper for chat completions
# Make requests to the actual Claude service
async def proxy_to_claude_service(messages: List[dict], stream: bool = False):
    """
    Proxy the request to the actual running Claude service.
    This allows us to use the real service instead of fake responses.
    """
    import aiohttp
    
    # The actual Claude service running on the default port
    CLAUDE_SERVICE_URL = "http://localhost:22434/v1/chat/completions"
    
    # Convert from ollama format to OpenAI format
    openai_messages = []
    for msg in messages:
        openai_messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    
    payload = {
        "model": "claude-3.7-sonnet",
        "messages": openai_messages,
        "stream": stream,
        "max_tokens": 1000
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(CLAUDE_SERVICE_URL, json=payload) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise Exception(f"Error from Claude service: {resp.status} - {error_text}")
                
                if stream:
                    return resp
                else:
                    return await resp.json()
    
    except Exception as e:
        print(f"Error proxying to Claude service: {str(e)}")
        # Return a fallback response in case of error
        if stream:
            return None
        else:
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": f"Error connecting to Claude service: {str(e)}"
                    },
                    "finish_reason": "error"
                }]
            }

async def get_claude_response_stream(model: str, messages: List[dict], max_tokens: int = 1000, temperature: float = 0.7):
    """
    Handle a streaming chat request by proxying to the real Claude service.
    """
    start_time = time.time() * 1000000  # microseconds
    
    try:
        # Get streaming response from Claude service
        resp = await proxy_to_claude_service(messages, stream=True)
        
        if resp is None:
            # Fallback if we couldn't connect
            yield json.dumps({
                "model": model,
                "created_at": datetime.datetime.now().isoformat() + "Z",
                "message": {
                    "role": "assistant",
                    "content": "Error: Could not connect to Claude service"
                },
                "done_reason": "error",
                "done": True
            }) + "\n"
            return
        
        # Read and transform the streaming response
        buffer = ""
        async for line in resp.content:
            line = line.decode('utf-8').strip()
            if not line or line == "data: [DONE]":
                continue
                
            if line.startswith("data: "):
                line = line[6:]  # Remove "data: " prefix
                
                try:
                    data = json.loads(line)
                    
                    # Transform from OpenAI format to Ollama format
                    if "choices" in data and len(data["choices"]) > 0:
                        choice = data["choices"][0]
                        
                        if "delta" in choice and "content" in choice["delta"]:
                            content = choice["delta"]["content"]
                            if content:
                                # Create Ollama-style response chunk
                                yield json.dumps({
                                    "model": model,
                                    "created_at": datetime.datetime.now().isoformat() + "Z",
                                    "message": {
                                        "role": "assistant",
                                        "content": content
                                    },
                                    "done": False
                                }) + "\n"
                                buffer += content
                        
                        # If this is the last chunk
                        if choice.get("finish_reason") is not None:
                            end_time = time.time() * 1000000  # microseconds
                            total_duration = int(end_time - start_time)
                            
                            # Create final Ollama-style chunk
                            yield json.dumps({
                                "model": model,
                                "created_at": datetime.datetime.now().isoformat() + "Z",
                                "message": {
                                    "role": "assistant", 
                                    "content": ""
                                },
                                "done_reason": choice.get("finish_reason", "stop"),
                                "done": True,
                                "total_duration": total_duration,
                                "load_duration": int(total_duration * 0.1),
                                "prompt_eval_count": len(str(messages)),
                                "prompt_eval_duration": int(total_duration * 0.3),
                                "eval_count": len(buffer),
                                "eval_duration": int(total_duration * 0.6)
                            }) + "\n"
                except json.JSONDecodeError:
                    print(f"Error parsing JSON in streaming response: {line}")
        
    except Exception as e:
        print(f"Error in streaming response: {str(e)}")
        yield json.dumps({
            "model": model,
            "created_at": datetime.datetime.now().isoformat() + "Z",
            "message": {
                "role": "assistant",
                "content": f"Error: {str(e)}"
            },
            "done_reason": "error",
            "done": True
        }) + "\n"

async def get_claude_response_complete(model: str, messages: List[dict], max_tokens: int = 1000, temperature: float = 0.7):
    """
    Handle a non-streaming chat request by proxying to the real Claude service.
    """
    start_time = time.time() * 1000000  # microseconds
    
    try:
        # Get non-streaming response from Claude service
        openai_response = await proxy_to_claude_service(messages, stream=False)
        
        # Transform from OpenAI format to Ollama format
        if "choices" in openai_response and len(openai_response["choices"]) > 0:
            choice = openai_response["choices"][0]
            content = ""
            
            if "message" in choice and "content" in choice["message"]:
                content = choice["message"]["content"]
            
            end_time = time.time() * 1000000  # microseconds
            total_duration = int(end_time - start_time)
            
            return {
                "model": model,
                "created_at": datetime.datetime.now().isoformat() + "Z",
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "done_reason": choice.get("finish_reason", "stop"),
                "done": True,
                "total_duration": total_duration,
                "load_duration": int(total_duration * 0.1),
                "prompt_eval_count": len(str(messages)),
                "prompt_eval_duration": int(total_duration * 0.3),
                "eval_count": len(content),
                "eval_duration": int(total_duration * 0.6)
            }
    
    except Exception as e:
        print(f"Error in non-streaming response: {str(e)}")
        return {
            "model": model,
            "created_at": datetime.datetime.now().isoformat() + "Z",
            "message": {
                "role": "assistant",
                "content": f"Error: {str(e)}"
            },
            "done_reason": "error",
            "done": True
        }

@app.get("/api/version")
async def get_api_version():
    """Get API version info (Ollama-compatible API version endpoint)."""
    return {
        "version": "0.1.0"
    }

@app.get("/api/tags")
async def get_tags():
    """Get list of available models (Ollama-compatible tags endpoint)."""
    models = []
    # Add all available models to the response
    for model in AVAILABLE_MODELS:
        model_entry = {
            "name": model["name"],
            "model": model["name"],
            "modified_at": model.get("modified_at", datetime.datetime.now().isoformat()),
            "size": model.get("size", 0),
            "digest": model.get("digest", ""),
            "details": {
                "parent_model": model["details"].get("parent_model", ""),
                "format": model["details"].get("format", "api"),
                "model": model["details"]["model"],
                "family": model["details"]["family"],
                "families": model["details"].get("families", [model["details"]["family"]]),
                "parameter_size": model["details"]["parameter_size"],
                "quantization_level": model["details"]["quantization_level"]
            }
        }
        models.append(model_entry)
    
    return {"models": models}

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Ollama-compatible chat endpoint.
    Supports both streaming and non-streaming responses.
    """
    # Extract parameters
    model_name = request.model
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    stream = request.stream
    max_tokens = request.max_tokens or 1000
    temperature = request.temperature or 0.7
    
    # Verify the model exists
    model_exists = False
    for model in AVAILABLE_MODELS:
        if model["name"] == model_name:
            model_exists = True
            break
    
    if not model_exists:
        # Default to our first model if the requested model doesn't exist
        model_name = AVAILABLE_MODELS[0]["name"]
    
    # For streaming responses
    if stream:
        return StreamingResponse(
            get_claude_response_stream(model_name, messages, max_tokens, temperature),
            media_type="text/event-stream"
        )
    
    # For non-streaming responses
    response = await get_claude_response_complete(model_name, messages, max_tokens, temperature)
    return JSONResponse(content=response)

# Run the server when the script is executed directly
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting test server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)