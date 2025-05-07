# Ollama API Implementation

This file contains the implementation of the `/api/chat` endpoint for Ollama API compatibility.

## Overview

Based on our analysis of the codebase, we already have:

1. An OpenAI-compatible `/v1/chat/completions` endpoint
2. Functions for handling both streaming and non-streaming responses
3. Existing model definitions in the `AVAILABLE_MODELS` array

Our implementation will:

1. Reuse existing chat functionality but format responses according to Ollama's API
2. Support both streaming and non-streaming modes
3. Use our existing Claude service backend

## Implementation

Add this code to claude_ollama_server.py:

```python
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

# Streaming function for Ollama format
async def stream_ollama_response(claude_prompt: str, model_name: str, 
                                 conversation_id: str = None, 
                                 request_id: str = None, 
                                 original_request=None):
    """
    Stream responses from Claude in Ollama-compatible format.
    Based on our existing stream_openai_response function.
    """
    import time
    
    logger.info(f"Starting stream_ollama_response, request_id={request_id}, conversation_id={conversation_id}")
    
    # Use the request_id if provided, otherwise generate one
    if not request_id:
        request_id = f"req-{uuid.uuid4().hex[:8]}"
    
    # Start timing
    start_time = time.time()
    
    # Create the Claude CLI process
    cmd = get_claude_command(claude_prompt)
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    logger.info(f"Started Claude process (PID: {process.pid}) for request: {request_id}")
    
    # Buffer for reading output
    buffer = ""
    last_chunk_time = time.time()
    streaming_complete = False
    
    # Statistics for tracking
    tokens_generated = 0
    
    # Loop until we get the full response or timeout
    while True:
        try:
            # Try to read more data with a timeout
            chunk = await asyncio.wait_for(process.stdout.read(1024), 1.0)
            
            if not chunk:
                logger.info(f"End of output from Claude for request: {request_id}")
                break
                
            # Got some data, update the last chunk time
            last_chunk_time = time.time()
            
            # Add to our buffer and process line by line
            buffer += chunk.decode('utf-8')
            
            # Process the buffer - splitting by newlines
            i = 0
            while True:
                i = buffer.find('\\n', i)
                if i == -1:
                    break
                
                # Extract the token and send it as a streaming response
                token = buffer[:i].strip()
                if token:
                    tokens_generated += 1
                    
                    # Format in Ollama streaming format
                    message_chunk = {
                        "model": model_name,
                        "created_at": datetime.datetime.now().isoformat() + "Z",
                        "message": {
                            "role": "assistant",
                            "content": token
                        },
                        "done": False
                    }
                    
                    # Send the chunk
                    yield json.dumps(message_chunk) + "\\n"
                    
                    # Remove the processed part from the buffer
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
            if time.time() - last_chunk_time > 15.0:
                logger.warning("No output from Claude for 15 seconds, assuming completion")
                break
    
    # Process any remaining content in the buffer
    if buffer:
        # Format and send the final content
        message_chunk = {
            "model": model_name,
            "created_at": datetime.datetime.now().isoformat() + "Z",
            "message": {
                "role": "assistant",
                "content": buffer
            },
            "done": False
        }
        
        yield json.dumps(message_chunk) + "\\n"
    
    # Clean up the process if it's still running
    if process.returncode is None:
        try:
            process.terminate()
            await process.wait()
        except:
            pass
    
    # Calculate timing information
    end_time = time.time()
    total_duration = int((end_time - start_time) * 1000000)  # microseconds
    
    # Send final completion message
    final_message = {
        "model": model_name,
        "created_at": datetime.datetime.now().isoformat() + "Z",
        "message": {
            "role": "assistant",
            "content": ""  # Empty final content
        },
        "done_reason": "stop",
        "done": True,
        "total_duration": total_duration,
        "load_duration": int(total_duration * 0.1),  # Simulated metrics
        "prompt_eval_count": len(claude_prompt),
        "prompt_eval_duration": int(total_duration * 0.3),
        "eval_count": tokens_generated,
        "eval_duration": int(total_duration * 0.6)
    }
    
    yield json.dumps(final_message) + "\\n"
    logger.info(f"Completed streaming response for request: {request_id}")

@app.post("/api/chat")
async def ollama_chat(request: OllamaChatRequest):
    """
    Ollama-compatible chat API endpoint.
    Reuses our existing Claude integration but formats responses in Ollama's format.
    """
    logger.info(f"Received Ollama chat request for model: {request.model}")
    
    # Find the model or use default
    model_name = request.model
    model_exists = False
    for model in config.AVAILABLE_MODELS:
        if model["name"] == model_name:
            model_exists = True
            break
    
    if not model_exists:
        model_name = config.AVAILABLE_MODELS[0]["name"] if config.AVAILABLE_MODELS else "claude-3.7-sonnet"
        logger.info(f"Model {request.model} not found, using {model_name} instead")
    
    # Extract options
    stream = request.stream
    options = request.options or {}
    temperature = options.get("temperature", 0.7)
    max_tokens = options.get("max_tokens", 4096)
    
    # Generate a conversation ID if needed
    conversation_id = f"conv-{uuid.uuid4().hex[:8]}"
    
    # Format messages for Claude
    formatted_messages = []
    for msg in request.messages:
        formatted_messages.append(f"{msg.role.capitalize()}: {msg.content}")
    
    claude_prompt = "\\n\\n".join(formatted_messages)
    if not claude_prompt.endswith("Assistant:"):
        claude_prompt += "\\n\\nAssistant:"
    
    # Handle streaming requests
    if stream:
        logger.info(f"Using streaming response for Ollama compatibility")
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "Access-Control-Allow-Origin": "*"
        }
        
        return StreamingResponse(
            stream_ollama_response(claude_prompt, model_name, conversation_id),
            media_type="text/event-stream", 
            headers=headers
        )
    
    # Handle non-streaming requests
    else:
        logger.info(f"Processing non-streaming Ollama request")
        
        # Run Claude CLI
        start_time = time.time()
        cmd = get_claude_command(claude_prompt)
        process = await asyncio.create_subprocess_exec(
            *cmd, 
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        end_time = time.time()
        
        if process.returncode != 0:
            logger.error(f"Claude process error: {stderr.decode()}")
            raise HTTPException(status_code=500, detail="Error processing request")
        
        response_text = stdout.decode().strip()
        total_duration = int((end_time - start_time) * 1000000)  # microseconds
        
        # Format in Ollama's response format
        ollama_response = {
            "model": model_name,
            "created_at": datetime.datetime.now().isoformat() + "Z",
            "message": {
                "role": "assistant",
                "content": response_text
            },
            "done_reason": "stop",
            "done": True,
            "total_duration": total_duration,
            "load_duration": int(total_duration * 0.1),  # Example values
            "prompt_eval_count": len(claude_prompt),
            "prompt_eval_duration": int(total_duration * 0.3),
            "eval_count": len(response_text),
            "eval_duration": int(total_duration * 0.6)
        }
        
        return JSONResponse(content=ollama_response)
```

## Usage

Once implemented, this endpoint will:

1. Accept POST requests to `/api/chat`
2. Process them using the Claude CLI
3. Format responses in Ollama's format
4. Support both streaming and non-streaming modes

This will enable Ollama clients to work directly with our Claude service.
