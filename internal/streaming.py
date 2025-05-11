import asyncio
import datetime
import json
import os
import shlex
import sys
import time
import uuid
import internal.process_tracking as process_tracking
import internal.models as models
import logging

logger = logging.getLogger(__name__)

# Process and streaming response timeouts
CLAUDE_STREAM_CHUNK_TIMEOUT = 18.0  # Seconds without output before checking process status (was 10, originally 5)
CLAUDE_STREAM_MAX_SILENCE = 180.0  # Maximum seconds to wait with no output before assuming process hung (was 60, originally 15)

# Stream response functions for both API formats

async def stream_openai_response(metrics, claude_cmd: str, claude_prompt: str, model_name: str, conversation_id: str = None, request_id: str = None, original_request=None):
    """
    Stream responses from Claude in the appropriate format based on the client.
    Handles both OpenAI-compatible and Ollama-compatible formats.
    Uses the request_id from the client if provided.
    
    The original_request parameter is critical - it contains the full client request data
    which is needed for the dashboard Request tab. Make sure this is propagated
    all the way through the streaming process.
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
        async for chunk in stream_claude_output(metrics, claude_cmd, claude_prompt, conversation_id, original_request):
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
                    ollama_model = models.get_ollama_model_name(model_name)
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
                        process_tracking.streaming_content_buffer[request_id] = full_response
                        process_tracking.streaming_content_buffer[f"claude-process-{request_id}"] = full_response
                        
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
                    ollama_model = models.get_ollama_model_name(model_name)
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
                    ollama_model = models.get_ollama_model_name(model_name)
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
                                    update_streaming_process_output(pid, tool_calls_formatted)
                    else:
                        # For regular responses, update with the text content
                        with process_tracking.process_lock:
                            # Try to update all recent processes in case we missed the right one
                            for pid, info in list(process_tracking.proxy_launched_processes.items()):
                                if (time.time() - info.get("start_time", 0) < 300):  # 5 minutes
                                    update_streaming_process_output(pid, full_response)
                    
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
                ollama_model = models.get_ollama_model_name(model_name)
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
                                update_streaming_process_output(pid, tool_calls_formatted)
                else:
                    # For regular responses, update with the text content
                    with process_tracking.process_lock:
                        # Try to update all recent processes in case we missed the right one
                        for pid, info in list(process_tracking.proxy_launched_processes.items()):
                            if (time.time() - info.get("start_time", 0) < 300):  # 5 minutes
                                update_streaming_process_output(pid, full_response)
                
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
            ollama_model = models.get_ollama_model_name(model_name)
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


async def stream_claude_output(metrics, claude_cmd:str, prompt: str, conversation_id: str = None, original_request=None):
    """
    Run Claude with streaming JSON output and extract the content.
    Processes multiline JSON objects in the stream.
    """
    logger.info(f"[TOOLS] Preparing to run Claude in streaming mode with prompt length: {len(prompt)}")

    # Use process_tracking's run_claude_command with stream=True
    # This will launch the process with streaming output format and handle common initialization
    try:
        process_result = await process_tracking.run_claude_command(
            claude_cmd=claude_cmd,
            prompt=prompt,
            conversation_id=conversation_id,
            original_request=original_request,
            timeout=CLAUDE_STREAM_MAX_SILENCE,
            stream=True,  # Enable streaming mode
            metrics=metrics
        )

        # For streaming mode, run_claude_command returns a tuple: (process, process_id, cmd, start_time, model)
        process, process_id, cmd, start_time, model = process_result

        logger.info(f"Successfully launched Claude process for streaming: {process_id}")
    except Exception as e:
        logger.error(f"Error launching Claude process for streaming: {str(e)}")
        raise

    logger.debug("Claude process started with prompt in command line, ready to read output")

    # Now continue with the streaming-specific handling of the process

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
                                                logger.debug(f"Extracted text content: {content[:50]}...")
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

                                        # For system messages, extract the result as content
                                        if "result" in json_obj and isinstance(json_obj["result"], str):
                                            content = json_obj["result"]
                                            logger.info(f"Extracted result from system message: {content[:50]}...")
                                            yield {"content": content}

                                        # Mark as complete
                                        streaming_complete = True

                                        # Record completion in metrics
                                        await metrics.record_claude_completion(
                                            process_id,
                                            duration_ms,
                                            output_tokens=int(output_tokens),
                                            conversation_id=conversation_id
                                        )

                                        # Then send completion signal
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
                                process_tracking.untrack_claude_process(process_id)
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
                elif "role" in json_obj and json_obj["role"] == "system" and "result" in json_obj:
                    # Handle system message with result
                    content = json_obj["result"]
                    logger.info(f"Extracted result from final system message: {content[:50]}...")
                    yield {"content": content}
                    streaming_complete = True
                    yield {"done": True}
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
                    elif "role" in json_obj and json_obj["role"] == "system" and "result" in json_obj:
                        # Handle system message with result
                        content = json_obj["result"]
                        logger.info(f"Extracted result from final buffered system message: {content[:50]}...")
                        yield {"content": content}
                        streaming_complete = True
                        yield {"done": True}
                        return
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
                    
                    # Also check the temp process ID for the original request if needed
                    if original_request is None and "current_request" in process_tracking.proxy_launched_processes.get(process_id, {}):
                        original_request = process_tracking.proxy_launched_processes[process_id]["current_request"]
                        logger.info(f"Found original request in temp process ID {process_id}")
                    
                    # Last chance - use the original_request parameter that was passed to this function
                    if original_request is None and 'original_request' in locals():
                        original_request = locals()['original_request']  # Use the parameter passed to this function
                        logger.info("Using original request from function parameter")
                    
                    # Log what we found
                    if original_request is not None:
                        logger.info(f"Found original request to store with streaming process output")
                    else:
                        logger.warning(f"No original request found for streaming process {process.pid}")
                    
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
                process_tracking.untrack_claude_process(process.pid)
                process_tracking.untrack_claude_process(process_id)

def update_streaming_process_output(pid, content):
    """Update a streaming process output with the full response content"""
    # Always store the streaming content in a global buffer for easy access
    process_tracking.streaming_content_buffer[pid] = content
    logger.info(f"Updating streaming process output for PID {pid} with content length: {len(content)}")
        
    # Update in process_tracking.process_outputs
    if pid in process_tracking.process_outputs:
        # Get original_request before overwriting anything else
        original_request = process_tracking.process_outputs[pid].get("original_request")
        
        # Set the actual response content directly - overwrite the placeholder
        process_tracking.process_outputs[pid]["response"] = content
        # Also store it in special fields for dashboard access
        process_tracking.process_outputs[pid]["final_output"] = content
        process_tracking.process_outputs[pid]["stream_buffer"] = content  # Add dedicated field for streaming buffer
        process_tracking.process_outputs[pid]["stream_data"] = content    # Alternative name in case code looks for this
        logger.info(f"Updated multiple content fields for process {pid}")
        
        # If we had a stored original_request, make sure it persists
        if original_request is not None:
            process_tracking.process_outputs[pid]["original_request"] = original_request
            logger.info(f"Preserved original_request during streaming update for {pid}")
        
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
            
            # Make sure we copy the original_request from proxy_launched_processes to process_outputs
            # This ensures the Request tab in the dashboard shows the request details
            if "current_request" in process_tracking.proxy_launched_processes[pid]:
                current_request = process_tracking.proxy_launched_processes[pid]["current_request"]
                if current_request is not None:
                    process_tracking.process_outputs[pid]["original_request"] = current_request
                    logger.info(f"Copied original_request from proxy_launched_processes to process_outputs for {pid}")
            
            logger.info(f"Added content in multiple fields for process {pid}")
        
        # Log a sample of the content for debugging
        content_sample = content[:50] + "..." if len(content) > 50 else content
        logger.info(f"Final streaming content for {pid}: {content_sample}")
