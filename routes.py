import asyncio
import json
import uuid
import time
import traceback
from fastapi import FastAPI, Request, HTTPException # type: ignore
from fastapi.responses import StreamingResponse, JSONResponse # type: ignore
import claude_metrics
from claude_metrics import ClaudeMetrics
import process_tracking
import models
import formatters
from models import ChatRequest
import conversations
import streaming
import logging

logger = logging.getLogger(__name__)

def register_routes(app: FastAPI):
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
            conversation_id = conversations.get_conversation_id(request.id)
            if conversation_id is None and len(request.messages) > 0:
                # This is a new conversation with an ID
                # Generate a unique conversation ID for Claude (use the request ID itself)
                conversation_id = request.id
                conversations.set_conversation_id(request.id, conversation_id)
                logger.info(f"Created new conversation with ID: {conversation_id}")
        # IMPORTANT: Only use the conversation ID provided by OpenWebUI/client
        # We should NOT generate our own random conversation IDs as they leak into the user's prompt
        # Metrics can work without a conversation ID, so we'll only use what's provided

        # Format the messages for Claude Code CLI
        claude_prompt = models.format_messages_for_claude(request)

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
                streaming.stream_openai_response(claude_metrics.global_metrics, claude_prompt, request.model, conversation_id, request.id, request_dict),
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
                    claude_response = await process_tracking.run_claude_command(claude_prompt, conversation_id=conversation_id, original_request=request_dict)
                    
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
                                    content=models.create_auth_error_response(error_message),
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
                                    content=models.create_auth_error_response(response_obj["result"]),
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
                    logger.info(f"Active conversations: {len(claude_metrics.global_metrics.active_conversations)}")
                    logger.info(f"Unique conversations: {len(claude_metrics.global_metrics.unique_conversations)}")
                
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
                        pid = process_info.get("pid")
                        if pid in process_tracking.process_outputs:
                            # Update the existing output with the request data
                            process_tracking.process_outputs[pid]["original_request"] = request_dict
                            logger.info(f"Updated output with request data for process {pid}")
                
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
