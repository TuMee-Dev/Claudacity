import json
import logging
import uuid

logger = logging.getLogger(__name__)

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
    has_tools = False  # Flag to track if tools are detected in the response
    
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
                    # Handle Python vs JSON quotes
                    parsed_result = None
                    if "'" in raw_result and not '"' in raw_result:
                        # Convert Python single quotes to JSON double quotes
                        quoted_result = raw_result.replace("'", '"')
                        try:
                            parsed_result = json.loads(quoted_result)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse quoted result: {quoted_result[:100]}")
                            parsed_result = {}
                    else:
                        # Otherwise try to parse as-is
                        try:
                            parsed_result = json.loads(raw_result)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse raw result: {raw_result[:100]}")
                            parsed_result = {}

                    # The critical fix - handle empty responses first (highest priority)
                    if parsed_result == {} or not parsed_result or (isinstance(parsed_result, dict) and len(parsed_result) == 0):
                        logger.info("[TOOLS] Empty object detected, formatting as empty tool_calls")
                        content = json.dumps({"tool_calls": []})
                        has_tools = False
                    # Then handle empty tool_calls array specifically
                    elif isinstance(parsed_result, dict) and "tool_calls" in parsed_result and parsed_result["tool_calls"] == []:
                        logger.info("[TOOLS] Empty tool_calls array detected")
                        content = raw_result
                        has_tools = False
                    # Handle direct tool_calls format with content
                    elif isinstance(parsed_result, dict) and "tool_calls" in parsed_result:
                        logger.info("[TOOLS] Tool calls array found, using as-is")
                        content = raw_result
                        has_tools = len(parsed_result["tool_calls"]) > 0
                        logger.info(f"[TOOLS] Has tools: {has_tools}")
                    # Handle Claude's native tools format
                    elif isinstance(parsed_result, dict) and "tools" in parsed_result:
                        logger.info("[TOOLS] Detected tools array in response, modifying for OpenWebUI compatibility")
                        logger.info(f"Original tools format: {json.dumps(parsed_result.get('tools', []))}")
                        # Convert Claude's "tools" format to OpenWebUI's expected "tool_calls" format
                        tool_calls = []
                        for tool in parsed_result.get("tools", []):
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
                        
                        # Create the OpenWebUI expected format
                        content = json.dumps({"tool_calls": tool_calls})
                        logger.info(f"Converted to tool_calls format: {content}")
                        
                        # Set a flag to indicate we have tools
                        has_tools = len(tool_calls) > 0
                        logger.info(f"[TOOLS] Tools detected: {has_tools}")
                    else:
                        # Not tool-related, use as-is
                        content = parsed_result
                except json.JSONDecodeError:
                    # If parsing fails, use the raw string
                    content = raw_result
                    logger.warning(f"Failed to parse result as JSON: {raw_result[:100]}... Using raw response.")
                except Exception as e:
                    # Catch-all error handler
                    logger.error(f"Error in tool response handling: {str(e)}")
                    content = json.dumps({"tool_calls": []})  # Default to empty tool calls on error
                    has_tools = False
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
    
    # Add more detailed logging for debugging
    logger.info(f"Final content type: {type(content)}")
    if isinstance(content, str):
        logger.info(f"Final content string (first 200 chars): {content[:200]}")
        if content.startswith('{') and "tool_calls" in content:
            logger.info("Content appears to be a JSON string containing tool_calls")
    else:
        logger.info(f"Final content dict/object: {json.dumps(content)[:200] if hasattr(content, '__dict__') else str(content)[:200]}")
    
    # Determine if we have tool calls and process them correctly for OpenAI format
    finish_reason = "stop"
    message = {
        "role": "assistant",
    }
    
    tool_calls_array = []
    
    # Check if content contains tool calls
    if has_tools:
        finish_reason = "tool_calls"
        logger.info(f"Processing tool calls for OpenAI format")
        
        try:
            # If content is a string that looks like JSON with tool_calls, parse it
            if isinstance(content, str) and "tool_calls" in content:
                tool_data = json.loads(content)
                if "tool_calls" in tool_data and isinstance(tool_data["tool_calls"], list):
                    # Extract the tool calls
                    raw_tool_calls = tool_data["tool_calls"]
                    
                    # Format according to OpenAI's function calling spec
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
                    
                    logger.info(f"Formatted {len(tool_calls_array)} tool calls for OpenAI compatibility")
            
            # If we have tool calls, set them in the message and set content to null
            if tool_calls_array:
                message["content"] = None
                message["tool_calls"] = tool_calls_array
                logger.info(f"Set tool_calls in message and set content to null")
            else:
                # No tool calls, use the content as normal
                message["content"] = content if isinstance(content, str) else json.dumps(content)
                finish_reason = "stop"
        except Exception as e:
            # If there's any error in tool call formatting, fall back to content
            logger.error(f"Error formatting tool calls: {str(e)}")
            message["content"] = content if isinstance(content, str) else json.dumps(content)
            finish_reason = "stop"
    else:
        # No tool calls detected, use normal content
        message["content"] = content if isinstance(content, str) else json.dumps(content)
    
    logger.info(f"Non-streaming response finish_reason: {finish_reason}")
    
    # Format the response in OpenAI chat completion format
    openai_response = {
        "id": message_id,
        "object": "chat.completion",
        "created": timestamp,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }
    
    # One more level of logging - exact message content that will be sent to OpenWebUI
    logger.info(f"Final message.content in OpenAI response: {openai_response['choices'][0]['message']['content'][:200]}")
    
    logger.debug(f"Converted to OpenAI format: {str(openai_response)[:200]}...")
    return openai_response
