import asyncio
import collections
import datetime
import json
import psutil # type: ignore
import os
import shlex
import subprocess
import sys
import time
import threading
import uuid
import logging
import internal.formatters as formatters
import internal.streaming as streaming
import traceback
from typing import Union
import internal.models as models
from internal.claude_metrics import ClaudeMetrics
import internal.claude_metrics as claude_metrics

logger = logging.getLogger(__name__)

# Dictionary to track proxy-launched Claude processes
proxy_launched_processes = {}
# Dictionary to store outputs from recent processes (limited to last 20)
process_outputs = collections.OrderedDict()
MAX_STORED_OUTPUTS = 20
# Lock for process access - we need to use threading.Lock() not asyncio.Lock() at module level
process_lock = threading.Lock()  # Must use threading.Lock() not asyncio.Lock() at module level

# Initialize streaming buffer if needed
streaming_content_buffer = {}

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

async def run_claude_command(claude_cmd: str, prompt: str, conversation_id: str = None, original_request=None, timeout: float = streaming.CLAUDE_STREAM_MAX_SILENCE) -> str:
    """Run a Claude Code command and return the output."""
    # Log the request details for debugging
    logger.debug(f"run_claude_command called with:")
    logger.debug(f"  - prompt length: {len(prompt)}")
    logger.debug(f"  - conversation_id: {conversation_id}")
    logger.debug(f"  - timeout: {timeout}s")
    
    # Start building the base command
    base_cmd = f"{claude_cmd}"

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
    await claude_metrics.global_metrics.record_claude_start(process_id, model, conversation_id)

    # Track this process with the original request data
    proxy_launched_processes[process_id] = {
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
    proxy_launched_processes[process_id]['chunk_timeout'] = current_chunk_timeout
    proxy_launched_processes[process_id]['max_silence'] = current_max_silence
    
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
            if original_request and process_id in proxy_launched_processes:
                original_request_data = proxy_launched_processes[process_id].get("current_request")
            else:
                original_request_data = original_request
                
            track_claude_process(process.pid, cmd, original_request_data)
        
        # Wait for Claude to process the command (prompt is already in the command line)
        # Add timeout handling for non-streaming mode
        try:
            # Get process-specific timeout if available (or use the default)
            process_info = proxy_launched_processes.get(str(process.pid), {})
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
                        auth_error = models.create_auth_error_response("Claude CLI authentication required. Please log in using the Claude CLI.")
                        # Untrack the temporary process ID before returning
                        untrack_claude_process(process_id)
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
                untrack_claude_process(process.pid)
                untrack_claude_process(process_id)
            except Exception as kill_error:
                logger.error(f"Failed to kill hung process: {kill_error}")
            
            # Record timeout error in metrics
            duration_ms = (time.time() - start_time) * 1000
            await claude_metrics.global_metrics.record_claude_completion(process_id, duration_ms, error="Process timeout", conversation_id=conversation_id)
            
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
                if "current_request" in proxy_launched_processes.get(str(process.pid), {}):
                    original_request = proxy_launched_processes[str(process.pid)]["current_request"]
                
                store_process_output(
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
            untrack_claude_process(process.pid)
            untrack_claude_process(process_id)
        
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
            await claude_metrics.global_metrics.record_claude_completion(process_id, duration_ms, error=error_msg, memory_mb=0, conversation_id=conversation_id)
            
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
                auth_error = models.create_auth_error_response(response["result"])
                # Untrack the temporary process ID before returning
                untrack_claude_process(process_id)
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
                
                # Record completion metrics with memory usage
                try:
                    memory_mb = 0
                    # Try to get current memory usage if this is a real process
                    if process and process.pid and psutil.pid_exists(process.pid):
                        proc = psutil.Process(process.pid)
                        memory_info = proc.memory_info()
                        memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
                    await claude_metrics.global_metrics.record_claude_completion(process_id, response_duration_ms, output_tokens, memory_mb=memory_mb, conversation_id=conversation_id)
                except Exception as e:
                    # If we can't get memory, just record the standard metrics
                    logger.error(f"Error getting memory usage for completion: {e}")
                    await claude_metrics.global_metrics.record_claude_completion(process_id, response_duration_ms, output_tokens, conversation_id=conversation_id)
                
                # Try to parse as JSON if it looks like JSON
                try:
                    if isinstance(result, str) and result.strip().startswith('{') and result.strip().endswith('}'):
                        parsed_result = json.loads(result)
                        # Return a structured response with parsed content
                        # Untrack the temporary process ID before returning
                        untrack_claude_process(process_id)
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
                # Untrack the temporary process ID before returning
                untrack_claude_process(process_id)
                return {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                    "role": "assistant",
                    "parsed_json": False,
                    "content": result,
                    "duration_ms": response_duration_ms,
                    "cost_usd": cost_usd
                }
            
            # Record completion metrics using our calculated duration - with memory usage
            try:
                memory_mb = 0
                # Try to get current memory usage if this is a real process
                if process and process.pid and psutil.pid_exists(process.pid):
                    proc = psutil.Process(process.pid)
                    memory_info = proc.memory_info()
                    memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
                await claude_metrics.global_metrics.record_claude_completion(process_id, duration_ms, output_tokens, memory_mb=memory_mb, conversation_id=conversation_id)
            except Exception as e:
                # If we can't get memory, just record the standard metrics
                logger.error(f"Error getting memory usage for completion: {e}")
                await claude_metrics.global_metrics.record_claude_completion(process_id, duration_ms, output_tokens, conversation_id=conversation_id)
            
            # Return the response as-is if it doesn't match expected format
            # Untrack the temporary process ID before returning
            untrack_claude_process(process_id)
            return response
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response, returning raw output")
            
            # Record completion metrics with memory usage
            try:
                memory_mb = 0
                # Try to get current memory usage if this is a real process
                if process and process.pid and psutil.pid_exists(process.pid):
                    proc = psutil.Process(process.pid)
                    memory_info = proc.memory_info()
                    memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
                await claude_metrics.global_metrics.record_claude_completion(process_id, duration_ms, None, memory_mb=memory_mb, conversation_id=conversation_id)
            except Exception as e:
                # If we can't get memory, just record the standard metrics
                logger.error(f"Error getting memory usage for completion: {e}")
                await claude_metrics.global_metrics.record_claude_completion(process_id, duration_ms, None, conversation_id=conversation_id)
            
            # Untrack the temporary process ID before returning
            untrack_claude_process(process_id)
            return output
            
    except Exception as e:
        # Record error in metrics with memory info if possible
        duration_ms = (time.time() - start_time) * 1000
        try:
            memory_mb = 0
            # Try to get system memory instead since process may have failed
            memory_mb = psutil.virtual_memory().used / (1024 * 1024)  # Convert to MB
            await claude_metrics.global_metrics.record_claude_completion(process_id, duration_ms, None, memory_mb=memory_mb, error=e, conversation_id=conversation_id)
        except Exception as mem_error:
            # If we can't get memory, just record the standard metrics
            logger.error(f"Error getting memory usage for error completion: {mem_error}")
            await claude_metrics.global_metrics.record_claude_completion(process_id, duration_ms, error=e, conversation_id=conversation_id)
        
        # Untrack the process if it's still tracked
        if process and process.pid:
            untrack_claude_process(process.pid)
        untrack_claude_process(process_id)
            
        logger.error(f"Error running Claude command: {str(e)}")
        raise


def track_claude_process(pid: Union[str, int], command: str, original_request=None):
    """Track a Claude process launched by this proxy server"""
    import time
    
    # If the pid is a string starting with "claude-process-", this is a temporary ID
    # In that case, look up the real process to find the corresponding proxy process ID
    temp_id = None
    if isinstance(pid, str) and not pid.startswith("claude-process-"):
        # This is a real system PID
        numeric_pid = int(pid)
        
        # Create a basic process output entry as soon as process starts
        # This ensures we can see in-progress processes in the dashboard
        if original_request:
            prompt = ""
            if isinstance(original_request, dict) and "messages" in original_request:
                for msg in original_request.get("messages", []):
                    if msg.get("role") == "user" and "content" in msg:
                        prompt += msg["content"] + "\n"
            
            store_process_output(
                pid,
                "",  # empty stdout since process is still running
                "",  # empty stderr since process is still running
                command,
                prompt.strip(),
                "Process is still running...",  # placeholder response
                {"status": "running"},  # placeholder converted response
                original_request.get("model", "<DEFAULT_MODEL>"),  # default model if not specified
                original_request
            )
        
        # Look for temporary process IDs that should be linked to this real PID
        for temp_pid, process_info in proxy_launched_processes.items():
            if isinstance(temp_pid, str) and temp_pid.startswith("claude-process-"):
                # Check if this temp_pid was recently created and doesn't already have a real PID
                if process_info.get("real_pid") is None:
                    # Link this temporary ID to the real PID
                    proxy_launched_processes[temp_pid]["real_pid"] = numeric_pid
                    temp_id = temp_pid
                    break
    
    # Store process information
    proxy_launched_processes[pid] = {
        "pid": pid,
        "command": command,
        "start_time": time.time(),
        "status": "running",
        "current_request": original_request,
        "temp_id": temp_id  # Store the temp ID if applicable
    }
    
    logger.info(f"Tracking new Claude process with PID {pid}")



def untrack_claude_process(pid):
    """Remove a Claude process from tracking or mark it as finished if needed"""
    with process_lock:
        if pid in proxy_launched_processes:
            # Get the process info before removing it
            process_info = proxy_launched_processes.get(pid, {})
            
            # Check if we should keep this process for record-keeping
            if process_info.get("keep_record", False):
                # Instead of removing, mark as finished
                proxy_launched_processes[pid]["status"] = "finished"
                logger.info(f"Marked Claude process with PID {pid} as finished (keeping record)")
                return
            
            # Otherwise remove this process
            proxy_launched_processes.pop(pid, None)
            logger.info(f"Untracked Claude process with PID {pid}")
            
            # If this is a real PID (numeric), also clean up any temporary IDs linked to it
            if not isinstance(pid, str) or not pid.startswith("claude-process-"):
                # This is a real PID - check for any temporary IDs that link to it
                for temp_pid, info in list(proxy_launched_processes.items()):
                    if (isinstance(temp_pid, str) and temp_pid.startswith("claude-process-") and 
                        info.get('real_pid') == pid):
                        # Check if we should keep this process for record-keeping
                        if info.get("keep_record", False):
                            # Instead of removing, mark as finished
                            proxy_launched_processes[temp_pid]["status"] = "finished" 
                            logger.info(f"Marked linked temporary process with PID {temp_pid} as finished (keeping record)")
                        else:
                            proxy_launched_processes.pop(temp_pid, None)
                            logger.info(f"Untracked linked temporary process with PID {temp_pid}")
            
            # If this is a temporary ID, also handle the real PID it links to
            if isinstance(pid, str) and pid.startswith("claude-process-"):
                real_pid = process_info.get('real_pid')
                if real_pid and real_pid in proxy_launched_processes:
                    # Check if we should keep this process for record-keeping
                    if proxy_launched_processes[real_pid].get("keep_record", False):
                        # Instead of removing, mark as finished
                        proxy_launched_processes[real_pid]["status"] = "finished"
                        logger.info(f"Marked linked real process with PID {real_pid} as finished (keeping record)")
                    else:
                        proxy_launched_processes.pop(real_pid, None)
                        logger.info(f"Untracked linked real process with PID {real_pid}")

def store_process_output(pid, stdout, stderr, command, prompt, response, converted_response=None, model="<DEFAULT_MODEL>", original_request=None):
    """Store the output from a Claude process"""
    timestamp = datetime.datetime.now().isoformat()
    
    # Check if we're updating an existing output for a streaming process
    if pid in process_outputs and response != "Streaming response - output sent directly to client":
        if process_outputs[pid].get("response") == "Streaming response - output sent directly to client":
            # We're updating a streaming process with the final output
            logger.info(f"Updating streaming process {pid} with final content")
            process_outputs[pid]["response"] = response
            
            # If we're passed an original request and the current one is None/empty, update it
            if original_request is not None and not process_outputs[pid].get("original_request"):
                process_outputs[pid]["original_request"] = original_request
                logger.info(f"Updated original_request for streaming process {pid}")
            
            # Also update the converted response if it exists
            if "converted_response" in process_outputs[pid] and converted_response:
                process_outputs[pid]["converted_response"] = converted_response
            elif "converted_response" in process_outputs[pid] and isinstance(response, (dict, str)):
                try:
                    # Update the message content in the existing converted response
                    conv_resp = process_outputs[pid]["converted_response"]
                    if isinstance(conv_resp, dict) and "choices" in conv_resp:
                        for choice in conv_resp["choices"]:
                            if "message" in choice:
                                choice["message"]["content"] = response
                    logger.info(f"Updated converted response content for streaming process {pid}")
                except Exception as e:
                    logger.error(f"Failed to update converted response: {e}")
            return
    
    # Create a new entry for this process
    entry = {
        "pid": pid,
        "timestamp": timestamp,
        "command": command,
        "prompt": prompt,
        "stdout": stdout,
        "stderr": stderr,
        "response": response,
        "original_request": original_request
    }
    
    # If we have a converted response, add it
    if converted_response:
        entry["converted_response"] = converted_response
    # Otherwise, try to convert it now
    elif response and isinstance(response, (dict, str)) and response != "Streaming response - output sent directly to client":
        try:
            converted = formatters.format_to_openai_chat_completion(response, model)
            entry["converted_response"] = converted
        except Exception as e:
            logger.error(f"Failed to convert response for storage: {e}")
    
    # Store the entry
    process_outputs[pid] = entry
    
    # Limit the number of stored outputs
    while len(process_outputs) > MAX_STORED_OUTPUTS:
        process_outputs.popitem(last=False)  # Remove oldest item (FIFO)
    
    logger.info(f"Stored output for process {pid}")


def get_process_output(pid):
    """Get the stored output for a process"""
    return process_outputs.get(pid, None)

def get_running_claude_processes():
    """Get information about currently running Claude processes that were launched by this proxy"""
    try:
        # List of processes to return
        active_processes = []
        
        # Track processes we've already processed (to avoid duplicates)
        processed_pids = set()
        
        # Check each tracked process to see if it's still running
        pids_to_remove = []
        max_process_age = 1800  # 30 minutes - remove processes older than this
        current_time = time.time()
        
        for pid, process_info in list(proxy_launched_processes.items()):  # Use list() to avoid dict changing during iteration
            try:
                # Check for stale processes regardless of type
                process_age = current_time - process_info.get('start_time', 0)
                if process_age > max_process_age:
                    logger.info(f"Removing stale process {pid} (age: {process_age:.1f}s)")
                    pids_to_remove.append(pid)
                    continue
                
                # Skip if already processed (avoids duplicates)
                real_pid = process_info.get('real_pid')
                if real_pid and real_pid in processed_pids:
                    continue
                
                # Handle different process ID types
                if isinstance(pid, str) and pid.startswith("claude-process-"):
                    # Check if this process has a real PID assigned
                    if real_pid:
                        # Check if the real PID process exists
                        try:
                            if not psutil.pid_exists(real_pid):
                                logger.info(f"Temporary process {pid} refers to non-existent real PID {real_pid}")
                                # Mark as finished instead of removing so it shows properly in dashboard
                                process_info['status'] = 'finished'
                                logger.info(f"Marking temp process {pid} as finished due to non-existent real PID {real_pid}")
                                continue
                            else:
                                # Skip as we'll process when we encounter the real PID
                                continue
                        except:
                            # Error checking real PID - mark as finished too
                            process_info['status'] = 'finished'
                            logger.info(f"Marking temp process {pid} as finished due to error checking real PID")
                            continue
                            
                    # Set status to finished for processes that have been around too long
                    if process_age > 300:  # 5 minutes
                        process_info['status'] = 'finished'
                        logger.info(f"Marking temp process {pid} as finished due to age: {process_age:.1f}s")
                    
                    # Only include running processes in the dashboard
                    if process_info.get('status') == 'running':
                        # For string-based process IDs without real PID, include in the list but mark as non-psutil
                        runtime_secs = int(current_time - process_info['start_time'])
                        hours, remainder = divmod(runtime_secs, 3600)
                        minutes, seconds = divmod(remainder, 60)
                        runtime = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                        
                        # Generate a temp numeric PID for display purposes only
                        # This is important since dashboard filters by numeric PIDs
                        display_pid = hash(pid) % 100000
                        if display_pid < 0:
                            display_pid += 100000
                        
                        active_processes.append({
                            "user": "claude",
                            "pid": display_pid,  # Use a derived numeric pid for filtering
                            "cpu_percent": 0.0,  # Use consistent field name
                            "memory_mb": 0.0,    # Use consistent field name 
                            "started": time.strftime("%H:%M:%S", time.localtime(process_info['start_time'])),
                            "runtime": runtime,
                            "command": process_info.get('command', 'Claude Process')[:80],
                            "original_id": pid   # Keep original ID for reference
                        })
                    continue
                
                # For numeric PIDs, convert and check if process exists
                try:
                    numeric_pid = int(pid) if isinstance(pid, str) else pid
                    
                    # Check if the process exists before creating Process object
                    if not psutil.pid_exists(numeric_pid):
                        pids_to_remove.append(pid)
                        logger.info(f"Process {numeric_pid} no longer exists")
                        continue
                    
                    # Check if it's actually a Claude process
                    process = psutil.Process(numeric_pid)
                    cmdline = process.cmdline()
                    if not ("claude" in process.name().lower() or any("claude" in arg.lower() for arg in cmdline)):
                        # Not a Claude process - either reused PID or wrong process
                        logger.info(f"Process {numeric_pid} exists but is not a Claude process: {' '.join(cmdline)}")
                        pids_to_remove.append(pid)
                        continue
                except (psutil.NoSuchProcess, ValueError, psutil.AccessDenied) as e:
                    # Process no longer exists or invalid PID format
                    logger.info(f"Error checking process {pid}: {str(e)}")
                    pids_to_remove.append(pid)
                    continue
                
                # Mark as processed to avoid duplicates
                processed_pids.add(numeric_pid)
                
                # Update the status in the process_info
                process_info['status'] = 'running'
                
                # Get process info
                with process.oneshot():
                    user = process.username()
                    
                    # Get CPU usage as a numeric value for proper formatting
                    cpu_percent = process.cpu_percent(interval=0.1)
                    
                    # Get memory info in MB rather than percentage
                    try:
                        memory_info = process.memory_info()
                        memory_mb = memory_info.rss / (1024 * 1024)  # Convert bytes to MB
                    except:
                        # Fallback to percentage if detailed memory info fails
                        memory_mb = psutil.virtual_memory().total * process.memory_percent() / 100 / (1024 * 1024)
                    
                    # Record these metrics in the metrics system too (important for Resource stats)
                    # This ensures the Resources section at the top of the dashboard has accurate data
                    if 'claude_metrics' in globals() and hasattr(claude_metrics, 'global_metrics'):
                        # Add to CPU and memory collections
                        claude_metrics.global_metrics.cpu_usage.append(cpu_percent)
                        claude_metrics.global_metrics.memory_usage.append(memory_mb)
                    
                    started = time.strftime("%H:%M:%S", time.localtime(process.create_time()))
                    command = " ".join(process.cmdline())[:80] + ("..." if len(" ".join(process.cmdline())) > 80 else "")
                    
                    # Calculate runtime
                    runtime_secs = int(current_time - process_info['start_time'])
                    hours, remainder = divmod(runtime_secs, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    runtime = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                    
                    active_processes.append({
                        "user": user,
                        "pid": numeric_pid,  # Use numeric PID for dashboard filtering
                        "cpu_percent": cpu_percent,  # Store numeric value
                        "memory_mb": memory_mb,  # Store in MB
                        "started": started,
                        "runtime": runtime,
                        "command": command
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError) as e:
                # Process no longer exists, can't be accessed, or invalid PID format
                logger.info(f"Error processing {pid}: {str(e)}")
                pids_to_remove.append(pid)
                
        # Clean up processes that no longer exist
        for pid in pids_to_remove:
            # First mark as finished before untracking
            try:
                proxy_launched_processes[pid]['status'] = 'finished'
                logger.info(f"Marked process {pid} as finished before cleanup")
            except:
                pass
                
            # Now untrack the process
            untrack_claude_process(pid)
            logger.info(f"Untracked Claude process with PID {pid}")
        
        return active_processes
    except Exception as e:
        logger.error(f"Error getting Claude processes: {e}")
        return []