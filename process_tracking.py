import collections
import datetime
import psutil # type: ignore
import re
import os
import subprocess
import time
import threading
import logging
import formatters

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

def track_claude_process(pid, command, original_request=None):
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
                        
                        active_processes.append({
                            "user": "claude",
                            "pid": pid,
                            "cpu": "N/A",
                            "memory": "N/A",
                            "started": time.strftime("%H:%M:%S", time.localtime(process_info['start_time'])),
                            "runtime": runtime,
                            "command": process_info.get('command', 'Claude Process')[:80]
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
                    cpu = f"{process.cpu_percent(interval=0.1):.1f}"
                    mem = f"{process.memory_percent():.1f}"
                    started = time.strftime("%H:%M:%S", time.localtime(process.create_time()))
                    command = " ".join(process.cmdline())[:80] + ("..." if len(" ".join(process.cmdline())) > 80 else "")
                    
                    # Calculate runtime
                    runtime_secs = int(current_time - process_info['start_time'])
                    hours, remainder = divmod(runtime_secs, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    runtime = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                    
                    active_processes.append({
                        "user": user,
                        "pid": pid,
                        "cpu": cpu,
                        "memory": mem,
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