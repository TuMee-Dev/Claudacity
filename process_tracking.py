import collections
import psutil
import re
import subprocess
import time
import threading
        

# Dictionary to track proxy-launched Claude processes
proxy_launched_processes = {}
# Dictionary to store outputs from recent processes (limited to last 20)
process_outputs = collections.OrderedDict()
MAX_STORED_OUTPUTS = 20
# Lock for process access - we need to use threading.Lock() not asyncio.Lock() at module level
process_lock = threading.Lock()  # Must use threading.Lock() not asyncio.Lock() at module level

# Initialize streaming buffer if needed
streaming_content_buffer = {}
    
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
        
        for pid, process_info in list(process_tracking.proxy_launched_processes.items()):  # Use list() to avoid dict changing during iteration
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
                process_tracking.proxy_launched_processes[pid]['status'] = 'finished'
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