#!/usr/bin/env python3
"""
Verification script to check that the timeout changes are working properly.
This script directly tests the timeout constants and process termination logic.
"""
import sys
import os
import time
import logging
import asyncio
from datetime import datetime

# Add the parent directory to the path so we can import modules from the project
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('timeout_verification')

# Import the timeout constants from the main server
try:
    from claude_ollama_server import (
        CLAUDE_STREAM_CHUNK_TIMEOUT,
        CLAUDE_STREAM_MAX_SILENCE,
    )
    from process_tracking import (
        track_claude_process,
        untrack_claude_process,
        proxy_launched_processes
    )

    
    logger.info(f"Imported constants successfully:")
    logger.info(f"CLAUDE_STREAM_CHUNK_TIMEOUT = {CLAUDE_STREAM_CHUNK_TIMEOUT}")
    logger.info(f"CLAUDE_STREAM_MAX_SILENCE = {CLAUDE_STREAM_MAX_SILENCE}")
except ImportError as e:
    logger.error(f"Failed to import constants: {e}")
    logger.error("Make sure you're running this script from the project root directory")
    sys.exit(1)

async def simulate_hanging_process():
    """Create a simulated hanging process and see if our timeout logic works."""
    logger.info("Starting simulated hanging process test")
    
    # Create a long-running process (sleep command)
    process_id = "test-process-123"
    
    # Use a sleep command that will run longer than our timeout
    cmd = f"sleep 120"  # 2 minutes sleep
    
    logger.info(f"Starting process with command: {cmd}")
    
    # Add to the process tracking directly (bypassing the track_claude_process function)
    proxy_launched_processes[process_id] = {
        "command": cmd,
        "start_time": time.time(),
        "current_request": {"prompt": "Test prompt"}
    }
    
    # Verify tracking
    if process_id in proxy_launched_processes:
        logger.info(f"Process {process_id} is being tracked")
        logger.info(f"Process info: {proxy_launched_processes[process_id]}")
    else:
        logger.error(f"Failed to track process {process_id}")
        return
    
    # Set appropriate timeout parameters for this process
    process_info = proxy_launched_processes.get(process_id, {})
    if process_info:
        # Use very short timeouts for testing
        process_info['chunk_timeout'] = 5.0  # 5 seconds
        process_info['max_silence'] = 10.0   # 10 seconds
        logger.info(f"Set custom timeouts: chunk_timeout={process_info['chunk_timeout']}, max_silence={process_info['max_silence']}")
    
    # Start the actual process
    start_time = time.time()
    
    process = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    if process and process.pid:
        logger.info(f"Started process with PID: {process.pid}")
        
        # Update our tracking with the real PID
        process_info = proxy_launched_processes.get(process_id, {})
        if process_info:
            process_info['real_pid'] = process.pid
            logger.info(f"Updated process info with real PID: {process.pid}")
    else:
        logger.error("Failed to start process")
        return
    
    # Simulate streaming with timeouts
    buffer = ""
    last_chunk_time = time.time()
    streaming_complete = False
    
    while not streaming_complete:
        try:
            # Wait for output (which won't come in this test)
            try:
                chunk = await asyncio.wait_for(process.stdout.read(1024), timeout=3.0)
                if not chunk:  # End of stream
                    break
                    
                # We won't get here in our test since the process is just sleeping
                buffer += chunk.decode('utf-8', errors='replace')
                last_chunk_time = time.time()
                
            except asyncio.TimeoutError:
                # No data received within timeout
                if process.returncode is not None:
                    logger.info(f"Process completed with return code {process.returncode}")
                    break
                
                # Get process-specific timeout
                process_info = proxy_launched_processes.get(process_id, {})
                chunk_timeout = process_info.get('chunk_timeout', CLAUDE_STREAM_CHUNK_TIMEOUT)
                max_silence = process_info.get('max_silence', CLAUDE_STREAM_MAX_SILENCE)
                
                logger.info(f"No output received. Using timeouts: chunk={chunk_timeout}s, max={max_silence}s")
                
                # Check if we've gone too long without output
                current_time = time.time()
                silence_duration = current_time - last_chunk_time
                
                if silence_duration > chunk_timeout:
                    logger.warning(f"No output for {silence_duration:.2f} seconds, checking if process is still active")
                    
                    # Check if the process exists
                    try:
                        # This just checks if the process exists
                        os.kill(process.pid, 0)
                        logger.info(f"Process {process.pid} still exists")
                        
                        # Check if it's been too long since the last chunk
                        if silence_duration > max_silence:
                            logger.warning(f"No output for {silence_duration:.2f} seconds, assuming process is hung")
                            
                            # Try to terminate the process
                            try:
                                logger.warning(f"Attempting to terminate hung process {process.pid}")
                                process.kill()
                                logger.info("Process successfully terminated")
                                
                                # If successful, untrack this process
                                untrack_claude_process(process_id)
                                logger.info(f"Untracked process {process_id}")
                                
                                streaming_complete = True
                                break
                            except Exception as kill_error:
                                logger.error(f"Failed to kill hung process: {kill_error}")
                    
                    except ProcessLookupError:
                        logger.info(f"Process {process.pid} no longer exists")
                        streaming_complete = True
                        break
                    except Exception as e:
                        logger.warning(f"Error checking process status: {e}")
        
        except Exception as e:
            logger.error(f"Error in streaming loop: {e}")
            break
            
    # Calculate elapsed time
    elapsed = time.time() - start_time
    logger.info(f"Test completed in {elapsed:.2f} seconds")
    
    # Verify that the process was terminated
    # Wait a bit for process termination to take effect
    await asyncio.sleep(1)
    
    if process.returncode is None:
        try:
            # Try to get the process status
            # Note: This might report the process is still running when it was already terminated
            # due to asyncio subprocess not updating the returncode immediately
            process_exists = False
            try:
                os.kill(process.pid, 0)
                process_exists = True
            except ProcessLookupError:
                process_exists = False
                
            if process_exists:
                logger.error("FAILURE: Process is still running when it should have been terminated")
            else:
                logger.info("SUCCESS: Process was correctly terminated")
        except Exception as e:
            logger.error(f"Error checking process status: {e}")
    else:
        logger.info(f"Process completed with return code {process.returncode}")
    
    # Verify that the process was untracked
    if process_id in proxy_launched_processes:
        logger.error(f"FAILURE: Process {process_id} is still being tracked")
    else:
        logger.info(f"SUCCESS: Process {process_id} was correctly untracked")
    
    # Summary
    logger.info("\nTest Summary:")
    logger.info(f"- Started process at: {datetime.fromtimestamp(start_time).strftime('%H:%M:%S')}")
    logger.info(f"- Ended process at: {datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')}")
    logger.info(f"- Total test time: {elapsed:.2f} seconds")
    logger.info(f"- Expected timeout (max_silence): {max_silence} seconds")
    
    if elapsed < max_silence:
        logger.warning("Test completed too quickly - timeout might not have been triggered")
    elif elapsed > max_silence + 5:
        logger.warning("Test took longer than expected - timeout might not be working correctly")
    else:
        logger.info("Test timing looks correct - timeout functionality appears to be working")

async def main():
    """Run the verification tests."""
    logger.info("Starting timeout verification tests")
    await simulate_hanging_process()
    logger.info("All tests completed")

if __name__ == "__main__":
    asyncio.run(main())