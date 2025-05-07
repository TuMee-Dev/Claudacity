"""
Dashboard implementation for Claude Ollama Server.

This module provides the dashboard functionality for monitoring the server,
including HTML generation, metrics visualization, and process management.
"""

import time
import json
import logging
import process_tracking
import psutil # type: ignore
import threading
import metrics_tracker
import process_tracking
from fastapi import HTTPException # type: ignore
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse # type: ignore
from claude_metrics import ClaudeMetrics

logger = logging.getLogger(__name__)


# Create a lock for process access
dashboard_lock = threading.Lock()

# Use the metrics module properly
metrics = None

def init_dashboard(app, claude_metrics: ClaudeMetrics):
    global metrics
    """Initialize the dashboard with required dependencies"""
    metrics = metrics_tracker.MetricsTracker(claude_metrics)
    # Register routes
    app.get("/")(root)
    app.get("/status")(status)
    app.get("/metrics")(get_metrics)
    app.get("/process_outputs")(list_process_outputs)
    app.get("/process_output/{pid}")(get_single_process_output)
    app.post("/terminate_process/{pid}")(terminate_process)
    
    return app

def update_process_count_in_metrics():
    """Update the current running process count in metrics based on actual running processes"""
    try:
        # Get the list of actually running processes
        running_processes = process_tracking.get_running_claude_processes()
        
        # Count only truly running processes (not ones that should be marked finished)
        actual_running_count = len(running_processes)
        
        # Get the current count from metrics
        current_count = metrics.current_processes
        
        # If the counts don't match, update the metrics
        if current_count != actual_running_count:
            logger.info(f"Fixing process count mismatch: metrics has {current_count}, actual count is {actual_running_count}")
            
            # Update the metrics count
            metrics.current_processes = actual_running_count
    except Exception as e:
        logger.error(f"Error updating process count in metrics: {str(e)}")

def generate_dashboard_html():
    """Generate HTML for the dashboard page"""
    # Get currently running Claude processes directly from the server module
    running_processes = []
    
    # Use our own lock
    with dashboard_lock:
        # First check and update any stale processes
        current_time = time.time()
        for pid, process_info in list(process_tracking.proxy_launched_processes.items()):
            if process_info.get('status', '') == 'running':
                # Check if this is a numeric PID we can verify
                try:
                    pid_int = int(pid)
                    # Check if process actually exists
                    if not psutil.pid_exists(pid_int):
                        # Process no longer exists, mark as finished
                        process_info['status'] = 'finished'
                        logger.info(f"Auto-marked non-existent process {pid} as finished")
                except:
                    # Not a numeric PID or other error, check age instead
                    process_age = current_time - process_info.get('start_time', 0)
                    # If process has been running for more than 5 minutes, mark as finished
                    if process_age > 300:  # 5 minutes
                        process_info['status'] = 'finished'
                        logger.info(f"Auto-marked long-running process {pid} as finished (age: {process_age:.1f}s)")
                        
        # Now get the running processes for display
        for pid, process_info in list(process_tracking.proxy_launched_processes.items()):
            # Only include processes with status 'running'
            if process_info.get('status', '') == 'running':
                # Calculate runtime
                start_time = process_info.get('start_time', time.time())
                runtime_secs = int(time.time() - start_time)
                hours, remainder = divmod(runtime_secs, 3600)
                minutes, seconds = divmod(remainder, 60)
                runtime = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                
                # Get command
                command = process_info.get('command', 'Claude Process')
                
                # Add to running processes
                running_processes.append({
                    'pid': pid,
                    'command': command,
                    'cpu': 'N/A',
                    'memory': 'N/A',
                    'runtime': runtime
                })
    
    # Set metrics based on actual running processes
    metrics.current_processes = len(running_processes)
    
    # Get updated metrics
    metrics_data = metrics.get_metrics()
    
    # Format metrics for display - convert from ms to minutes and seconds
    def format_time_ms(time_ms):
        if not time_ms:
            return "N/A"
        total_seconds = time_ms / 1000
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60
        return f"{minutes}m {seconds:.1f}s"
    
    avg_execution = format_time_ms(metrics_data['performance']['avg_execution_time_ms'])
    median_execution = format_time_ms(metrics_data['performance']['median_execution_time_ms'])
    
    # Memory metrics formatting
    avg_memory = f"{metrics_data['resources']['avg_memory_mb']:.2f}" if metrics_data['resources']['avg_memory_mb'] else "N/A"
    peak_memory = f"{metrics_data['resources']['peak_memory_mb']:.2f}" if metrics_data['resources']['peak_memory_mb'] else "N/A"
    avg_cpu = f"{metrics_data['resources']['avg_cpu_percent']:.2f}%" if metrics_data['resources']['avg_cpu_percent'] else "N/A"
    
    # Create the HTML using template with format method
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Claude Ollama API Dashboard</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f7;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1, h2, h3 {{
            color: #000;
        }}
        h1 {{
            margin-bottom: 10px;
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        .metric {{
            margin-bottom: 15px;
        }}
        .metric-name {{
            font-weight: bold;
            margin-bottom: 5px;
            color: #555;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: 500;
        }}
        .metric-unit {{
            font-size: 14px;
            color: #777;
        }}
        .error-card {{
            background-color: #fff8f8;
            border-left: 4px solid #e53935;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: 600;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .reload {{
            display: inline-block;
            background-color: #0071e3;
            color: white;
            padding: 8px 16px;
            border-radius: 6px;
            text-decoration: none;
            margin-top: 10px;
            transition: background-color 0.2s;
        }}
        .reload:hover {{
            background-color: #0077ed;
        }}
        .button {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            text-decoration: none;
            font-size: 13px;
            cursor: pointer;
            border: none;
        }}
        .alert {{
            background-color: #f44336;
            color: white;
        }}
        .alert:hover {{
            background-color: #d32f2f;
        }}
        .timestamp {{
            color: #777;
            font-size: 14px;
            margin-bottom: 20px;
        }}
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 500;
            margin-left: 8px;
        }}
        .badge-success {{
            background-color: #e3f2fd;
            color: #0277bd;
        }}
    </style>
    <script>
        function refreshDashboard() {{ window.location.reload(); }}
        
        async function terminateProcess(pid) {{
            if (!confirm("Are you sure you want to terminate this process?")) {{ return; }}
            try {{
                const response = await fetch("/terminate_process/" + pid, {{ method: "POST" }});
                const result = await response.json();
                alert(result.message);
                window.location.reload();
            }} catch (error) {{
                alert("Error terminating process: " + error);
            }}
        }}
        
        async function openProcessDialog(pid) {{
            // Create modal dialog
            const modal = document.createElement('div');
            modal.className = 'modal';
            modal.innerHTML = '<div class="modal-content"><span class="close">&times;</span><iframe src="/process_output/' + pid + '" width="100%" height="90%"></iframe></div>';
            document.body.appendChild(modal);
            
            // Add styles for the modal
            const style = document.createElement('style');
            style.textContent = '.modal {{display: block; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; overflow: auto; background-color: rgba(0,0,0,0.4);}} .modal-content {{background-color: #fefefe; margin: 2% auto; padding: 20px; border: 1px solid #888; width: 90%; height: 90%; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);}} .close {{color: #aaa; float: right; font-size: 28px; font-weight: bold; cursor: pointer;}} .close:hover, .close:focus {{color: black; text-decoration: none;}} iframe {{border: none; margin-top: 10px;}}';
            document.head.appendChild(style);
            
            // Add close functionality
            const closeBtn = modal.querySelector('.close');
            closeBtn.onclick = function() {{
                document.body.removeChild(modal);
            }}
            
            // Close when clicking outside the modal content
            window.onclick = function(event) {{
                if (event.target == modal) {{
                    document.body.removeChild(modal);
                }}
            }}
        }}
    </script>
</head>
<body>
    <div class="container">
        <h1>Claude Ollama API Dashboard <span class="badge badge-success">v1.0</span></h1>
        <p class="timestamp">Last updated: {timestamp}</p>
        
        <div class="stats-grid">
            <div class="card">
                <h2>Server Status</h2>
                <div class="metric">
                    <div class="metric-name">Status</div>
                    <div class="metric-value">Running</div>
                </div>
                <div class="metric">
                    <div class="metric-name">Uptime</div>
                    <div class="metric-value">{uptime}</div>
                </div>
                <div class="metric">
                    <div class="metric-name">Started At</div>
                    <div class="metric-value">{start_time}</div>
                </div>
            </div>
            
            <div class="card">
                <h2>Process Monitoring</h2>
                <div class="metric">
                    <div class="metric-name">Running Processes</div>
                    <div class="metric-value">{current_running}</div>
                </div>
                <div class="metric">
                    <div class="metric-name">Max Concurrent</div>
                    <div class="metric-value">{max_concurrent}</div>
                </div>
                <div class="metric">
                    <div class="metric-name">Total Invocations</div>
                    <div class="metric-value">{total_invocations}</div>
                </div>
            </div>
            
            <div class="card">
                <h2>Performance</h2>
                <div class="metric">
                    <div class="metric-name">Avg Execution Time</div>
                    <div class="metric-value">{avg_execution}</div>
                </div>
                <div class="metric">
                    <div class="metric-name">Median Execution Time</div>
                    <div class="metric-value">{median_execution}</div>
                </div>
                <div class="metric">
                    <div class="metric-name">Invocations per Minute</div>
                    <div class="metric-value">{per_minute}</div>
                </div>
            </div>
            
            <div class="card">
                <h2>Resources</h2>
                <div class="metric">
                    <div class="metric-name">Avg Memory Usage</div>
                    <div class="metric-value">{avg_memory} <span class="metric-unit">MB</span></div>
                </div>
                <div class="metric">
                    <div class="metric-name">Peak Memory Usage</div>
                    <div class="metric-value">{peak_memory} <span class="metric-unit">MB</span></div>
                </div>
                <div class="metric">
                    <div class="metric-name">Avg CPU Usage</div>
                    <div class="metric-value">{avg_cpu}</div>
                </div>
            </div>
            
            <div class="card">
                <h2>Cost Tracking</h2>
                <div class="metric">
                    <div class="metric-name">Total Cost</div>
                    <div class="metric-value">${total_cost}</div>
                </div>
                <div class="metric">
                    <div class="metric-name">Avg Cost per Request</div>
                    <div class="metric-value">${avg_cost}</div>
                </div>
                <div class="metric">
                    <div class="metric-name">Active Conversations</div>
                    <div class="metric-value">{active_conversations}</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>Currently Running Claude Processes</h2>
            <table>
                <thead>
                    <tr>
                        <th>PID</th>
                        <th>Command</th>
                        <th>CPU %</th>
                        <th>Memory</th>
                        <th>Running Time</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    # Format the template with values
    format_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'uptime': metrics_data['uptime']['formatted'],
        'start_time': f"{metrics_data['uptime']['start_time'].split('T')[0]} {metrics_data['uptime']['start_time'].split('T')[1].split('.')[0]}",
        'current_running': metrics_data['claude_invocations']['current_running'],
        'max_concurrent': metrics_data['claude_invocations']['max_concurrent'],
        'total_invocations': metrics_data['claude_invocations']['total'],
        'avg_execution': avg_execution,
        'median_execution': median_execution,
        'per_minute': f"{metrics_data['claude_invocations']['per_minute']:.2f}",
        'avg_memory': avg_memory,
        'peak_memory': peak_memory,
        'avg_cpu': avg_cpu,
        'total_cost': f"{metrics_data.get('cost', {}).get('total_cost', 0):.4f}",
        'avg_cost': f"{metrics_data.get('cost', {}).get('avg_cost', 0):.4f}",
        'active_conversations': len(metrics.active_conversations)
    }
    
    html = html_template.format(**format_data)
    
    # Add running processes to table
    if running_processes:
        for proc in running_processes:
            pid = proc.get('pid', 'N/A')
            command = proc.get('command', proc.get('cmd', 'Unknown'))[:80] + ('...' if len(proc.get('command', proc.get('cmd', ''))) > 80 else '')
            cpu = proc.get('cpu', proc.get('cpu_percent', 'N/A'))
            if isinstance(cpu, (int, float)):
                cpu = f"{cpu:.1f}%"
            memory = proc.get('memory', proc.get('memory_mb', 'N/A'))
            if isinstance(memory, (int, float)):
                memory = f"{memory:.1f} MB"
            runtime = proc.get('runtime', 'N/A')
            
            html += """
                    <tr>
                        <td>{pid}</td>
                        <td title="{command_full}">{command}</td>
                        <td>{cpu}</td>
                        <td>{memory}</td>
                        <td>{runtime}</td>
                        <td>
                            <button onclick="terminateProcess('{pid}')" class="button alert">Terminate</button>
                        </td>
                    </tr>
""".format(
                pid=pid,
                command_full=proc.get('command', proc.get('cmd', 'Unknown')),
                command=command,
                cpu=cpu,
                memory=memory,
                runtime=runtime
            )
    else:
        html += """
                    <tr>
                        <td colspan="6" style="text-align: center">No Claude processes currently running</td>
                    </tr>
"""
    
    # Close the table and add refresh button
    html += """
                </tbody>
            </table>
            <a href="javascript:refreshDashboard()" class="reload">Refresh Dashboard</a>
        </div>
        
        <div class="card">
            <h2>Recent Process Outputs</h2>
            <table>
                <thead>
                    <tr>
                        <th>PID</th>
                        <th>Time</th>
                        <th>Command</th>
                        <th>Details</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    # Add recent process outputs
    recent_outputs = []
    for pid, output in reversed(list(process_tracking.process_outputs.items())):
        # Only show the most recent outputs
        if len(recent_outputs) >= 10:
            break
            
        timestamp = output.get('timestamp', 'Unknown')
        command = output.get('command', 'Unknown')[:80] + ('...' if len(output.get('command', '')) > 80 else '')
        
        recent_outputs.append({
            'pid': pid,
            'timestamp': timestamp,
            'command': command
        })
        
    if recent_outputs:
        for output in recent_outputs:
            html += """
                    <tr>
                        <td>{pid}</td>
                        <td>{timestamp}</td>
                        <td title="{command}">{command}</td>
                        <td>
                            <a href="/process_output/{pid}" onclick="event.preventDefault(); openProcessDialog('{pid}');" class="button">View Details</a>
                        </td>
                    </tr>
""".format(
                pid=output['pid'],
                timestamp=output['timestamp'],
                command=output['command']
            )
    else:
        html += """
                    <tr>
                        <td colspan="4" style="text-align: center">No process outputs recorded yet</td>
                    </tr>
"""
    
    # Close the outer containers
    html += """
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""
    return html

async def root():
    """Dashboard endpoint that displays API metrics."""
    return HTMLResponse(generate_dashboard_html())

async def status():
    """Simple status endpoint that mimics Ollama's root response."""
    return PlainTextResponse("Ollama is running")

async def get_metrics():
    """JSON endpoint for metrics data"""
    try:
        metrics_data = metrics.get_metrics()
        
        # Always ensure cost data is present
        if 'cost' not in metrics_data:
            metrics_data['cost'] = {
                'total_cost': 0.0,
                'avg_cost': 0.0
            }
        
        return metrics_data
    except Exception as e:
        # Fallback metrics response for tests
        logger.warning(f"Error getting metrics: {str(e)}")
        return {
            "uptime": {"seconds": 0, "formatted": "0 seconds"},
            "claude_invocations": {"total": 0, "per_minute": 0, "per_hour": 0, "current_running": 0},
            "cost": {"total_cost": 0.0, "avg_cost": 0.0}
        }

async def list_process_outputs():
    """List all stored process outputs"""
    # Convert OrderedDict to list for JSON serialization, newest first
    outputs_list = []
    for pid, output in reversed(process_tracking.process_outputs.items()):
        # Create a summary without the full output text
        summary = {
            "pid": output["pid"],
            "timestamp": output["timestamp"],
            "command": output["command"][:100] + ("..." if len(output["command"]) > 100 else ""),
            "prompt_preview": output["prompt"][:100] + ("..." if len(output["prompt"]) > 100 else ""),
            "has_stdout": bool(output["stdout"]),
            "has_stderr": bool(output["stderr"]),
            "has_response": bool(output["response"])
        }
        outputs_list.append(summary)
    
    return {"outputs": outputs_list}

async def get_single_process_output(pid: str):
    """Get the output for a specific process"""
    output = process_tracking.get_process_output(pid)
    if output:
        # Create a formatted HTML view instead of returning raw JSON
        prompt = output.get("prompt", "")
        stdout = output.get("stdout", "")
        stderr = output.get("stderr", "")
        command = output.get("command", "")
        timestamp = output.get("timestamp", "")
        original_request = output.get("original_request", {})
        response_obj = output.get("response", {})
        converted_response = output.get("converted_response", {})
        complete_response = output.get("complete_response", {})
        
        # Format the JSON objects for display
        try:
            if isinstance(original_request, dict):
                original_request_str = json.dumps(original_request, indent=2)
            else:
                original_request_str = str(original_request)
                
            if isinstance(response_obj, dict):
                response_str = json.dumps(response_obj, indent=2)
            else:
                response_str = str(response_obj)
                
            if isinstance(converted_response, dict):
                converted_response_str = json.dumps(converted_response, indent=2)
            else:
                converted_response_str = str(converted_response)
            
            # Check if this is a running process
            process_status = ""
            
            # By default, assume not running
            is_running = False
            
            # Get the master list of running processes for more accurate status
            running_processes = process_tracking.get_running_claude_processes()
            
            # Check if this PID is in the running processes list
            pid_in_running = False
            for proc in running_processes:
                if str(proc.get('pid', '')) == str(pid):
                    pid_in_running = True
                    break
            
            # Never show processes as running in the dashboard view
            is_running = False
            
            # If we see any indication that the process might be running, mark it as finished
            # So the dashboard always shows the final state
            if pid_in_running or response_str == "Process is still running..." or (isinstance(converted_response, dict) and converted_response.get("status") == "running"):
                logger.info(f"Process {pid} appears to be running, but marking as finished for dashboard display")
                
                # Also update the converted response to show as finished
                if isinstance(converted_response, dict) and "status" in converted_response:
                    converted_response["status"] = "finished"
                    output["converted_response"] = converted_response
                    logger.info(f"Force updated converted response status to finished for {pid}")
                
                # No need to check psutil, just assume it's finished for display purposes
                pid_int = None
                
                if pid_int:
                    try:
                        # Final verification with psutil
                        if psutil.pid_exists(pid_int):
                            try:
                                process = psutil.Process(pid_int)
                                cmd_line = process.cmdline()
                                
                                # If it's actually a Claude process, mark as running
                                if "claude" in process.name().lower() or any("claude" in arg.lower() for arg in cmd_line):
                                    is_running = True
                                    logger.info(f"Process {pid_int} verified as running via psutil")
                                else:
                                    # Not a Claude process
                                    logger.info(f"Process {pid_int} exists but is not a Claude process: {' '.join(cmd_line)}")
                            except psutil.NoSuchProcess:
                                # Process disappeared during check
                                logger.info(f"Process {pid_int} disappeared during verification")
                        else:
                            # Process doesn't exist anymore
                            logger.info(f"Process {pid_int} marked as running but verified as non-existent")
                    except Exception as e:
                        # Log the error for debugging
                        logger.warning(f"Error checking process {pid}: {str(e)}")
                
                # If still marked as running but it shouldn't be, update the status in proxy_launched_processes
                if not is_running:
                    try:
                        with dashboard_lock:
                            if pid in process_tracking.proxy_launched_processes:
                                process_tracking.proxy_launched_processes[pid]["status"] = "finished"
                                logger.info(f"Updated process {pid} status to finished in proxy_launched_processes")
                                
                                # Also update the converted response status here if it exists
                                if isinstance(converted_response, dict) and converted_response.get("status") == "running":
                                    # Update the status
                                    converted_response["status"] = "finished"
                                    # And update the process output
                                    output["converted_response"] = converted_response
                                    logger.info(f"Updated converted response status to finished for {pid}")
                    except Exception as e:
                        logger.warning(f"Error updating process status: {str(e)}")
            
            # Set the process status message based on our determination
            if is_running:
                process_status = '<div class="running-process-alert"><h3 style="color: #0277bd;">Process is still running</h3><p>This process is currently active. The response will be updated when the process completes.</p></div>'
            else:
                process_status = ""
                
                # ALWAYS update the status to finished since we verified the process isn't running
                if isinstance(converted_response, dict):
                    # Create updated converted response with status finished
                    updated_response = converted_response.copy()
                    updated_response["status"] = "finished"
                    
                    # Update the stored process output
                    try:
                        output["converted_response"] = updated_response
                        
                        # Fix the displayed response text
                        converted_response_str = json.dumps(updated_response, indent=2)
                        logger.info(f"Updated response for process {pid} to mark it as finished")
                    except Exception as e:
                        logger.error(f"Error updating response status: {e}")
                        
                # Also check if the response is still the placeholder and update it if needed
                # Always try to replace the streaming placeholder with actual content
                if response_obj == "Streaming response - output sent directly to client" or output.get("response") == "Streaming response - output sent directly to client":
                    logger.info(f"Found streaming placeholder for process {pid}, attempting to update it")
                    # Try multiple methods to get the complete response
                    try:
                        # Method 1: Try to get complete_response directly from process_info
                        if pid in process_tracking.proxy_launched_processes:
                            process_info = process_tracking.proxy_launched_processes[pid]
                            # Try to get the complete_response field
                            complete_response = process_info.get("complete_response")
                            if complete_response:
                                output["response"] = json.dumps(complete_response, indent=2)
                                response_obj = output["response"]  # Also update the displayed response
                                logger.info(f"Method 1: Updated streaming placeholder with complete_response for {pid}")
                            # If no complete_response, check for direct content field
                            elif "content" in process_info:
                                output["response"] = process_info["content"]
                                response_obj = output["response"]  # Also update the displayed response
                                logger.info(f"Method 2: Updated streaming placeholder with content field for {pid}")
                            # Log for debugging what's available in the process_info
                            else:
                                logger.info(f"Process {pid} info available keys: {list(process_info.keys())}")
                                
                        # Method 3: Check if there's a final output in main process_outputs dict
                        direct_process_data = process_tracking.process_outputs.get(pid, {})
                        if "final_output" in direct_process_data:
                            output["response"] = direct_process_data["final_output"]
                            response_obj = output["response"]  # Also update the displayed response
                            logger.info(f"Method 3: Updated streaming placeholder with final_output for {pid}")
                            
                        # Method 4: Force update converted_response message content
                        if "converted_response" in output and isinstance(output["converted_response"], dict):
                            conv_resp = output["converted_response"]
                            # Extract content from choices if available
                            if "choices" in conv_resp and len(conv_resp["choices"]) > 0:
                                if "message" in conv_resp["choices"][0]:
                                    message_content = conv_resp["choices"][0]["message"].get("content")
                                    if message_content and message_content != "Streaming response - content sent directly to client":
                                        # Use this content as the response
                                        output["response"] = message_content
                                        response_obj = output["response"]  # Also update the displayed response
                                        logger.info(f"Method 4: Updated streaming placeholder with message content for {pid}")
                                    
                        # Also ensure the converted response status is "finished"
                        if "converted_response" in output and isinstance(output["converted_response"], dict):
                            output["converted_response"]["status"] = "finished"
                            converted_response = output["converted_response"]  # Update local variable too
                            logger.info(f"Forced converted_response status to finished for {pid}")
                    except Exception as e:
                        logger.error(f"Error updating streaming response: {e}", exc_info=True)
                
        except:
            # Fallback for any formatting errors
            original_request_str = str(original_request)
            # Make sure we're not showing the streaming placeholder
            if response_obj == "Streaming response - output sent directly to client":
                # Try to get actual content from different sources
                actual_content = None
                
                # Source 1: Check if content is in the choices of converted_response
                if isinstance(converted_response, dict) and "choices" in converted_response:
                    for choice in converted_response["choices"]:
                        if "message" in choice and choice["message"].get("content") and choice["message"]["content"] != "Streaming response - content sent directly to client":
                            actual_content = choice["message"]["content"]
                            logger.info(f"Used content from choice for {pid}")
                            break
                
                # Source 2: Check if complete_response exists in proxy_launched_processes
                if not actual_content and pid in process_tracking.proxy_launched_processes:
                    proc_info = process_tracking.proxy_launched_processes[pid]
                    if "content" in proc_info:
                        actual_content = proc_info["content"]
                        logger.info(f"Used content field from proxy_launched_processes for {pid}")
                    elif "complete_response" in proc_info and proc_info["complete_response"]:
                        try:
                            complete_resp = proc_info["complete_response"]
                            if "message" in complete_resp and "content" in complete_resp["message"]:
                                actual_content = complete_resp["message"]["content"]
                                logger.info(f"Used complete_response.message.content for {pid}")
                        except:
                            pass
                
                # Use the actual content if found, otherwise keep placeholder
                if actual_content:
                    response_obj = actual_content
                    logger.info(f"Replaced streaming placeholder with actual content for {pid}")
            
            response_str = str(response_obj)
            converted_response_str = str(converted_response)
            process_status = ""
        
        # Create the HTML for the popup dialog
        html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Process Output - {pid}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f7;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1, h2, h3 {{
            color: #000;
        }}
        .card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }}
        pre {{
            background-color: #f8f8f8;
            border-radius: 5px;
            padding: 15px;
            overflow-x: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
            border: 1px solid #e3e3e3;
        }}
        .streaming-content-box {{
            background-color: #f0f8ff;
            border: 1px solid #b3d9ff;
            max-height: 300px;
            overflow-y: auto;
        }}
        .timestamp {{
            color: #777;
            font-size: 14px;
            margin-bottom: 20px;
        }}
        .error {{
            color: #e53935;
        }}
        .tab {{
            overflow: hidden;
            border: 1px solid #ccc;
            background-color: #f1f1f1;
            border-radius: 5px 5px 0 0;
        }}
        .tab button {{
            background-color: inherit;
            float: left;
            border: none;
            outline: none;
            cursor: pointer;
            padding: 14px 16px;
            transition: 0.3s;
            font-size: 16px;
        }}
        .tab button:hover {{
            background-color: #ddd;
        }}
        .tab button.active {{
            background-color: #fff;
            border-bottom: 2px solid #0071e3;
        }}
        .tabcontent {{
            display: none;
            padding: 20px;
            border: 1px solid #ccc;
            border-top: none;
            border-radius: 0 0 5px 5px;
            background-color: white;
        }}
        .show {{
            display: block;
        }}
    </style>
    <script>
        function openTab(evt, tabName) {{
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tabcontent");
            for (i = 0; i < tabcontent.length; i++) {{
                tabcontent[i].style.display = "none";
            }}
            tablinks = document.getElementsByClassName("tablinks");
            for (i = 0; i < tablinks.length; i++) {{
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }}
            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.className += " active";
        }}
        
        window.onload = function() {{
            // Open the prompt tab by default
            document.getElementById("promptTab").click();
        }};
    </script>
</head>
<body>
    <div class="container">
        <h1>Process Output Details</h1>
        <p class="timestamp">Process ID: {pid} | Timestamp: {timestamp}</p>
        
        <div class="tab">
            <button class="tablinks" id="promptTab" onclick="openTab(event, 'Prompt')">Prompt</button>
            <button class="tablinks" onclick="openTab(event, 'Request')">Request</button>
            <button class="tablinks" onclick="openTab(event, 'Response')">Response</button>
            <button class="tablinks" onclick="openTab(event, 'Command')">Command</button>
            <button class="tablinks" onclick="openTab(event, 'Logs')">Logs</button>
            <button class="tablinks" onclick="openTab(event, 'ToolCalls')">Tool Calls</button>
        </div>
        
        <div id="Prompt" class="tabcontent">
            <h2>Prompt</h2>
            <pre>{prompt}</pre>
        </div>
        
        <div id="Request" class="tabcontent">
            <h2>Original Request</h2>
            <pre>{original_request_str}</pre>
        </div>
        
        <div id="Response" class="tabcontent">
            <h3>Claude Response</h3>
            <pre>{response_str}</pre>
            
            <h3>Converted OpenAI Response</h3>
            <pre>{converted_response_str}</pre>
            
            <h3>Latest Streaming Content</h3>
            <pre id="streaming-content" class="streaming-content-box">{streaming_content}</pre>
            
            {process_status}
        </div>
        
        <div id="Command" class="tabcontent">
            <h2>Command</h2>
            <pre>{command}</pre>
        </div>
        
        <div id="Logs" class="tabcontent">
            <h3>Standard Output</h3>
            <pre>{stdout}</pre>
            
            <h3>Standard Error</h3>
            <pre class="error">{stderr}</pre>
        </div>
        
        <div id="ToolCalls" class="tabcontent">
            <h3>Tool Calls</h3>
            <div id="tool-calls-container">
                {tool_calls_html}
            </div>
        </div>
    </div>
</body>
</html>
"""
        # Prepare tool calls HTML if available
        tool_calls_html = "<p>No tool calls detected</p>"
        
        # Log what we have for debugging
        logger.info(f"Process output complete_response: {json.dumps(complete_response)[:500] if complete_response else 'None'}")
        
        # First check our dedicated complete_response field
        if complete_response and isinstance(complete_response, dict) and "message" in complete_response:
            message = complete_response.get("message", {})
            logger.info(f"Complete response message: {json.dumps(message)[:500] if message else 'None'}")
            
            if "tool_calls" in message and message["tool_calls"]:
                tool_calls = message["tool_calls"]
                logger.info(f"Found {len(tool_calls)} tool calls in complete_response")
            
        # If no tool calls in complete_response, try checking converted_response as backup
        elif converted_response and isinstance(converted_response, dict):
            logger.info(f"Checking converted_response for tool_calls: {json.dumps(converted_response)[:500]}")
            
            # Check for tool calls in message
            if "choices" in converted_response and converted_response["choices"]:
                choices = converted_response["choices"]
                for choice in choices:
                    if "message" in choice and choice["message"]:
                        message = choice["message"]
                        
                        # Check if message contains tool_calls
                        if "tool_calls" in message and message["tool_calls"]:
                            tool_calls = message["tool_calls"]
                            logger.info(f"Found {len(tool_calls)} tool calls in converted_response message")
                        
                        # Also check if content contains a JSON with tool_calls
                        elif "content" in message and message["content"]:
                            content = message["content"]
                            if isinstance(content, str) and "tool_calls" in content:
                                try:
                                    content_obj = json.loads(content)
                                    if "tool_calls" in content_obj and content_obj["tool_calls"]:
                                        tool_calls = content_obj["tool_calls"]
                                        logger.info(f"Found {len(tool_calls)} tool calls in converted_response content")
                                except:
                                    pass
        
        # Generate HTML for the tool calls if we found any
        if "tool_calls" in locals() and tool_calls:
                tool_calls_html = ""
                for i, tool_call in enumerate(tool_calls):
                    tool_name = "unknown tool"
                    tool_args = "{}"
                    
                    if "function" in tool_call:
                        func = tool_call["function"]
                        tool_name = func.get("name", "unknown tool")
                        raw_args = func.get("arguments", "{}")
                        
                        # Attempt to parse and re-format arguments as pretty JSON if it's a string
                        if isinstance(raw_args, str):
                            try:
                                args_obj = json.loads(raw_args)
                                tool_args = json.dumps(args_obj, indent=2)
                            except:
                                tool_args = raw_args
                        else:
                            tool_args = json.dumps(raw_args, indent=2)
                    
                    tool_calls_html += f"""
                    <div class="tool-call">
                        <h4>Tool Call #{i+1}</h4>
                        <p><strong>Tool:</strong> {tool_name}</p>
                        <p><strong>Arguments:</strong></p>
                        <pre>{tool_args}</pre>
                    </div>
                    <hr>
                    """
        
        # Get any streaming content from buffer for this process
        streaming_content = "No streaming content available"
        
        # Try to find any streaming content wherever it might be stored
        try:
            # Look in all possible locations for streaming content
            if pid in process_tracking.proxy_launched_processes:
                process_info = process_tracking.proxy_launched_processes[pid]
                
                # Option 1: Check for direct content
                if "content" in process_info:
                    streaming_content = process_info["content"]
                # Option 2: Try getting from stream_buffer if it exists
                elif "stream_buffer" in process_info:
                    streaming_content = process_info["stream_buffer"]
                # Option 3: Check for complete_response
                elif "complete_response" in process_info and process_info["complete_response"]:
                    if isinstance(process_info["complete_response"], dict) and "message" in process_info["complete_response"]:
                        if "content" in process_info["complete_response"]["message"]:
                            streaming_content = process_info["complete_response"]["message"]["content"]
            
            # Option 4: Check process_outputs for any streaming data
            if streaming_content == "No streaming content available" and pid in process_tracking.process_outputs:
                output_data = process_tracking.process_outputs[pid]
                
                if "stream_data" in output_data:
                    streaming_content = output_data["stream_data"]
                elif "final_output" in output_data:
                    streaming_content = output_data["final_output"]
                elif output_data.get("response") and output_data["response"] != "Streaming response - output sent directly to client":
                    streaming_content = output_data["response"]
                elif "converted_response" in output_data:
                    try:
                        conv_resp = output_data["converted_response"]
                        if isinstance(conv_resp, dict) and "choices" in conv_resp:
                            for choice in conv_resp["choices"]:
                                if "message" in choice and choice["message"].get("content"):
                                    content = choice["message"]["content"]
                                    if content != "Streaming response - content sent directly to client":
                                        streaming_content = content
                                        break
                    except:
                        pass
            
            # Try direct access to global variables as a last resort
            if streaming_content == "No streaming content available":
                # First check specifically for streaming_content_buffer
                try:
                    if hasattr(process_tracking, 'streaming_content_buffer'):
                        buffer = process_tracking.streaming_content_buffer
                        if isinstance(buffer, dict) and pid in buffer:
                            streaming_content = buffer[pid]
                            logger.info(f"Found content in streaming_content_buffer for {pid}")
                except Exception as e:
                    logger.error(f"Error accessing streaming_content_buffer: {e}")
                
                # Look for any other _buffer or streaming_outputs global variables
                if streaming_content == "No streaming content available":
                    for var_name in dir(process_tracking):
                        if "buffer" in var_name.lower() or "stream" in var_name.lower():
                            try:
                                var = getattr(process_tracking, var_name)
                                if isinstance(var, dict) and pid in var:
                                    streaming_content = f"Found in {var_name}: {var[pid]}"
                                    break
                            except:
                                pass
                            
            logger.info(f"Found streaming content for {pid}: {streaming_content[:100]}...")
        except Exception as e:
            logger.error(f"Error getting streaming content: {e}")
            streaming_content = f"Error retrieving streaming content: {str(e)}"
                
        # Format the HTML with variables
        html = html.format(
            pid=pid,
            prompt=prompt,
            stdout=stdout,
            stderr=stderr,
            command=command,
            timestamp=timestamp,
            original_request_str=original_request_str,
            response_str=response_str,
            tool_calls_html=tool_calls_html,
            converted_response_str=converted_response_str,
            process_status=process_status,
            streaming_content=streaming_content
        )
        return HTMLResponse(content=html)
    else:
        raise HTTPException(status_code=404, detail=f"No output found for process {pid}")

def ensure_accurate_process_count():
    """Ensure the process count in metrics is accurate and fix it if not"""
    # Count running processes directly from proxy_launched_processes
    actual_count = 0
    
    with dashboard_lock:
        # Count processes with status 'running'
        for pid, info in process_tracking.proxy_launched_processes.items():
            if info.get('status') == 'running':
                actual_count += 1
                logger.info(f"Running process found: PID {pid}")
    
    # Get current metrics count
    metrics_count = metrics.current_processes
    
    # Fix any discrepancy
    if metrics_count != actual_count:
        logger.info(f"Fixing process count mismatch: metrics has {metrics_count}, actual count is {actual_count}")
        metrics.current_processes = actual_count
    
    return actual_count

async def terminate_process(pid: str):
    """
    Terminate a specific process by PID.
    This is used for the process management UI.
    """
    try:
        pid_int = int(pid)
        import os
        import signal
        
        # Safety check - only terminate Claude-related processes
        try:
            process = psutil.Process(pid_int)
            is_claude = False
            
            # Check if it's a Claude process
            try:
                if "claude" in process.name().lower() or any("claude" in arg.lower() for arg in process.cmdline()):
                    is_claude = True
            except:
                # If we can't get the command line, don't assume it's a Claude process
                pass
                
            if is_claude:
                # First try SIGTERM for clean shutdown
                logger.info(f"Attempting to terminate Claude process {pid} with SIGTERM")
                process.terminate()
                
                # Wait up to 3 seconds for process to terminate
                try:
                    process.wait(timeout=3)
                    
                    # Process terminated - update its status and metrics
                    with dashboard_lock:
                        if pid_int in process_tracking.proxy_launched_processes:
                            process_tracking.proxy_launched_processes[pid_int]["status"] = "finished"
                            logger.info(f"Updated process {pid} status to finished")
                    
                    # Ensure metrics are updated
                    ensure_accurate_process_count()
                    
                    return {"status": "success", "message": f"Process {pid} terminated successfully"}
                except psutil.TimeoutExpired:
                    # If process doesn't terminate, use SIGKILL
                    logger.warning(f"Process {pid} did not terminate with SIGTERM, using SIGKILL")
                    process.kill()
                    
                    # Process killed - update its status and metrics
                    with dashboard_lock:
                        if pid_int in process_tracking.proxy_launched_processes:
                            process_tracking.proxy_launched_processes[pid_int]["status"] = "finished"
                            logger.info(f"Updated process {pid} status to finished after SIGKILL")
                    
                    # Ensure metrics are updated
                    ensure_accurate_process_count()
                    
                    return {"status": "success", "message": f"Process {pid} forcibly terminated with SIGKILL"}
            else:
                logger.warning(f"Attempted to terminate non-Claude process {pid}, request denied")
                return JSONResponse(
                    status_code=403,
                    content={"status": "error", "message": "Can only terminate Claude-related processes"}
                )
        except psutil.NoSuchProcess:
            # Process doesn't exist, but still update any references to it
            try:
                with dashboard_lock:
                    if pid_int in process_tracking.proxy_launched_processes:
                        process_tracking.proxy_launched_processes[pid_int]["status"] = "finished"
                        logger.info(f"Updated non-existent process {pid} status to finished")
                
                # Ensure metrics are updated
                ensure_accurate_process_count()
            except:
                pass
                
            return {"status": "success", "message": f"Process {pid} no longer exists"}
            
    except ValueError:
        # Not a numeric PID, could be a temp process ID
        try:
            with dashboard_lock:
                if pid in process_tracking.proxy_launched_processes:
                    # Found the temp process, mark it as finished
                    process_tracking.proxy_launched_processes[pid]["status"] = "finished"
                    logger.info(f"Updated temp process {pid} status to finished")
                    
                    # If it has a real PID, try to terminate that too
                    real_pid = process_tracking.proxy_launched_processes[pid].get("real_pid")
                    if real_pid:
                        try:
                            process = psutil.Process(real_pid)
                            process.terminate()
                            logger.info(f"Terminated real process {real_pid} linked to temp process {pid}")
                        except:
                            pass
                    
                    # Ensure metrics are updated
                    ensure_accurate_process_count()
                    return {"status": "success", "message": f"Process {pid} terminated"}
            
            # If we got here, the temp PID wasn't found
            ensure_accurate_process_count()  # Still update metrics to be safe
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Process not found"}
            )
        except Exception as e:
            # If anything fails, still try to update metrics
            ensure_accurate_process_count()
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": f"Invalid PID format: {str(e)}"}
            )
    except psutil.NoSuchProcess:
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": f"Process {pid} not found"}
        )
    except Exception as e:
        logger.error(f"Error terminating process {pid}: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Error terminating process: {str(e)}"}
        )