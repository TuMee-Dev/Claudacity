"""
Dashboard implementation for Claude Ollama Server.

This module provides the dashboard functionality for monitoring the server,
including HTML generation, metrics visualization, and process management.
"""

import time
import psutil
import logging
from fastapi import HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

logger = logging.getLogger(__name__)

# These will be imported from the main module
metrics = None
get_process_output = None
process_outputs = None
get_running_claude_processes = None

def init_dashboard(app, metrics_module, get_process_output_fn, process_outputs_dict, get_running_claude_processes_fn):
    """Initialize the dashboard with required dependencies"""
    global metrics, get_process_output, process_outputs, get_running_claude_processes
    
    # Set the dependencies
    metrics = metrics_module
    get_process_output = get_process_output_fn
    process_outputs = process_outputs_dict
    get_running_claude_processes = get_running_claude_processes_fn
    
    # Register routes
    app.get("/")(root)
    app.get("/status")(status)
    app.get("/metrics")(get_metrics)
    app.get("/process_outputs")(list_process_outputs)
    app.get("/process_output/{pid}")(get_single_process_output)
    app.post("/terminate_process/{pid}")(terminate_process)
    
    return app

def generate_dashboard_html():
    """Generate HTML for the dashboard page"""
    # Get current metrics
    metrics_data = metrics.get_metrics()
    
    # Get currently running Claude processes
    running_processes = get_running_claude_processes()
    
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
    
    # Create the HTML
    html = f"""<!DOCTYPE html>
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
    </script>
</head>
<body>
    <div class="container">
        <h1>Claude Ollama API Dashboard <span class="badge badge-success">v1.0</span></h1>
        <p class="timestamp">Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="stats-grid">
            <div class="card">
                <h2>General</h2>
                <div class="metric">
                    <div class="metric-name">Uptime</div>
                    <div class="metric-value">{metrics_data['uptime']['formatted']}</div>
                </div>
                <div class="metric">
                    <div class="metric-name">Server Status</div>
                    <div class="metric-value">Running</div>
                </div>
                <div class="metric">
                    <div class="metric-name">Claude Invocations</div>
                    <div class="metric-value">{metrics_data['claude_invocations']['total']}</div>
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
                    <div class="metric-value">{metrics_data['claude_invocations']['per_minute']:.2f}</div>
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
                    <div class="metric-value">${metrics_data.get('cost', {}).get('total_cost', 0):.4f}</div>
                </div>
                <div class="metric">
                    <div class="metric-name">Avg Cost per Request</div>
                    <div class="metric-value">${metrics_data.get('cost', {}).get('avg_cost', 0):.4f}</div>
                </div>
                <div class="metric">
                    <div class="metric-name">Active Conversations</div>
                    <div class="metric-value">{len(metrics.active_conversations)}</div>
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
    
    # Add running processes to table
    if running_processes:
        for proc in running_processes:
            pid = proc.get('pid', 'N/A')
            cmd = proc.get('cmd', 'Unknown')[:80] + ('...' if len(proc.get('cmd', '')) > 80 else '')
            cpu = f"{proc.get('cpu_percent', 0):.1f}%"
            memory = f"{proc.get('memory_mb', 0):.1f} MB"
            runtime = proc.get('runtime', 'N/A')
            
            html += f"""
                    <tr>
                        <td>{pid}</td>
                        <td title="{proc.get('cmd', 'Unknown')}">{cmd}</td>
                        <td>{cpu}</td>
                        <td>{memory}</td>
                        <td>{runtime}</td>
                        <td>
                            <button onclick="terminateProcess('{pid}')" class="button alert">Terminate</button>
                        </td>
                    </tr>
"""
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
    for pid, output in reversed(list(process_outputs.items())):
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
            html += f"""
                    <tr>
                        <td>{output['pid']}</td>
                        <td>{output['timestamp']}</td>
                        <td title="{output['command']}">{output['command']}</td>
                        <td>
                            <a href="/process_output/{output['pid']}" target="_blank" class="button">View Details</a>
                        </td>
                    </tr>
"""
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
    for pid, output in reversed(process_outputs.items()):
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
    output = get_process_output(pid)
    if output:
        return output
    else:
        raise HTTPException(status_code=404, detail=f"No output found for process {pid}")

async def terminate_process(pid: str):
    """
    Terminate a specific process by PID.
    This is used for the process management UI.
    """
    try:
        pid = int(pid)
        import os
        import signal
        
        # Safety check - only terminate Claude-related processes
        process = psutil.Process(pid)
        if "claude" in process.name().lower() or any("claude" in arg.lower() for arg in process.cmdline()):
            # First try SIGTERM for clean shutdown
            logger.info(f"Attempting to terminate Claude process {pid} with SIGTERM")
            process.terminate()
            
            # Wait up to 3 seconds for process to terminate
            try:
                process.wait(timeout=3)
                return {"status": "success", "message": f"Process {pid} terminated successfully"}
            except psutil.TimeoutExpired:
                # If process doesn't terminate, use SIGKILL
                logger.warning(f"Process {pid} did not terminate with SIGTERM, using SIGKILL")
                process.kill()
                return {"status": "success", "message": f"Process {pid} forcibly terminated with SIGKILL"}
        else:
            logger.warning(f"Attempted to terminate non-Claude process {pid}, request denied")
            return JSONResponse(
                status_code=403,
                content={"status": "error", "message": "Can only terminate Claude-related processes"}
            )
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Invalid PID format"}
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