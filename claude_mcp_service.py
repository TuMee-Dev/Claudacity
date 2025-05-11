#!/usr/bin/env python3
"""
Claude MCP Service with cross-platform service support.

This script allows installing and managing the Claude MCP service,
which runs "uvx mcpo --port 8100 -- claude mcp serve" as a system service.

Usage:
  python claude_mcp_service.py --install  # Install as a service
  python claude_mcp_service.py --start    # Start the service
  python claude_mcp_service.py --stop     # Stop the service
  python claude_mcp_service.py --restart  # Restart the service
  python claude_mcp_service.py --status   # Check service status
  python claude_mcp_service.py --uninstall # Uninstall the service
  python claude_mcp_service.py --workdir=PATH # Specify a working directory
  python claude_mcp_service.py            # Run directly (not as a service)
"""

import os
import sys
import logging
import subprocess
from pathlib import Path

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Add the script directory to the path if it's not already
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("claude_mcp_service")

# Import sys_log_dir from logging_config
from internal.logging_config import get_system_log_dir

# Import the helper modules
try:
    import cross_service
    from internal.service_helper import (
        parse_service_arguments, 
        setup_working_dir, 
        handle_service_commands
    )
except ImportError as e:
    logger.error(f"Required modules not found: {e}")
    logger.error("Please make sure cross_service.py and internal/service_helper.py exist.")
    sys.exit(1)

def find_uvx_executable():
    """Find the path to the uvx executable."""
    try:
        uvx_path = subprocess.check_output(["which", "uvx"], text=True).strip()
        logger.info(f"Found uvx at: {uvx_path}")
        return uvx_path
    except subprocess.CalledProcessError:
        logger.error("Could not find uvx executable. Make sure it's installed and in your PATH.")
        sys.exit(1)

def run_mcp_server():
    """Run the MCP server directly."""
    # Parse arguments and set up working directory
    args = parse_service_arguments("Claude MCP Server")
    working_dir = setup_working_dir(args)
    
    # Change to the working directory
    os.chdir(working_dir)
    logger.info(f"Changed working directory to: {working_dir}")
    
    # Find the path to uvx executable
    uvx_path = find_uvx_executable()
    
    # Set the port
    port = 8100
    
    logger.info(f"Starting Claude MCP Server on port {port}")
    
    # Run the command
    cmd = [uvx_path, "mcpo", "--port", str(port), "--", "claude", "mcp", "serve"]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run MCP server: {e}")
        sys.exit(1)

def main():
    """Main entry point."""
    # Parse arguments and set up working directory
    args = parse_service_arguments("Claude MCP Server")
    working_dir = setup_working_dir(args)
    
    # Find the path to uvx executable
    uvx_path = find_uvx_executable()
    
    # Create the service manager with the specified working directory
    # Get system log directory
    system_log_dir = get_system_log_dir()
    os.makedirs(system_log_dir, exist_ok=True)

    service_manager = cross_service.ServiceManager(
        service_name="claude_mcp_server",
        description="Claude MCP Server",
        exe_path=sys.executable,
        args=[os.path.abspath(__file__), "--run", f"--workdir={working_dir}"],
        working_dir=working_dir,
        log_dir=system_log_dir
    )
    
    # Handle service commands
    if handle_service_commands(service_manager, args, run_mcp_server):
        return

if __name__ == "__main__":
    main()