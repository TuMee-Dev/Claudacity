#!/usr/bin/env python3
"""
Claude-compatible Ollama API Server with cross-platform service support.

This script extends the original FastAPI application with service installation
capabilities for Windows, macOS, and Linux using the cross_service library.

Usage:
  python claude_service.py --install  # Install as a service
  python claude_service.py --start    # Start the service
  python claude_service.py --stop     # Stop the service
  python claude_service.py --restart  # Restart the service
  python claude_service.py --status   # Check service status
  python claude_service.py --uninstall # Uninstall the service
  python claude_service.py --workdir=PATH # Specify a working directory
  python claude_service.py            # Run directly (not as a service)
"""

import os
import sys
import logging
import uvicorn
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
logger = logging.getLogger("claude_service")

# Import sys_log_dir from logging_config
from internal.logging_config import get_system_log_dir

def install_dependencies():
    """Install required dependencies if not already installed."""
    try:
        import fastapi
        import uvicorn
        import httpx
        import psutil
        # Try to import the original claude_ollama_server module if available
        try:
            import claude_ollama_server
        except ImportError:
            pass
    except ImportError:
        logger.info("Installing required dependencies...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", 
                              "fastapi", "uvicorn", "httpx", "psutil"])
        logger.info("Dependencies installed successfully.")

# Try to install dependencies
install_dependencies()

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

def run_server():
    """Run the API server as a standalone process."""
    # Parse arguments and set up working directory
    args = parse_service_arguments("Claude Ollama API Server")
    working_dir = setup_working_dir(args)
    
    # Change to the working directory
    os.chdir(working_dir)
    logger.info(f"Changed working directory to: {working_dir}")
    
    # Import the actual server code here
    server_module_path = os.path.join(SCRIPT_DIR, "claude_ollama_server.py")
    
    if not os.path.exists(server_module_path):
        logger.error(f"Server module not found at {server_module_path}")
        sys.exit(1)
    
    # Use importlib to load the module dynamically
    import importlib.util
    spec = importlib.util.spec_from_file_location("server_module", server_module_path)
    server_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server_module)
    
    # Get the host and port
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 22434))
    
    logger.info(f"Starting Claude Ollama API Server at http://{host}:{port}")
    
    # Start the server
    uvicorn.run(
        server_module.app,
        host=host,
        port=port,
        log_level="info"
    )

def main():
    """Main entry point."""
    # Parse arguments and set up working directory
    args = parse_service_arguments("Claude Ollama API Server")
    working_dir = setup_working_dir(args)
    
    # Create the service manager
    # Get system log directory
    system_log_dir = get_system_log_dir()
    os.makedirs(system_log_dir, exist_ok=True)

    service_manager = cross_service.ServiceManager(
        service_name="claude_ollama_api",
        description="Claude-compatible Ollama API Server",
        exe_path=sys.executable,
        args=[os.path.abspath(__file__), "--run", f"--workdir={working_dir}"],
        working_dir=working_dir,
        log_dir=system_log_dir
    )
    
    # Handle service commands
    if handle_service_commands(service_manager, args, run_server):
        return

if __name__ == "__main__":
    main()