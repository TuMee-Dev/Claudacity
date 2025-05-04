#!/usr/bin/env python3
"""
Claude-compatible Ollama API Server with cross-platform service support.

This script extends the original FastAPI application with service installation
capabilities for Windows, macOS, and Linux using the cross_service library.

Usage:
  python claude_service.py --install  # Install as a service
  python claude_service.py --start    # Start the service
  python claude_service.py --stop     # Stop the service
  python claude_service.py --status   # Check service status
  python claude_service.py --uninstall # Uninstall the service
  python claude_service.py            # Run directly (not as a service)
"""

import os
import sys
import logging
import uvicorn
from pathlib import Path

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Add the cross_service.py to the path if it's in the same directory
if os.path.exists(os.path.join(SCRIPT_DIR, "cross_service.py")):
    sys.path.insert(0, SCRIPT_DIR)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("claude_service")

def install_dependencies():
    """Install required dependencies if not already installed."""
    try:
        import fastapi
        import uvicorn
        import httpx
        # Try to import the original claude_ollama_server module if available
        try:
            import claude_ollama_server
        except ImportError:
            pass
    except ImportError:
        logger.info("Installing required dependencies...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", 
                              "fastapi", "uvicorn", "httpx"])
        logger.info("Dependencies installed successfully.")

# Try to install dependencies
install_dependencies()

# Import the cross_service module
try:
    import cross_service
except ImportError:
    logger.error("cross_service.py not found. Please make sure it's in the same directory.")
    sys.exit(1)

def run_server():
    """Run the API server as a standalone process."""
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
    # Create the service manager
    service_manager = cross_service.ServiceManager(
        service_name="claude_ollama_api",
        description="Claude-compatible Ollama API Server",
        exe_path=sys.executable,
        args=[os.path.abspath(__file__), "--run"],
        working_dir=SCRIPT_DIR,
        log_dir=os.path.join(SCRIPT_DIR, "logs")
    )
    
    # If --run is specified, run the server as a service
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        logger.info("Running as a service...")
        service_manager.run_as_service(run_server)
        return
    
    # Otherwise, handle command-line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--install":
            logger.info("Installing service...")
            service_manager.install()
            logger.info("Service installed. You can now start it with --start")
        elif sys.argv[1] == "--uninstall":
            logger.info("Uninstalling service...")
            service_manager.uninstall()
            logger.info("Service uninstalled.")
        elif sys.argv[1] == "--start":
            logger.info("Starting service...")
            service_manager.start()
            logger.info("Service started.")
        elif sys.argv[1] == "--stop":
            logger.info("Stopping service...")
            service_manager.stop()
            logger.info("Service stopped.")
        elif sys.argv[1] == "--status":
            status = service_manager.status()
            logger.info(f"Service status: {status}")
        else:
            logger.error(f"Unknown command: {sys.argv[1]}")
            print("Usage:")
            print("  --install   : Install the service")
            print("  --uninstall : Uninstall the service")
            print("  --start     : Start the service")
            print("  --stop      : Stop the service")
            print("  --status    : Check service status")
            print("  --run       : Run as a service (internal use)")
            print("  (no args)   : Run directly (not as a service)")
    else:
        # Run the server directly (not as a service)
        run_server()

if __name__ == "__main__":
    main()