#!/usr/bin/env python3
"""
Shared service helper module for Claude service implementations.

This module provides common functionality used by various Claude services
including working directory management, argument parsing, and service
management utilities.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Get the directory of the script that imports this module
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define default working directory
DEFAULT_WORKING_DIR = os.path.expanduser("~/Projects.ai")

logger = logging.getLogger("service_helper")

def ensure_dir_exists(directory):
    """Ensure the specified directory exists, creating it if necessary."""
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            logger.info(f"Created working directory: {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            sys.exit(1)
    return directory

def parse_service_arguments(description="Claude Service"):
    """Parse common service command line arguments."""
    parser = argparse.ArgumentParser(description=description)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--install", action="store_true", help="Install as a service")
    group.add_argument("--uninstall", action="store_true", help="Uninstall the service")
    group.add_argument("--start", action="store_true", help="Start the service")
    group.add_argument("--stop", action="store_true", help="Stop the service")
    group.add_argument("--restart", action="store_true", help="Restart the service")
    group.add_argument("--status", action="store_true", help="Check service status")
    group.add_argument("--run", action="store_true", help=argparse.SUPPRESS)  # Internal use only
    
    parser.add_argument("--workdir", dest="working_dir", default=DEFAULT_WORKING_DIR,
                        help=f"Set the working directory (default: {DEFAULT_WORKING_DIR})")
    
    return parser.parse_args()

def setup_working_dir(args=None):
    """Set up and validate the working directory."""
    if args is None:
        args = parse_service_arguments()

    # Expand user paths (like ~) to absolute paths
    working_dir = os.path.expanduser(args.working_dir)

    # Ensure the directory exists
    working_dir = ensure_dir_exists(working_dir)
    return working_dir

def handle_service_commands(service_manager, args=None, server_func=None):
    """Handle common service commands based on parsed arguments."""
    if args is None:
        args = parse_service_arguments()
    
    if args.run and server_func:
        # Running as a service
        logger.info(f"Running as a service in {args.working_dir}...")
        server_func()
        return True
    
    if args.install:
        logger.info("Installing service...")
        service_manager.install()
        logger.info("Service installed. You can now start it with --start")
    elif args.uninstall:
        logger.info("Uninstalling service...")
        service_manager.uninstall()
        logger.info("Service uninstalled.")
    elif args.start:
        logger.info("Starting service...")
        service_manager.start()
        logger.info("Service started.")
    elif args.stop:
        logger.info("Stopping service...")
        service_manager.stop()
        logger.info("Service stopped.")
    elif args.restart:
        logger.info("Restarting service...")
        service_manager.stop()
        logger.info("Service stopped. Waiting for full shutdown...")
        # Wait a moment to ensure the service has fully stopped
        import time
        time.sleep(2)
        service_manager.start()
        logger.info("Service restarted.")
    elif args.status:
        status = service_manager.status()
        logger.info(f"Service status: {status}")
    else:
        # No specific command, run the server directly
        if server_func:
            logger.info(f"Running server directly in {args.working_dir}...")
            server_func()
        return False
    
    return True