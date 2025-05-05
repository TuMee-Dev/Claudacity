#!/usr/bin/env python3
"""
cross_service.py - Cross-platform service management library

This library provides a unified API for managing services across Windows, macOS, and Linux.
It builds on Daemoniker for Windows and Linux, and adds custom macOS support via launchd.
"""

import os
import sys
import platform
import subprocess
import tempfile
import shutil
import socket
import logging
import time
from pathlib import Path
import inspect

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("cross_service")

class ServiceError(Exception):
    """Exception raised for service-related errors."""
    pass

def _is_windows():
    """Check if running on Windows."""
    return platform.system() == "Windows"

def _is_macos():
    """Check if running on macOS."""
    return platform.system() == "Darwin"

def _is_linux():
    """Check if running on Linux."""
    return platform.system() == "Linux"

def _get_default_service_name():
    """Generate a default service name based on the script name."""
    frame = inspect.stack()[2]
    module = inspect.getmodule(frame[0])
    if module and module.__file__:
        service_name = os.path.basename(module.__file__).split('.')[0]
        if _is_macos():
            return f"com.user.{service_name}"
        else:
            return service_name
    return "python_service"

def _ensure_daemoniker():
    """Ensure Daemoniker is installed."""
    try:
        import daemoniker
        return daemoniker
    except ImportError:
        logger.info("Installing Daemoniker...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "daemoniker"])
            import daemoniker
            return daemoniker
        except Exception as e:
            raise ServiceError(f"Failed to install Daemoniker: {e}")

class MacOSService:
    """macOS service implementation using launchd."""

    def __init__(self, service_name=None, description=None, exe_path=None, args=None, 
                 working_dir=None, pid_file=None, log_dir=None):
        """
        Initialize a macOS service.
        
        Args:
            service_name: Name of the service (should be in reverse domain notation)
            description: Description of the service
            exe_path: Path to the executable
            args: List of arguments to pass to the executable
            working_dir: Working directory for the service
            pid_file: Path to the PID file
            log_dir: Directory for log files
        """
        self.service_name = service_name or _get_default_service_name()
        if not self.service_name.startswith("com."):
            self.service_name = f"com.user.{self.service_name}"
            
        self.description = description or f"Python service: {self.service_name}"
        self.exe_path = exe_path or sys.executable
        self.args = args or [os.path.abspath(sys.argv[0])]
        
        # Ensure args is a list
        if isinstance(self.args, str):
            self.args = [self.args]
        
        self.working_dir = working_dir or os.getcwd()
        
        # Setup directories
        self.user_home = os.path.expanduser("~")
        self.agent_dir = os.path.join(self.user_home, "Library", "LaunchAgents")
        self.plist_path = os.path.join(self.agent_dir, f"{self.service_name}.plist")
        
        # Set log directory
        if log_dir:
            self.log_dir = log_dir
        else:
            self.log_dir = os.path.join(self.user_home, "Library", "Logs", self.service_name)
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set log file paths
        self.stdout_log = os.path.join(self.log_dir, "stdout.log")
        self.stderr_log = os.path.join(self.log_dir, "stderr.log")
        
        # Set PID file path
        self.pid_file = pid_file or os.path.join(tempfile.gettempdir(), f"{self.service_name}.pid")

    def _create_plist(self):
        """Create a launchd property list file for the service."""
        # Create the LaunchAgents directory if it doesn't exist
        os.makedirs(self.agent_dir, exist_ok=True)
        
        # Construct the program arguments
        program_args = [f"<string>{self.exe_path}</string>"]
        for arg in self.args:
            program_args.append(f"<string>{arg}</string>")
        
        # Get user's PATH from shell
        try:
            shell_path = subprocess.check_output(['bash', '-c', 'echo $PATH'], text=True).strip()
            logger.info(f"Using user PATH: {shell_path}")
        except:
            shell_path = os.environ.get('PATH', '/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin')
            logger.info(f"Using fallback PATH: {shell_path}")
        
        plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{self.service_name}</string>
    <key>ProgramArguments</key>
    <array>
        {os.linesep.join(program_args)}
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>WorkingDirectory</key>
    <string>{self.working_dir}</string>
    <key>StandardOutPath</key>
    <string>{self.stdout_log}</string>
    <key>StandardErrorPath</key>
    <string>{self.stderr_log}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>{shell_path}</string>
        <key>HOME</key>
        <string>{self.user_home}</string>
    </dict>
</dict>
</plist>
"""
        
        with open(self.plist_path, "w") as f:
            f.write(plist_content)
        
        logger.info(f"Created launchd plist at: {self.plist_path}")
        return self.plist_path

    def install(self):
        """Install the service."""
        try:
            # Create the plist file
            self._create_plist()
            
            # Load the service
            subprocess.run(["launchctl", "load", self.plist_path], check=True)
            logger.info(f"Service '{self.service_name}' installed and loaded successfully!")
            logger.info(f"Logs are available at: {self.log_dir}")
            return True
        except subprocess.CalledProcessError as e:
            raise ServiceError(f"Failed to install service: {e}")

    def uninstall(self):
        """Uninstall the service."""
        try:
            # Check if plist file exists
            if not os.path.exists(self.plist_path):
                logger.warning(f"Service '{self.service_name}' is not installed (plist not found).")
                return False
            
            # Unload the service first
            try:
                subprocess.run(["launchctl", "unload", self.plist_path], check=True)
                logger.info(f"Service '{self.service_name}' unloaded.")
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to unload the service (it might not be running): {e}")
            
            # Remove the plist file
            os.remove(self.plist_path)
            logger.info(f"Service plist removed: {self.plist_path}")
            return True
        except Exception as e:
            raise ServiceError(f"Failed to uninstall service: {e}")

    def start(self):
        """Start the service."""
        try:
            subprocess.run(["launchctl", "start", self.service_name], check=True)
            logger.info(f"Service '{self.service_name}' started.")
            return True
        except subprocess.CalledProcessError as e:
            raise ServiceError(f"Failed to start service: {e}")

    def stop(self):
        """Stop the service."""
        try:
            subprocess.run(["launchctl", "stop", self.service_name], check=True)
            logger.info(f"Service '{self.service_name}' stopped.")
            return True
        except subprocess.CalledProcessError as e:
            raise ServiceError(f"Failed to stop service: {e}")

    def restart(self):
        """Restart the service."""
        self.stop()
        # Give it a moment to fully stop
        time.sleep(1)
        self.start()
        return True

    def status(self):
        """Check if the service is running."""
        try:
            result = subprocess.run(
                ["launchctl", "list"], 
                check=True, 
                capture_output=True, 
                text=True
            )
            
            # Check if the service is in the list
            for line in result.stdout.splitlines():
                if self.service_name in line:
                    parts = line.split()
                    if len(parts) >= 3:
                        pid = parts[0]
                        status = parts[1]
                        if pid != "-":
                            logger.info(f"Service '{self.service_name}' is running with PID {pid}")
                            return "RUNNING"
                        else:
                            logger.info(f"Service '{self.service_name}' is loaded but not running (status: {status})")
                            return "STOPPED"
            
            # Service not in the list
            logger.info(f"Service '{self.service_name}' is not loaded.")
            return "NOT_INSTALLED"
            
        except subprocess.CalledProcessError as e:
            raise ServiceError(f"Failed to check service status: {e}")


class WindowsService:
    """Windows service implementation using Daemoniker."""

    def __init__(self, service_name=None, description=None, exe_path=None, args=None, 
                 working_dir=None, pid_file=None, log_dir=None):
        """
        Initialize a Windows service using Daemoniker.
        
        Args:
            service_name: Name of the service
            description: Description of the service
            exe_path: Path to the executable
            args: List of arguments to pass to the executable
            working_dir: Working directory for the service
            pid_file: Path to the PID file
            log_dir: Directory for log files
        """
        self.daemoniker = _ensure_daemoniker()
        
        self.service_name = service_name or _get_default_service_name()
        self.description = description or f"Python service: {self.service_name}"
        self.exe_path = exe_path or sys.executable
        self.args = args or [os.path.abspath(sys.argv[0])]
        
        # Ensure args is a list
        if isinstance(self.args, str):
            self.args = [self.args]
        
        self.working_dir = working_dir or os.getcwd()
        
        # Set log directory
        if log_dir:
            self.log_dir = log_dir
        else:
            appdata = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
            self.log_dir = os.path.join(appdata, "Logs", self.service_name)
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set log file paths
        self.stdout_log = os.path.join(self.log_dir, "stdout.log")
        self.stderr_log = os.path.join(self.log_dir, "stderr.log")
        
        # Set PID file path
        self.pid_file = pid_file or os.path.join(tempfile.gettempdir(), f"{self.service_name}.pid")

    def install(self):
        """Install the service."""
        # For Daemoniker on Windows, we use pywin32 through a batch file
        try:
            # First, check if pywin32 is installed
            try:
                import win32serviceutil
                import win32service
            except ImportError:
                logger.info("Installing pywin32...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pywin32"])
            
            # Create a batch file that will run the script
            batch_dir = os.path.join(os.environ.get("TEMP", os.path.expanduser("~")), "service_scripts")
            os.makedirs(batch_dir, exist_ok=True)
            
            batch_path = os.path.join(batch_dir, f"{self.service_name}.bat")
            
            # Create batch file content
            batch_content = f"""@echo off
cd /d "{self.working_dir}"
"{self.exe_path}" {' '.join(self.args)} > "{self.stdout_log}" 2> "{self.stderr_log}"
"""
            
            with open(batch_path, "w") as f:
                f.write(batch_content)
            
            # Now use nssm or sc to create the service
            try:
                # Try with nssm first
                subprocess.run(["nssm", "install", self.service_name, batch_path], check=True)
                subprocess.run(["nssm", "set", self.service_name, "Description", self.description], check=True)
                logger.info(f"Service '{self.service_name}' installed with NSSM.")
                return True
            except (FileNotFoundError, subprocess.CalledProcessError):
                # If nssm fails, try with sc
                logger.info("NSSM not found, trying with sc command...")
                bin_path = f'"{batch_path}"'
                subprocess.run(["sc", "create", self.service_name, "binPath=", bin_path, 
                                "start=", "auto", "DisplayName=", self.service_name], check=True)
                subprocess.run(["sc", "description", self.service_name, self.description], check=True)
                logger.info(f"Service '{self.service_name}' installed with SC.")
                return True
                
        except Exception as e:
            raise ServiceError(f"Failed to install service: {e}")

    def uninstall(self):
        """Uninstall the service."""
        try:
            try:
                # Try with nssm first
                subprocess.run(["nssm", "stop", self.service_name], check=True)
                subprocess.run(["nssm", "remove", self.service_name, "confirm"], check=True)
                logger.info(f"Service '{self.service_name}' uninstalled with NSSM.")
                return True
            except (FileNotFoundError, subprocess.CalledProcessError):
                # If nssm fails, try with sc
                logger.info("NSSM not found, trying with sc command...")
                subprocess.run(["sc", "stop", self.service_name], check=True)
                subprocess.run(["sc", "delete", self.service_name], check=True)
                logger.info(f"Service '{self.service_name}' uninstalled with SC.")
                return True
                
        except Exception as e:
            raise ServiceError(f"Failed to uninstall service: {e}")

    def start(self):
        """Start the service."""
        try:
            try:
                # Try with nssm first
                subprocess.run(["nssm", "start", self.service_name], check=True)
                logger.info(f"Service '{self.service_name}' started with NSSM.")
            except (FileNotFoundError, subprocess.CalledProcessError):
                # If nssm fails, try with sc
                subprocess.run(["sc", "start", self.service_name], check=True)
                logger.info(f"Service '{self.service_name}' started with SC.")
            return True
        except Exception as e:
            raise ServiceError(f"Failed to start service: {e}")

    def stop(self):
        """Stop the service."""
        try:
            try:
                # Try with nssm first
                subprocess.run(["nssm", "stop", self.service_name], check=True)
                logger.info(f"Service '{self.service_name}' stopped with NSSM.")
            except (FileNotFoundError, subprocess.CalledProcessError):
                # If nssm fails, try with sc
                subprocess.run(["sc", "stop", self.service_name], check=True)
                logger.info(f"Service '{self.service_name}' stopped with SC.")
            return True
        except Exception as e:
            raise ServiceError(f"Failed to stop service: {e}")

    def restart(self):
        """Restart the service."""
        self.stop()
        # Give it a moment to fully stop
        time.sleep(1)
        self.start()
        return True

    def status(self):
        """Check if the service is running."""
        try:
            try:
                # Try with nssm first
                result = subprocess.run(["nssm", "status", self.service_name], 
                                        check=False, capture_output=True, text=True)
                if "SERVICE_RUNNING" in result.stdout:
                    logger.info(f"Service '{self.service_name}' is running.")
                    return "RUNNING"
                elif "SERVICE_STOPPED" in result.stdout:
                    logger.info(f"Service '{self.service_name}' is stopped.")
                    return "STOPPED"
                else:
                    logger.info(f"Service '{self.service_name}' status unknown: {result.stdout}")
                    return "UNKNOWN"
            except FileNotFoundError:
                # If nssm fails, try with sc
                result = subprocess.run(["sc", "query", self.service_name], 
                                        check=False, capture_output=True, text=True)
                if "RUNNING" in result.stdout:
                    logger.info(f"Service '{self.service_name}' is running.")
                    return "RUNNING"
                elif "STOPPED" in result.stdout:
                    logger.info(f"Service '{self.service_name}' is stopped.")
                    return "STOPPED"
                else:
                    logger.info(f"Service '{self.service_name}' status unknown: {result.stdout}")
                    return "UNKNOWN"
        except Exception as e:
            raise ServiceError(f"Failed to check service status: {e}")


class LinuxService:
    """Linux service implementation using Daemoniker or systemd."""

    def __init__(self, service_name=None, description=None, exe_path=None, args=None, 
                 working_dir=None, pid_file=None, log_dir=None, use_systemd=True):
        """
        Initialize a Linux service.
        
        Args:
            service_name: Name of the service
            description: Description of the service
            exe_path: Path to the executable
            args: List of arguments to pass to the executable
            working_dir: Working directory for the service
            pid_file: Path to the PID file
            log_dir: Directory for log files
            use_systemd: Whether to use systemd or Daemoniker
        """
        self.daemoniker = _ensure_daemoniker()
        
        self.service_name = service_name or _get_default_service_name()
        # Ensure service name is valid for systemd
        self.service_name = self.service_name.replace(".", "-")
        
        self.description = description or f"Python service: {self.service_name}"
        self.exe_path = exe_path or sys.executable
        self.args = args or [os.path.abspath(sys.argv[0])]
        
        # Ensure args is a list
        if isinstance(self.args, str):
            self.args = [self.args]
        
        self.working_dir = working_dir or os.getcwd()
        self.use_systemd = use_systemd
        
        # Set log directory
        if log_dir:
            self.log_dir = log_dir
        else:
            self.log_dir = os.path.join("/var/log", self.service_name)
            if not os.access("/var/log", os.W_OK):
                # Use user home if /var/log is not writable
                self.log_dir = os.path.join(os.path.expanduser("~"), ".log", self.service_name)
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set log file paths
        self.stdout_log = os.path.join(self.log_dir, "stdout.log")
        self.stderr_log = os.path.join(self.log_dir, "stderr.log")
        
        # Set PID file path
        if pid_file:
            self.pid_file = pid_file
        else:
            # Check if we can write to /var/run
            if os.access("/var/run", os.W_OK):
                self.pid_file = f"/var/run/{self.service_name}.pid"
            else:
                self.pid_file = os.path.join(tempfile.gettempdir(), f"{self.service_name}.pid")
        
        # Systemd service file path
        self.service_file = f"/etc/systemd/system/{self.service_name}.service"
        self.user_service_dir = os.path.join(os.path.expanduser("~"), ".config/systemd/user")

    def _create_systemd_service_file(self, user_mode=False):
        """Create a systemd service file."""
        cmd = f"{self.exe_path} {' '.join(self.args)}"
        
        service_content = f"""[Unit]
Description={self.description}
After=network.target

[Service]
Type=simple
ExecStart={cmd}
WorkingDirectory={self.working_dir}
StandardOutput=append:{self.stdout_log}
StandardError=append:{self.stderr_log}
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
"""
        
        if user_mode:
            # Create user systemd directory if it doesn't exist
            os.makedirs(self.user_service_dir, exist_ok=True)
            service_file = os.path.join(self.user_service_dir, f"{self.service_name}.service")
            
            with open(service_file, "w") as f:
                f.write(service_content)
            
            return service_file
        else:
            # Create a temporary file first
            temp_service_file = os.path.join(tempfile.gettempdir(), f"{self.service_name}.service")
            
            with open(temp_service_file, "w") as f:
                f.write(service_content)
            
            return temp_service_file

    def install(self):
        """Install the service."""
        if self.use_systemd:
            try:
                # Check if we have root access
                has_root = os.geteuid() == 0 if hasattr(os, "geteuid") else False
                
                if has_root:
                    # System-wide installation
                    temp_service_file = self._create_systemd_service_file(user_mode=False)
                    shutil.copy(temp_service_file, self.service_file)
                    os.chmod(self.service_file, 0o644)
                    
                    # Reload systemd daemon
                    subprocess.run(["systemctl", "daemon-reload"], check=True)
                    
                    # Enable the service
                    subprocess.run(["systemctl", "enable", self.service_name], check=True)
                    
                    logger.info(f"Service '{self.service_name}' installed system-wide with systemd.")
                else:
                    # User-mode installation
                    service_file = self._create_systemd_service_file(user_mode=True)
                    
                    # Reload user systemd daemon
                    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
                    
                    # Enable the service
                    subprocess.run(["systemctl", "--user", "enable", self.service_name], check=True)
                    
                    logger.info(f"Service '{self.service_name}' installed for user with systemd.")
                
                return True
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to install with systemd: {e}")
                logger.info("Falling back to Daemoniker...")
                self.use_systemd = False
        
        if not self.use_systemd:
            # Use Daemoniker
            logger.info(f"Service '{self.service_name}' installed with Daemoniker.")
            logger.info(f"PID file: {self.pid_file}")
            logger.info(f"To start the service, run: python {' '.join(self.args)} --run")
            return True

    def uninstall(self):
        """Uninstall the service."""
        if self.use_systemd:
            try:
                # Check if we have root access
                has_root = os.geteuid() == 0 if hasattr(os, "geteuid") else False
                
                if has_root:
                    # System-wide uninstallation
                    
                    # Stop the service if running
                    subprocess.run(["systemctl", "stop", self.service_name], check=False)
                    
                    # Disable the service
                    subprocess.run(["systemctl", "disable", self.service_name], check=True)
                    
                    # Remove the service file
                    if os.path.exists(self.service_file):
                        os.remove(self.service_file)
                    
                    # Reload systemd daemon
                    subprocess.run(["systemctl", "daemon-reload"], check=True)
                    
                    logger.info(f"Service '{self.service_name}' uninstalled system-wide with systemd.")
                else:
                    # User-mode uninstallation
                    user_service_file = os.path.join(self.user_service_dir, f"{self.service_name}.service")
                    
                    # Stop the service if running
                    subprocess.run(["systemctl", "--user", "stop", self.service_name], check=False)
                    
                    # Disable the service
                    subprocess.run(["systemctl", "--user", "disable", self.service_name], check=True)
                    
                    # Remove the service file
                    if os.path.exists(user_service_file):
                        os.remove(user_service_file)
                    
                    # Reload user systemd daemon
                    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
                    
                    logger.info(f"Service '{self.service_name}' uninstalled for user with systemd.")
                
                return True
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to uninstall with systemd: {e}")
                logger.info("Falling back to Daemoniker...")
                self.use_systemd = False
        
        if not self.use_systemd:
            # Use Daemoniker
            if os.path.exists(self.pid_file):
                os.remove(self.pid_file)
                logger.info(f"Removed PID file: {self.pid_file}")
            
            logger.info(f"Service '{self.service_name}' uninstalled with Daemoniker.")
            return True

    def start(self):
        """Start the service."""
        if self.use_systemd:
            try:
                # Check if we have root access
                has_root = os.geteuid() == 0 if hasattr(os, "geteuid") else False
                
                if has_root:
                    # System-wide start
                    subprocess.run(["systemctl", "start", self.service_name], check=True)
                    logger.info(f"Service '{self.service_name}' started system-wide with systemd.")
                else:
                    # User-mode start
                    subprocess.run(["systemctl", "--user", "start", self.service_name], check=True)
                    logger.info(f"Service '{self.service_name}' started for user with systemd.")
                
                return True
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to start with systemd: {e}")
                logger.info("Falling back to Daemoniker...")
                self.use_systemd = False
        
        if not self.use_systemd:
            # Use Daemoniker
            logger.info(f"Starting service '{self.service_name}' with Daemoniker...")
            logger.info(f"Run python {' '.join(self.args)} --run to start the service.")
            return True

    def stop(self):
        """Stop the service."""
        if self.use_systemd:
            try:
                # Check if we have root access
                has_root = os.geteuid() == 0 if hasattr(os, "geteuid") else False
                
                if has_root:
                    # System-wide stop
                    subprocess.run(["systemctl", "stop", self.service_name], check=True)
                    logger.info(f"Service '{self.service_name}' stopped system-wide with systemd.")
                else:
                    # User-mode stop
                    subprocess.run(["systemctl", "--user", "stop", self.service_name], check=True)
                    logger.info(f"Service '{self.service_name}' stopped for user with systemd.")
                
                return True
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to stop with systemd: {e}")
                logger.info("Falling back to Daemoniker...")
                self.use_systemd = False
        
        if not self.use_systemd:
            # Use Daemoniker
            if os.path.exists(self.pid_file):
                try:
                    # Read PID from file
                    with open(self.pid_file, "r") as f:
                        pid = int(f.read().strip())
                    
                    # Send SIGTERM to the process
                    os.kill(pid, 15)  # 15 is SIGTERM
                    logger.info(f"Sent SIGTERM to process {pid}")
                    
                    # Wait for process to exit
                    for _ in range(10):
                        try:
                            os.kill(pid, 0)  # Check if process exists
                            time.sleep(0.5)
                        except OSError:
                            break
                    
                    logger.info(f"Service '{self.service_name}' stopped with Daemoniker.")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to stop service: {e}")
            
            logger.warning(f"PID file not found: {self.pid_file}")
            return False

    def restart(self):
        """Restart the service."""
        if self.use_systemd:
            try:
                # Check if we have root access
                has_root = os.geteuid() == 0 if hasattr(os, "geteuid") else False
                
                if has_root:
                    # System-wide restart
                    subprocess.run(["systemctl", "restart", self.service_name], check=True)
                    logger.info(f"Service '{self.service_name}' restarted system-wide with systemd.")
                else:
                    # User-mode restart
                    subprocess.run(["systemctl", "--user", "restart", self.service_name], check=True)
                    logger.info(f"Service '{self.service_name}' restarted for user with systemd.")
                
                return True
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to restart with systemd: {e}")
                logger.info("Falling back to Daemoniker...")
                self.use_systemd = False
        
        if not self.use_systemd:
            # Use Daemoniker
            self.stop()
            time.sleep(1)
            return self.start()

    def status(self):
        """Check if the service is running."""
        if self.use_systemd:
            try:
                # Check if we have root access
                has_root = os.geteuid() == 0 if hasattr(os, "geteuid") else False
                
                if has_root:
                    # System-wide status
                    result = subprocess.run(["systemctl", "is-active", self.service_name], 
                                          check=False, capture_output=True, text=True)
                else:
                    # User-mode status
                    result = subprocess.run(["systemctl", "--user", "is-active", self.service_name], 
                                          check=False, capture_output=True, text=True)
                
                status = result.stdout.strip()
                
                if status == "active":
                    logger.info(f"Service '{self.service_name}' is running.")
                    return "RUNNING"
                else:
                    logger.info(f"Service '{self.service_name}' is not running (status: {status}).")
                    return "STOPPED"
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to check status with systemd: {e}")
                logger.info("Falling back to Daemoniker...")
                self.use_systemd = False
        
        if not self.use_systemd:
            # Use Daemoniker
            if os.path.exists(self.pid_file):
                try:
                    # Read PID from file
                    with open(self.pid_file, "r") as f:
                        pid = int(f.read().strip())
                    
                    # Check if process exists
                    try:
                        os.kill(pid, 0)
                        logger.info(f"Service '{self.service_name}' is running with PID {pid}.")
                        return "RUNNING"
                    except OSError:
                        logger.info(f"Service '{self.service_name}' is not running (process {pid} not found).")
                        return "STOPPED"
                except Exception as e:
                    logger.warning(f"Failed to check service status: {e}")
            
            logger.info(f"Service '{self.service_name}' is not running (PID file not found).")
            return "STOPPED"


class ServiceManager:
    """
    Unified cross-platform service management.
    
    This class provides a consistent API for managing services across
    Windows, macOS, and Linux platforms.
    """
    
    def __init__(self, service_name=None, description=None, exe_path=None, args=None, 
                 working_dir=None, pid_file=None, log_dir=None):
        """
        Initialize the service manager.
        
        Args:
            service_name: Name of the service
            description: Description of the service
            exe_path: Path to the executable
            args: List of arguments to pass to the executable
            working_dir: Working directory for the service
            pid_file: Path to the PID file
            log_dir: Directory for log files
        """
        self.service_name = service_name or _get_default_service_name()
        self.description = description or f"Python service: {self.service_name}"
        self.exe_path = exe_path or sys.executable
        self.args = args or [os.path.abspath(sys.argv[0])]
        self.working_dir = working_dir or os.getcwd()
        self.pid_file = pid_file
        self.log_dir = log_dir
        
        # Create platform-specific service implementation
        if _is_windows():
            self.service = WindowsService(
                service_name=self.service_name,
                description=self.description,
                exe_path=self.exe_path,
                args=self.args,
                working_dir=self.working_dir,
                pid_file=self.pid_file,
                log_dir=self.log_dir
            )
        elif _is_macos():
            self.service = MacOSService(
                service_name=self.service_name,
                description=self.description,
                exe_path=self.exe_path,
                args=self.args,
                working_dir=self.working_dir,
                pid_file=self.pid_file,
                log_dir=self.log_dir
            )
        elif _is_linux():
            self.service = LinuxService(
                service_name=self.service_name,
                description=self.description,
                exe_path=self.exe_path,
                args=self.args,
                working_dir=self.working_dir,
                pid_file=self.pid_file,
                log_dir=self.log_dir
            )
        else:
            raise ServiceError(f"Unsupported platform: {platform.system()}")
    
    def install(self):
        """Install the service."""
        return self.service.install()
    
    def uninstall(self):
        """Uninstall the service."""
        return self.service.uninstall()
    
    def start(self):
        """Start the service."""
        return self.service.start()
    
    def stop(self):
        """Stop the service."""
        return self.service.stop()
    
    def restart(self):
        """Restart the service."""
        return self.service.restart()
    
    def status(self):
        """Check if the service is running."""
        return self.service.status()
    
    def run_as_service(self, main_func):
        """
        Run a function as a service.
        
        Args:
            main_func: The function to run as a service
        """
        if _is_windows() or _is_linux():
            # Use Daemoniker for Windows and Linux
            daemoniker = _ensure_daemoniker()
            
            # Check for command-line arguments
            if len(sys.argv) > 1:
                if sys.argv[1] == "--run":
                    # Run the service
                    with daemoniker.Daemonizer() as (is_setup, daemonizer):
                        if is_setup:
                            # This code runs before daemonization
                            pass
                        
                        # Daemonize the process
                        is_parent, *args = daemonizer(
                            self.pid_file
                        )
                        
                        if is_parent:
                            # This code runs in the parent process
                            logger.info(f"Service started with PID file at {self.pid_file}")
                            return
                        
                        # This code runs in the daemon process
                        try:
                            main_func()
                        except Exception as e:
                            logger.error(f"Error in service: {e}")
                            sys.exit(1)
        
        elif _is_macos():
            # For macOS, we'll just run the function directly
            # as launchd handles the daemonization
            try:
                main_func()
            except Exception as e:
                logger.error(f"Error in service: {e}")
                sys.exit(1)


def handle_command_line(service_manager):
    """
    Handle command-line arguments for service management.
    
    Args:
        service_manager: The ServiceManager instance
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Service Management")
    
    # Add service management commands
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--install", action="store_true", help="Install the service")
    group.add_argument("--uninstall", action="store_true", help="Uninstall the service")
    group.add_argument("--start", action="store_true", help="Start the service")
    group.add_argument("--stop", action="store_true", help="Stop the service")
    group.add_argument("--restart", action="store_true", help="Restart the service")
    group.add_argument("--status", action="store_true", help="Check service status")
    group.add_argument("--run", action="store_true", help="Run the service")
    
    args = parser.parse_args()
    
    # Execute the command
    try:
        if args.install:
            service_manager.install()
        elif args.uninstall:
            service_manager.uninstall()
        elif args.start:
            service_manager.start()
        elif args.stop:
            service_manager.stop()
        elif args.restart:
            service_manager.restart()
        elif args.status:
            status = service_manager.status()
            print(f"Service status: {status}")
        elif args.run:
            # This is handled by run_as_service()
            pass
    except ServiceError as e:
        logger.error(f"Service error: {e}")
        sys.exit(1)


# Example usage
if __name__ == "__main__":
    # Example service implementation
    def service_main():
        """Main service function."""
        print("Service started")
        
        # Keep the service running
        while True:
            print("Service is running...")
            time.sleep(10)
    
    # Create a service manager
    service_manager = ServiceManager(
        service_name="example_service",
        description="Example Python Service",
        log_dir="logs"
    )
    
    # Run the service if --run is specified, otherwise handle other commands
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        service_manager.run_as_service(service_main)
    else:
        handle_command_line(service_manager)