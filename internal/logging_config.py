import logging
import os
import sys
import platform

def get_system_log_dir():
    """Get the appropriate system log directory based on platform."""
    app_name = "Claudacity"

    # macOS - Use ~/Library/Logs/AppName
    if platform.system() == "Darwin":
        return os.path.expanduser(f"~/Library/Logs/{app_name}")

    # Windows - Use %LOCALAPPDATA%\AppName\Logs
    elif platform.system() == "Windows":
        local_app_data = os.environ.get("LOCALAPPDATA", os.path.expanduser("~\\AppData\\Local"))
        return os.path.join(local_app_data, app_name, "Logs")

    # Linux/Unix - Use ~/.local/share/AppName/logs
    else:
        xdg_data_home = os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
        return os.path.join(xdg_data_home, app_name, "logs")

def setup_logging(debug):
    # Get system log directory
    system_log_dir = get_system_log_dir()
    os.makedirs(system_log_dir, exist_ok=True)

    log_file_path = os.path.join(system_log_dir, "server.log")
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S'))

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    logger.addHandler(file_handler)

    # Log where we're saving logs
    logger.info(f"Logs are being saved to: {system_log_dir}")

    return logger