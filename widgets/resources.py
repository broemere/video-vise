import sys
import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from config import APP_NAME

def resource_path(*path_parts: str) -> str:
    """
    Return the absolute path to a bundled resource.
    - In “frozen” (PyInstaller) mode, base is sys._MEIPASS.
    - Otherwise, base is the folder where this .py file lives.
    Usage examples:
        icon_path   = resource_path("icons", "app.ico")
        ffmpeg_path = resource_path("ffmpeg", "ffmpeg.exe")
    """
    if getattr(sys, "frozen", False):
        base = sys._MEIPASS
    else:
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(base, *path_parts)


def get_exe(name_base: str) -> str:
    """
    Return full path to ffmpeg/ffprobe:
      - In dev mode, looks under "<script_dir>/ffmpeg/<name>.exe"
      - In frozen mode, looks under "<_MEIPASS>/<name>.exe"
    """
    ext = ".exe" if sys.platform.startswith("win") else ""
    exe_name = name_base + ext
    if getattr(sys, "frozen", False):
        # PyInstaller will have put ffmpeg.exe and ffprobe.exe directly in _MEIPASS
        return resource_path(exe_name)
    else:
        # In dev, they live in a subfolder called “ffmpeg”
        return resource_path("ffmpeg", exe_name)


icon_path = resource_path("icons", "app.ico")
FFMPEG   = get_exe('ffmpeg')
FFPROBE  = get_exe('ffprobe')

def setup_logging():
    MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
    BACKUP_COUNT = 1                 # Keeps one file (e.g., app_debug.log.1)
    LOGNAME = f"{APP_NAME.replace(' ', '_').lower()}_debug.log"
    LOGFILE = resource_path(LOGNAME)
    print(LOGFILE)
    if getattr(sys, "frozen", False):
        LOGFILE = Path(LOGFILE).parent.parent / LOGNAME
        file_handler = RotatingFileHandler(
            str(LOGFILE),
            mode="a",  # Append mode is crucial for rotation
            maxBytes=MAX_LOG_SIZE,
            backupCount=BACKUP_COUNT,
            encoding="utf-8"
        )
        handlers = [file_handler]
    else:
        file_handler = RotatingFileHandler(
            str(LOGFILE),
            mode="a",  # Append mode
            maxBytes=MAX_LOG_SIZE,
            backupCount=BACKUP_COUNT,
            encoding="utf-8"
        )
        handlers = [file_handler, logging.StreamHandler(sys.stderr)]  # Also log to stderr in dev
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(threadName)s] %(levelname)s: %(message)s", handlers=handlers)
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {LOGFILE}")
    logger.info(f"ffmpeg: {FFMPEG}")
    return logger
