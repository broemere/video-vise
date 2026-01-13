import datetime
import json
import logging
import os
import re
import subprocess
import sys
import threading
import traceback
from fractions import Fraction
from logging.handlers import RotatingFileHandler
from pathlib import Path, PurePath
from statistics import median
import urllib.request

# Third-Party Imports
from PySide6.QtCore import QSettings, Qt, QThread, Signal, QTimer
from PySide6.QtGui import QBrush, QColor, QIcon, QPalette, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QDialog, QFileDialog, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar,
    QPushButton, QSizePolicy, QSplashScreen, QStyle, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget
)
from tifffile import TiffFile, imread
from widgets.file_scanning import find_file_on_network_drives, find_original_file
from widgets.resources import setup_logging, icon_path, FFMPEG, FFPROBE
from widgets.inspecting import get_video_info
from config import *


log = logging.getLogger(__name__)


instructions = (
    "Use this tool to reduce the size of AVI/TIF files and save storage space.<br><br>"

    "<b>Usage Instructions:</b><br>"
    "1. Compress large AVI/TIF files for archiving on server<br>"
    "2. Uncompress files on your computer for use with ImageJ<br>"

    "<b>File Functions:</b><br>"
    "<i>Compress:</i> Compress video file.<br>"
    "<i>Validate:</i> Compare AVI and MKV files to verify no data loss.<br>"
    "<i>Uncompress:</i> Convert compressed video back to AVI.<br><br>"

    "• Can run directly on server folders/files, but will be slower.<br>"
    "• Compressed videos cannot be opened directly in ImageJ.<br>"
    "• VLC can play the MKV files: <a href='https://www.videolan.org/vlc/'>VLC Video Player</a><br>"
    "• This app will never delete data, it will only create new files.<br>"
    "• Validation compares videos frame by frame for any pixel mismatches.<br><br>"

    "This program takes files in raw/uncompressed format, and uses FFMPEG<br>"
    "to convert to .MKV with the FFV1 codec.<br>"
    "FFV1 is a modern lossless codec which is used by the Library of Congress,<br>"
    "the US National Archives, and universities for video archiving.<br>"
    "FFV1 ensures no data loss, so the MKV files can be uncompressed back to the<br>"
    "original AVI if necessary for compatibility on older software or hardware.<br><br>"

    f"v{version}<br><br>"

    "<b>Third party components:</b><br>"
    "• This app uses Qt (PySide6) under terms of LGPL 3.0.<br>"
    "• This app bundles FFmpeg binaries under the terms of the LGPL-2.1+.<br>"
    "For full license text and source code links, see <a href=https://github.com/broemere/video-vise/tree/main/LICENSES>LICENSES</a>."

)


class PixelLoaderThread(QThread):
    # emits (fp, small_frame)
    finished = Signal(object, object)

    def __init__(self, fp, width, height, pix_fmt, parent=None):
        super().__init__(parent)
        self.fp = fp
        self.w = width
        self.h = height
        self.pix_fmt = pix_fmt

    def run(self):
        if self.fp.suffix.lower() in (".tif", ".tiff"):
            # Read the first page as a NumPy array
            arr = imread(self.fp, key=0).astype("uint8")
            # arr.shape is (H,W) for gray or (H,W,3) for RGB
            if arr.ndim == 2:
                # each row is a list of ints
                frame = arr.tolist()
            else:
                # convert each [r,g,b] list into a tuple
                frame = [
                    [tuple(pixel) for pixel in row]
                    for row in arr.tolist()
                ]
        else:
            cmd = [
                FFMPEG, "-hide_banner", "-loglevel", "error",
                "-ss", "0",
                "-i", str(self.fp),
                "-frames:v", "1",
                "-f", "image2pipe",
                "-vcodec", "ppm",              # P6 Portable Pixmap
                "pipe:1"
            ]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            stdout = proc.stdout

            # PPM header
            magic = stdout.readline().strip()      # b'P6'
            line  = stdout.readline()
            while line.startswith(b'#'):          # skip comments
                line = stdout.readline()
            width, height = map(int, line.split())
            stdout.readline()                      # skip maxval line

            # read all RGB triples
            n_bytes = width * height * 3
            raw = stdout.read(n_bytes)
            proc.wait()

            flat = list(raw)
            frame = []
            idx = 0
            if self.pix_fmt.lower().startswith("gray"):
                # treat each 3-byte group as one gray value, skip the other two
                for _ in range(height):
                    row = []
                    for _ in range(width):
                        gray = flat[idx]  # R == G == B
                        row.append(gray)
                        idx += 3  # jump to next pixel
                    frame.append(row)
            else:
                # full RGB parsing
                for _ in range(height):
                    row = []
                    for _ in range(width):
                        r = flat[idx]
                        g = flat[idx + 1]
                        b = flat[idx + 2]
                        row.append((r, g, b))
                        idx += 3
                    frame.append(row)

        # 3) slice to top-left 10×10 (or smaller)
        h_slice = min(5, self.h)
        w_slice = min(5, self.w)
        small_frame = [r[:w_slice] for r in frame[:h_slice]]

        # 4) emit back to GUI thread
        self.finished.emit(self.fp, small_frame)

class PixelDialog(QDialog):
    def __init__(self, fp, small_frame, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Inspect: {fp.name}")
        layout = QVBoxLayout(self)
        # simple text dump of the 10×10 block
        txt = "\n".join(str(row) for row in small_frame)
        layout.addWidget(QLabel(txt))
        self.setAttribute(Qt.WA_DeleteOnClose)


class UpdateChecker(QThread):
    """
    Background thread to check for updates on GitHub.
    Emits 'update_available' signal with the new version string if a newer version is found.
    """
    update_available = Signal(str)

    def run(self):
        try:
            # We use urllib to avoid adding an external dependency like 'requests'
            # The timeout ensures we don't hang indefinitely if the internet is flaky
            with urllib.request.urlopen(REPO_URL, timeout=5) as response:
                # GitHub redirects 'latest' to the specific tag URL.
                # Example: .../releases/tag/v0.1
                final_url = response.geturl()

                # Extract the tag (the last part of the URL)
                latest_tag = final_url.split('/')[-1]

                if self._is_version_newer(latest_tag, version):
                    self.update_available.emit(latest_tag)
        except Exception as e:
            # Log silently; we don't want to annoy the user if they are offline
            log.warning(f"Update check failed: {e}")

    def _is_version_newer(self, remote_tag: str, current_version: str) -> bool:
        """
        Compares two version strings (e.g., 'v0.1' vs '0.0.5').
        Returns True if remote_tag is logically greater than current_version.
        """
        try:
            # 1. Strip 'v' prefix and whitespace
            r_clean = remote_tag.lower().lstrip('v').strip()
            c_clean = current_version.lower().lstrip('v').strip()

            # 2. Convert to tuples of integers for accurate comparison
            # e.g. "0.1" -> (0, 1) and "0.0.5" -> (0, 0, 5)
            # Python natively compares tuples element-by-element:
            # (0, 1) > (0, 0, 5) evaluates to True because 1 > 0 at the second index.
            remote_parts = tuple(map(int, r_clean.split('.')))
            current_parts = tuple(map(int, c_clean.split('.')))

            return remote_parts > current_parts
        except ValueError:
            # Failsafe for tags that aren't standard numbers (e.g., "beta-release")
            log.warning(f"Could not parse version tags for comparison: {remote_tag} vs {current_version}")
            return False
