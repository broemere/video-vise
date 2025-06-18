"""
video_vise.py

A simple GUI for compressing raw .avi and .tif files with ffmpeg via ffv1, and validating lossless compression.

Usage:
    python video_vise.py

Author: Eli Broemer
Created: 2025-04-29
Version: 1.2

Dependencies:
    - Python >= 3.11
    - psutil >= 7
    - PySide6 >= 6.9
    - tifffile >= 2025.6
    - ffmpeg
"""

import os
import re
import sys
import psutil
import logging
import datetime
import traceback
import threading
import subprocess
from pathlib import Path, PurePath
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLineEdit, QPushButton, QDialog, QSizePolicy,
    QTableWidget, QTableWidgetItem, QFileDialog, QHeaderView,
    QProgressBar, QAbstractItemView, QLabel, QStyle, QSplashScreen, QMessageBox
)
from PySide6.QtGui import QPalette, QPixmap, QPainter, QFont, QFontMetrics, QIcon, QColor, QBrush
from PySide6.QtCore import QSettings, Qt, QThread, Signal
import json
from tifffile import TiffFile, imread
from fractions import Fraction
from statistics import median

APP_NAME = "VideoVise"
__version__ = "1.2.0"
supported_extensions = ["avi", "tif", "tiff", "mkv"]
DEFAULT_FPS = 10  # Make this a user settable option in the UI?

# region ─────────── resource loading ───────────

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
        base = os.path.dirname(__file__)
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
print(FFMPEG)
# endregion

# region ─────────── configure logging ───────────
LOGNAME = f"{APP_NAME.replace(' ', '_').lower()}_debug.log"
LOGFILE = resource_path(LOGNAME)
if getattr(sys, "frozen", False):     # when frozen, only log to file (no console output)
    LOGFILE = Path(LOGFILE).parent.parent / LOGNAME
    handlers = [logging.FileHandler(str(LOGFILE), mode="w", encoding="utf-8")]
else:    # in dev mode, log to file AND stderr
    handlers = [logging.FileHandler(str(LOGFILE), mode="w", encoding="utf-8"), logging.StreamHandler(sys.stderr)]
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(threadName)s] %(levelname)s: %(message)s", handlers=handlers)
logger = logging.getLogger(__name__)
print(f"Logging to {LOGFILE}")
# endregion ───────────────────────────────────────────


class FFmpegConverter(QThread):
    progress = Signal(int)
    result = Signal(str, int)

    def __init__(self, input_path: Path, output_path: Path, frames: int, mode: str, track: bool):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.frames = frames
        self.mode = mode
        self.track = track

    def run(self):
        logger.debug(
            f"FFmpegConverter thread STARTED: input={self.input_path}, "
            f"output={self.output_path}, frames={self.frames}, mode={self.mode!r}"
        )

        frame_re = re.compile(r"frame=\s*(\d+)\b")
        cpu_count = os.cpu_count() or 1
        threads = 3 if cpu_count == 4 else min(max(cpu_count - 4, 1), 16)
        slices = self.choose_slices(threads)

        if self._is_tiff():
            self._process_tiff(frame_re, threads, slices)
        else:
            self._process_video(frame_re, threads, slices)

        # finish up
        self.progress.emit(100)
        if self.track:
            self._emit_size_diff()

    def _is_tiff(self) -> bool:
        return self.input_path.suffix.lower() in [".tif", ".tiff"]

    def _process_tiff(self, frame_re, threads: int, slices: int):
        # 1) extract metadata
        fps, pix_fmt, w, h = self._extract_tiff_metadata()

        # 2) build & spawn ffmpeg
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pix_fmt", pix_fmt,
            "-s", f"{w}x{h}",
            "-r", f"{fps:.3f}",
            "-i", "pipe:0",
            "-an",
            "-c:v", "ffv1",
            "-level", "3",
            "-threads", str(threads),
            "-coder", "1", "-context", "1",
            "-g", "1", "-slices", str(slices),
            "-slicecrc", "1",
            str(self.output_path)
        ]
        print("Running compress command:", " ".join(cmd))
        logger.info("Spawning FFmpeg process: " + " ".join(cmd))
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        logger.debug(f"Launched FFmpeg (PID={proc.pid})")

        # 3) stream frames & report progress
        self._stream_tiff_frames(proc)

        if proc.wait() != 0:
            raise RuntimeError("FFmpeg TIFF→video failed")

    def _extract_tiff_metadata(self):
        with TiffFile(self.input_path) as tif:
            first = tif.pages[0].asarray()
            h, w = first.shape[:2]
            if first.ndim == 2:
                pix_fmt = "gray"
                logger.info("Found grayscale tif data")
            elif first.ndim == 3 and first.shape[2] == 3:  # Does not account for "false" rgb (ie 3 channel grayscale)
                pix_fmt = "rgb24"
                logger.info("Found rgb tif data")
            else:
                raise ValueError(f"Unsupported shape: {first.shape}")

            # --- fps logic ---
            imgj = tif.imagej_metadata or {}
            fps = 0

            if imgj:
                fps = imgj.get("fps", 0)
                logger.info(f"Acquired frame rate from tif metadata: {fps}")
            else:
                # 3) no ImageJ metadata → scan deviceTime
                times: list[float] = []
                for pg in tif.pages:
                    desc = pg.tags.get("ImageDescription")
                    if not desc:
                        logger.debug(f"No deviceTime in first page, stopping search.")
                        continue
                    try:
                        info = json.loads(desc.value)
                    except Exception:
                        continue
                    t = info.get("deviceTime")
                    if isinstance(t, (int, float)):
                        times.append(t)

                if len(times) >= 2:
                    deltas = [t2 - t1 for t1, t2 in zip(times, times[1:]) if (t2 - t1) > 0]
                    median_dt = median(deltas) if deltas else 0.1
                    fps = 1.0 / median_dt
                    logger.info(f"Derived frame rate from deviceTime: {fps}")
                else:
                    print(f"Cannot parse frame rate from this tif, falling back to default {DEFAULT_FPS}")
                    logger.debug(f"Cannot parse frame rate from this tif, falling back to default {DEFAULT_FPS}")

            # ensure non-negative
            fps = max(int(fps), 0)

        return fps, pix_fmt, w, h

    def _stream_tiff_frames(self, proc):
        step = 3
        next_emit = step
        with TiffFile(self.input_path) as tif:
            for idx, page in enumerate(tif.pages, start=1):
                frame = page.asarray().astype("uint8")
                proc.stdin.write(frame.tobytes())

                pct = min(int(idx / self.frames * 100), 100)
                if pct >= next_emit:
                    self.progress.emit(pct)
                    next_emit += step
        proc.stdin.close()

    def _process_video(self, frame_re, threads: int, slices: int):
        cmd = self._build_video_cmd(threads, slices)
        print(f"Running {'compress' if self.mode=='compress' else 'uncompress'} command:", " ".join(cmd))
        logger.info("Spawning FFmpeg process: " + " ".join(cmd))

        proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True)
        logger.debug(f"Launched FFmpeg (PID={proc.pid})")

        for line in proc.stderr:
            if self.frames:
                m = frame_re.search(line)
                if m:
                    frame_num = int(m.group(1))
                    pct = min(int(frame_num / self.frames * 100), 100)
                    self.progress.emit(pct)
            else:
                self.progress.emit(50)
        proc.wait()

    def _build_video_cmd(self, threads: int, slices: int):
        base = [FFMPEG, "-y", "-i", str(self.input_path)]
        if self.mode == "compress":
            return base + [
                "-acodec", "copy",
                "-vcodec", "ffv1",
                "-level", "3",
                "-threads", str(threads),
                "-coder", "1", "-context", "1",
                "-g", "1", "-slices", str(slices),
                "-slicecrc", "1",
                str(self.output_path)
            ]
        else:
            return base + [
                "-vcodec", "rawvideo",
                "-acodec", "copy",
                str(self.output_path)
            ]

    def _emit_size_diff(self):
        try:
            input_size = self.input_path.stat().st_size
            output_size = self.output_path.stat().st_size
            diff = int(round((input_size - output_size) / (1024 ** 3)))
            if diff == input_size:
                diff = 0
        except Exception:
            diff = 0

        name = str(self.input_path.name)[:-4].replace(" ", "")
        self.result.emit(name + str(input_size), diff)

    def choose_slices(self, threads: int) -> int:
        """
        Pick the best -slices value for FFV1 level 3.
        - Allowed:  4, 6, 9, 12, 16, 24, 30
        - Should be as close below or equal to threads*2 as possible
        """
        allowed = [4, 6, 9, 12, 16, 24, 30]
        ideal = threads * 2
        valid = [s for s in allowed if s <= ideal]
        if valid:
            return max(valid)
        return min(allowed)

class FrameValidator(QThread):
    progress = Signal(int)
    result   = Signal(str, bool)

    def __init__(self, orig: Path, comp: Path, frames: int):
        super().__init__()
        self.orig   = orig
        self.comp   = comp
        self.frames = frames
        self.step_pct = 3

    def run(self):
        logger.debug(f"FrameValidator thread STARTED: orig={self.orig}, comp={self.comp}")
        try:
            orig_hashes = self._hash_file(self.orig, 0, 50, is_original=True)
            comp_hashes = self._hash_file(self.comp, 50, 100)

            self.progress.emit(100)
            is_lossless = (orig_hashes == comp_hashes)
            logger.info(
                f"FrameValidator result: is_lossless={is_lossless} "
                f"(orig_frames={len(orig_hashes)} vs comp_frames={len(comp_hashes)})"
            )
            self.result.emit(self.comp.name, is_lossless)

        except Exception:
            logger.error("Exception in FrameValidator.run()", exc_info=True)
            # ensure UI doesn’t hang
            self.result.emit(self.comp.name, False)

        finally:
            logger.debug("FrameValidator thread END")

    def _hash_file(self, path: Path, start_pct: int, end_pct: int, *, is_original=False) -> list[str]:
        """Unified hasher for both video and TIFF."""
        suffix = path.suffix.lower()
        if suffix in (".tif", ".tiff"):
            return self._hash_tiff(path, start_pct, end_pct)
        else:
            return self._hash_video(path, start_pct, end_pct, is_original)

    def _hash_video(self, path, start_pct, end_pct, is_original):
        cmd = [FFMPEG, "-i", str(path), "-map", "0:v:0", "-f", "framemd5", "pipe:1"]
        if not is_original:
            pix = self._probe_pix_fmt(path)
            if pix == "bgr0":
                cmd.insert(3, "-vf"); cmd.insert(4, "format=bgr24")
            print("Running cmd", cmd)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

        hashes = []
        count = 0
        for line in proc.stdout:
            if line.startswith('#'):
                continue
            hashes.append(line.split(',')[-1].strip())
            count += 1
            if self.frames:
                # Map to the appropriate slice of the progress bar
                pct = start_pct + min(int(count / self.frames * (end_pct - start_pct)), end_pct - start_pct)
                self.progress.emit(pct)

        proc.wait()
        logger.debug(f"Completed framemd5 for {'ORIG' if is_original else 'COMP'} ({count} frames)")
        # if no frame-count was provided, emit the boundary progress
        if self.frames == 0:
            self.progress.emit(end_pct)
        return hashes

    def _hash_tiff(self, path: Path, start_pct: int, end_pct: int) -> list[str]:
        """
        Compute framemd5 hashes for a multi-page TIFF by piping raw frames to FFmpeg.
        Emits progress between start_pct and end_pct.
        Returns a list of MD5 hashes (one per page).
        """
        with TiffFile(path) as tif:
            pages = tif.pages
            total = len(pages)
            first = pages[0].asarray().astype("uint8")
            h, w = first.shape[:2]
            pix_fmt = "gray" if first.ndim == 2 else "rgb24"

            cmd = [
                FFMPEG, "-f", "rawvideo",
                "-pix_fmt", pix_fmt,
                "-s", f"{w}x{h}",
                "-i", "pipe:0",
                "-map", "0:v:0",
                "-vf", f"format={pix_fmt}",
                "-f", "framemd5", "pipe:1"
            ]
            proc = subprocess.Popen(cmd,
                                    stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL)

            # 2) Drain stdout on a background thread
            hashes: list[str] = []
            reader = threading.Thread(target=self._drain_stdout, args=(proc.stdout, hashes))
            reader.daemon = True
            reader.start()

            # 3) Feed each frame into stdin
            last_emit = start_pct
            for idx, page in enumerate(pages, start=1):
                frame = page.asarray().astype("uint8")
                # drop extra channels if grayscale declared
                if pix_fmt == "gray" and frame.ndim == 3:
                    frame = frame[..., 0]
                proc.stdin.write(frame.tobytes())

                frac = idx / total
                scaled = start_pct + frac * (end_pct - start_pct)
                if scaled - last_emit >= self.step_pct:
                    last_emit += self.step_pct
                    self.progress.emit(int(last_emit))

            proc.stdin.close()

            # 4) Wait for reader + process to finish
            reader.join()
            ret = proc.wait()
            if ret != 0:
                raise RuntimeError("FFmpeg framemd5 failed on TIFF")

        return hashes

    def _probe_pix_fmt(self, path: Path) -> str:
        """
        Use ffprobe to extract the pixel format of the first video stream.
        """
        try:
            out = subprocess.check_output([
                FFPROBE, "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=pix_fmt",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path)
            ], text=True).strip()
            logger.debug(f"Pixel format of {path}: {out}")
            return out
        except subprocess.CalledProcessError as e:
            logger.error(f"pix_fmt error: {e}", exc_info=True)
            return ""

    def _drain_stdout(self, pipe, hashes):
        for raw in pipe:
            line = raw.decode("utf-8", "ignore")
            if not line.startswith("#"):
                hashes.append(line.rsplit(",", 1)[-1].strip())

class PixelLoaderThread(QThread):
    # emits (fp, small_frame)
    finished = Signal(object, object)

    def __init__(self, fp, width, height, pix_fmt):
        super().__init__()
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
                "ffmpeg", "-hide_banner", "-loglevel", "error",
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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1300, 750)
        self.settings = QSettings("TykockiLab", APP_NAME.replace(" ",""))
        self.worker = None
        self.batch_queue = []
        self.total_tasks = 0
        self.lossless_results = {}
        self.durations = {}
        self.frames = {}
        self.cancel_requested = False
        self.num_compress = 0
        self.filesizes_reduced = {}
        self.track_storage = False
        self.storage_path = None
        self.active_threads = []   # keep threads alive
        self.active_dialogs = []   # keep dialogs alive
        self.init_ui()
        last = self.settings.value("lastFolder", "")
        if last and os.path.isdir(last):
            self.path_edit.setText(last)
            self.update_table(last)
        self.init_storage_tracking()

    def init_ui(self):
        widget = QWidget()
        self.setCentralWidget(widget)
        layout = QVBoxLayout(widget)
        path_layout = QHBoxLayout()
        self.path_edit = QLineEdit(); self.path_edit.setReadOnly(True)
        browse_btn = QPushButton(self.style().standardIcon(QStyle.SP_DialogOpenButton), "Open Folder"); browse_btn.clicked.connect(self.browse_folder)
        refresh_btn = QPushButton(); refresh_btn.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        refresh_btn.setToolTip("Refresh"); refresh_btn.clicked.connect(lambda: self.update_table(self.path_edit.text()))
        help_btn = QPushButton("Help"); help_btn.setToolTip("Show usage instructions"); help_btn.clicked.connect(self.show_help)
        path_layout.addWidget(self.path_edit); path_layout.addWidget(browse_btn); path_layout.addWidget(refresh_btn); path_layout.addWidget(help_btn)
        layout.addLayout(path_layout)
        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        #self.table.setStyleSheet("QTableWidget { alternate-background-color: #f7f7f7; }")
        pal = self.table.palette()
        base = pal.color(QPalette.Base)
        # Calculate perceived brightness
        brightness = (base.red() * 0.299 + base.green() * 0.587 + base.blue() * 0.114)
        if brightness < 128:
            # Dark background: lighten alternate rows
            alt = base.lighter(50)
        else:
            # Light background: darken alternate rows
            alt = base.darker(110)
        pal.setColor(QPalette.AlternateBase, alt)
        self.table.setPalette(pal)
        cols = [
            " ", "Filename","Size (GB)","Created","Duration","Codec","PixelFmt",
            "Resolution","Tag","Color/Gray","FPS","Compress","Validate","Lossless", "Relative Size", "Uncompress", "Inspect"
        ]
        self.table.setColumnCount(len(cols)); self.table.setHorizontalHeaderLabels(cols)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        layout.addWidget(self.table)

        # Batch buttons + Cancel
        batch_layout = QHBoxLayout()
        self.btn_batch_compress = QPushButton("Compress All"); self.btn_batch_compress.clicked.connect(self.process_all_compress)
        self.btn_batch_validate = QPushButton("Validate All"); self.btn_batch_validate.clicked.connect(self.process_all_validate)
        self.btn_batch_uncompress = QPushButton("Uncompress All"); self.btn_batch_uncompress.clicked.connect(self.process_all_uncompress)
        self.btn_batch_compress_validate = QPushButton("Compress and Validate All");
        self.btn_batch_compress_validate.clicked.connect(self.process_all_compress_validate)
        self.btn_cancel = QPushButton(self.style().standardIcon(QStyle.SP_BrowserStop), "Cancel"); self.btn_cancel.clicked.connect(self.cancel_batch)
        self.btn_cancel.setEnabled(False)
        batch_layout.addWidget(self.btn_cancel)
        batch_layout.addStretch()
        for btn in [self.btn_batch_compress, self.btn_batch_validate,
                    self.btn_batch_uncompress, self.btn_batch_compress_validate]:
            btn.setMinimumHeight(40)
            batch_layout.addWidget(btn)
        layout.addLayout(batch_layout)

        status_layout = QHBoxLayout()
        self.status = QLabel("Ready")#; self.status.setAlignment(Qt.AlignLeft)
        self.gb_saved = QLabel()#; self.gb_saved.setAlignment(Qt.AlignRight)
        self.progress = QProgressBar(); self.progress.setRange(0, 100); self.progress.setTextVisible(False)
        self.progress.setStyleSheet("QProgressBar{border:1px solid gray; border-radius:5px;}"+
                                   "QProgressBar::chunk{background-color:#4CAF50}")
        self.progress.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.status); status_layout.addStretch(); status_layout.addWidget(self.gb_saved)
        layout.addLayout(status_layout)
        layout.addWidget(self.progress)

    def init_storage_tracking(self):
        # Use POSIX‐style separator here; pathlib will convert as needed on Windows.
        self.tracker_file = "Lab/Software/storage_saved.jsonl"
        print(f"Looking for '{self.tracker_file}' on network drives…")
        matches = find_file_on_network_drives(self.tracker_file)
        if matches:
            print("  → Found:", matches[0])
            self.tracker_file = matches[0]
        else:
            print("  → File not found on network, tracking disabled\n")
        self.load_storage_jsonl(self.tracker_file)
        if self.track_storage:
            print("Tracking storage savings\n")
            self.gb_saved.setText(f"{self.get_gb_saved()} GB saved")

    def flash_window(self):
        # This will flash the taskbar button on Windows, and bounce the Dock icon on macOS.
        QApplication.alert(self, 0)  # 0 means “use the default platform timeout”

    def show_help(self):
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

            f"v{__version__}"
        )
        msg = QMessageBox(self)
        msg.setWindowTitle("Help")
        msg.setTextFormat(Qt.RichText)
        msg.setText(instructions)
        # Allow clicking links
        msg.setTextInteractionFlags(Qt.TextBrowserInteraction)
        msg.setStandardButtons(QMessageBox.Close)
        msg.setDefaultButton(QMessageBox.Close)
        msg.exec()

    def get_files(self, folder=None, mode=None):
        folder = folder if folder is not None else self.path_edit.text()
        folder_path = Path(folder)
        if not folder_path.is_dir():
            return []
        exts = list(supported_extensions)
        if mode == "compress":
            exts.remove("mkv")
        elif mode == "validate":
            exts = ["mkv"]
        files = [
            p for p in folder_path.iterdir()
            if p.is_file() and p.suffix.lower().lstrip('.') in exts
        ]
        return sorted(
            files,
            key=lambda p: (
                exts.index(p.suffix.lower().lstrip('.')),
                p.name.lower()     # or p.name if you want case‐sensitive name sort
            )
        )

    def get_gb_saved(self):
        """
        Return the sum of all numeric values in `data`.
        If `data` is empty, sum() will return 0 automatically.
        """
        self.total_gb_saved = sum(self.filesizes_reduced.values())
        return self.total_gb_saved

    def load_storage_jsonl(self, path):
        """
        Read every line from storage_saved.jsonl,
        parse as JSON objects, and merge into a single dict.
        Later lines override earlier ones if keys collide.
        """
        result = {}
        path = Path(path)
        if not path.exists():
            self.filesizes_reduced = result
            return None

        self.track_storage = True
        self.storage_path = path
        with self.storage_path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # skip invalid lines
                    continue

                # Expecting each line to be a dict with exactly one key
                # or potentially multiple keys if you want. We just update()
                if isinstance(obj, dict):
                    result.update(obj)
                else:
                    # if it’s not a dict, skip
                    continue
        self.filesizes_reduced = result

    def add_member_jsonl(self, key: str, value) -> None:
        """
        Append a new JSON line: {key: value}.
        This is constant-time (just writes one line).
        """
        # Ensure directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Create the JSON object for this one key/value pair
        with self.storage_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()  # push Python’s internal buffer out to the OS
            os.fsync(f.fileno())  # ask the OS to flush its buffers all the way to the physical file

    def browse_folder(self):
        # 1) Read the saved “lastFolder” (may be empty)
        last = self.settings.value("lastFolder", type=str)

        if last:
            # Always try to use the parent of the last folder
            parent = Path(last).parent
            if parent.exists() and parent.is_dir():
                initial_dir = str(parent)
            else:
                initial_dir = str(Path.home())
        else:
            initial_dir = str(Path.home())

        # 2) Launch the dialog, forcing ShowDirsOnly/ReadOnly (optional flags)
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder",
            initial_dir,
            QFileDialog.ShowDirsOnly | QFileDialog.ReadOnly
        )

        # 3) If the user chose something, save it and refresh
        if folder:
            self.path_edit.setText(folder)
            self.settings.setValue("lastFolder", folder)
            self.lossless_results.clear()
            self.update_table(folder)

    def update_progress(self, val: int):
        self.progress.setValue(val)

    def update_table(self, folder: str):
        if not folder or not os.path.isdir(folder): return
        self.status.setText("Refreshing...")
        self.progress.setValue(0)
        self.table.clearContents()
        QApplication.processEvents()

        files = self.get_files(folder)
        self.table.setRowCount(len(files))
        QApplication.processEvents()

        ext_color = {
            ".tif": "#4e84af",  # FIJI/ImageJ color
            ".tiff": "#4e84af",
            ".avi": "#FF5500",  # burnt orange
            ".mkv": "#9c27b0",  # deep purple (lighter: #9575cd)
        }

        for r, fp in enumerate(files):
            info = get_video_info(fp)
            codec = info.get("codec_name", "Unknown")
            # — Column 0: Color swatch
            color_item = QTableWidgetItem()
            hexcol = ext_color.get(fp.suffix.lower())
            if hexcol:
                color_item.setBackground(QBrush(QColor(hexcol)))
            self.table.setItem(r, 0, color_item)
            self.table.setItem(r, 1, QTableWidgetItem(fp.name))
            size_gb = fp.stat().st_size / 1024 ** 3
            size_item = QTableWidgetItem(f"{size_gb:.2f}")
            size_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(r, 2, size_item)
            created = datetime.datetime.fromtimestamp(fp.stat().st_ctime)
            self.table.setItem(r, 3, QTableWidgetItem(created.strftime("%Y-%m-%d %H:%M:%S")))
            dur = float(info.get("duration", "0"))
            self.frames[fp.name] = int(info.get("frames", 0))
            dur_str = str(datetime.timedelta(seconds=int(dur))) if dur else "N/A"
            self.table.setItem(r, 4, QTableWidgetItem(dur_str))
            self.table.setItem(r, 5, QTableWidgetItem(codec))
            pix = info.get("pix_fmt", "Unknown")
            self.table.setItem(r, 6, QTableWidgetItem(pix))
            w = info.get("width", "0")
            h = info.get("height", "0")
            res = f"{w}x{h}" if w and h else "N/A"
            self.table.setItem(r, 7, QTableWidgetItem(res))
            tag = info.get("codec_tag_string", "")
            if not re.match(r"^[A-Za-z0-9_.]+$", tag): tag = "N/A"
            self.table.setItem(r, 8, QTableWidgetItem(tag))
            color = "Gray" if "gray" in pix.lower() else "Color"
            self.table.setItem(r, 9, QTableWidgetItem(color))
            fps = info.get("fps", "0")
            self.table.setItem(r, 10, QTableWidgetItem(fps))
            # compress only if not ffv1
            if codec != "ffv1":
                btnc = QPushButton("Compress")
                btnc.clicked.connect(lambda _, p=fp: self.start_convert(p, "compress"))
                if codec == "mjpeg":  # Check for MJPEG (lossy compressed format, cannot compress more)
                    btnc = QLabel("Already compressed")
                if dur == 0 and codec == "Unknown" and float(w) == 0 and float(h) == 0:  # Check for a bad video file
                    btnc = QLabel("Cannot read")
                    btnc.setAlignment(Qt.AlignCenter)
                self.table.setCellWidget(r, 11, btnc)
            # Validate & uncompress for ffv1
            if codec == "ffv1":
                btnv = QPushButton("Validate")
                btnv.clicked.connect(lambda _, p=fp: self.start_validate(p))
                self.table.setCellWidget(r, 12, btnv)
                btnd = QPushButton("Uncompress")
                btnd.clicked.connect(lambda _, p=fp: self.start_convert(p, "uncompress"))
                self.table.setCellWidget(r, 15, btnd)
            # Lossless column: center-align icon
            label = QLabel()
            label.setAlignment(Qt.AlignCenter)
            if fp.name in self.lossless_results:
                ok = self.lossless_results[fp.name]
                icon = self.style().standardIcon(QStyle.SP_DialogApplyButton) if ok else self.style().standardIcon(QStyle.SP_DialogCancelButton)
                label.setPixmap(icon.pixmap(24, 24))
            self.table.setCellWidget(r, 13, label)

            # Size (%) vs original AVI
            pct_item = QTableWidgetItem()
            if fp.suffix.lower() == ".mkv":
                avi_fp = fp.with_suffix(".avi")
                tif_fp = fp.with_suffix(".tif")
                tiff_fp = fp.with_suffix(".tiff")
                if avi_fp.exists():
                    mkv_size = fp.stat().st_size
                    avi_size = avi_fp.stat().st_size
                    pct = (mkv_size / avi_size * 100) if avi_size else 0
                    pct_str = f"{pct:.0f}%"
                elif tif_fp.exists():
                    mkv_size = fp.stat().st_size
                    tif_size = tif_fp.stat().st_size
                    pct = (mkv_size / tif_size * 100) if tif_size else 0
                    pct_str = f"{pct:.0f}%"
                elif tiff_fp.exists():
                    mkv_size = fp.stat().st_size
                    tiff_size = tiff_fp.stat().st_size
                    pct = (mkv_size / tiff_size * 100) if tiff_size else 0
                    pct_str = f"{pct:.0f}%"
                else:
                    pct_str = "N/A"
            else:
                pct_str = ""
            pct_item.setText(pct_str)
            pct_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(r, 14, pct_item)

            btni = QPushButton()
            btni.setIcon(self.style().standardIcon(QStyle.SP_FileDialogContentsView))
            btni.setProperty("fp", fp)
            btni.setProperty("width", w)
            btni.setProperty("height", h)
            btni.setProperty("pix_fmt", pix)
            btni.clicked.connect(self.inspect_pixels)
            btni.setStyleSheet("""
                QPushButton {
                    padding-top:    2px;
                    padding-bottom: 2px;
                    padding-left:   4px;
                    padding-right:  4px;
                }
            """)
            self.table.setCellWidget(r, 16, btni)

            self.progress.setValue(int(100*(r/len(files))))

        self.table.resizeColumnsToContents()
        self.status.setText("Ready")
        self.progress.setValue(0)
        if self.track_storage:
            self.gb_saved.setText(f"{int(self.get_gb_saved())} GB saved")
        QApplication.processEvents()

    # def inspect_pixels(self):
    #     btn = self.sender()
    #     fp = btn.property("fp")  # pathlib.Path or str
    #     w = btn.property("width")  # int
    #     h = btn.property("height")  # int
    #     pix_fmt = btn.property("pix_fmt")  # “gray” or “rgb24” (or similar)
    #
    #     # build the ffmpeg command
    #     cmd = [
    #         "ffmpeg", "-hide_banner", "-loglevel", "error",
    #         "-ss", "0",
    #         "-i", str(fp),
    #         "-frames:v", "1",
    #         "-f", "rawvideo",
    #         "-pix_fmt", pix_fmt,
    #         "-s", f"{w}x{h}",
    #         "pipe:1"
    #     ]
    #
    #     # launch and read exactly the right number of bytes
    #     proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    #     bpp = 1 if pix_fmt == "gray" else 3  # bytes per pixel
    #     n_bytes = w * h * bpp
    #     raw = proc.stdout.read(n_bytes)
    #     proc.wait()
    #
    #     # raw is a bytes object of length n_bytes
    #     # now turn that into a Python structure:
    #
    #     if pix_fmt.startswith("gray"):
    #         # Convert the flat bytes to a flat list of ints:
    #         flat = list(raw)  # each raw[i] → 0–255
    #
    #         # (Option A) keep it flat and index with raw[y*w + x]
    #         # pixel = flat[y*w + x]
    #
    #         # (Option B) make a 2D list for convenience:
    #         frame = [
    #             flat[row * w: (row + 1) * w]
    #             for row in range(h)
    #         ]
    #     else:
    #         # your existing BGR/RGB path…
    #         frame = []
    #         idx = 0
    #         for row in range(h):
    #             pixels = []
    #             for col in range(w):
    #                 b, g, r = raw[idx], raw[idx + 1], raw[idx + 2]
    #                 pixels.append((r, g, b))
    #                 idx += 3
    #             frame.append(pixels)
    #
    #     h_slice = min(5, h)
    #     w_slice = min(5, w)
    #
    #     # this works for both gray (rows of ints) and rgb (rows of (r,g,b) tuples):
    #     small_frame = [row[:w_slice] for row in frame[:h_slice]]
    #
    #     print(small_frame)

    def inspect_pixels(self):
        btn     = self.sender()
        fp      = btn.property("fp")
        w       = btn.property("width")
        h       = btn.property("height")
        pix_fmt = btn.property("pix_fmt")

        loader = PixelLoaderThread(fp, w, h, pix_fmt)
        loader.finished.connect(self.on_pixels_ready)
        loader.finished.connect(lambda *_: self.active_threads.remove(loader))
        self.active_threads.append(loader)
        loader.start()

    def on_pixels_ready(self, fp, small_frame):
        dlg = PixelDialog(fp, small_frame, parent=self)
        dlg.finished.connect(lambda _: self.active_dialogs.remove(dlg))
        self.active_dialogs.append(dlg)
        dlg.show()

    def start_convert(self, fp: Path, mode: str):
        if mode == "compress":
            out_fp = fp.with_name(fp.stem + ".mkv")
            nframes = self.frames[fp.name]
        else:
            stem = fp.stem
            out_fp = fp.with_name(fp.stem + "_RAW.avi")
            try:
                # Estimate total frames from duration and FPS
                out = subprocess.check_output([
                    "ffprobe", "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(fp)
                ], text=True).strip()
                duration = float(out)
                out = subprocess.check_output([
                    "ffprobe", "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=avg_frame_rate",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(fp)
                ], text=True).strip()
                num, den = out.split("/")
                fps = float(num) / float(den)
                nframes = int(round(duration * fps))
            except Exception as e:
                logger.error(f"Failed to estiamte frames: {e}", exc_info=True)

        self.status.setText(f"{mode.title()}ing {fp.name}")
        self.progress.setValue(0)
        self.worker = FFmpegConverter(fp, out_fp, nframes, mode, self.track_storage)
        self.worker.progress.connect(self.update_progress)
        self.worker.result.connect(self.on_conversion_result)
        self.worker.finished.connect(lambda: self.update_table(self.path_edit.text()))
        self.worker.start()

    def on_conversion_result(self, key_name: str, result: int):
        if result > 0:
            self.load_storage_jsonl(self.tracker_file)
            if key_name not in self.filesizes_reduced.keys():
                self.filesizes_reduced[key_name] = result
                self.gb_saved.setText(f"{self.get_gb_saved()} GB saved")
                self.add_member_jsonl(key_name, result)

    def start_validate(self, fp: Path):
        orig_fp = find_original_file(fp)
        nframes = self.frames[orig_fp.name]
        self.status.setText(f"Validating {fp.name} against {orig_fp.name}")
        self.progress.setValue(0)
        self.worker = FrameValidator(orig_fp, fp, nframes)
        self.worker.progress.connect(self.update_progress)
        self.worker.result.connect(self.on_validation_result)
        self.worker.start()

    def on_validation_result(self, filename: str, ok: bool):
        self.lossless_results[filename] = ok
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 1)
            if item and item.text() == filename:
                label = QLabel()
                label.setAlignment(Qt.AlignCenter)
                icon = self.style().standardIcon(QStyle.SP_DialogApplyButton) if ok else self.style().standardIcon(QStyle.SP_DialogCancelButton)
                label.setPixmap(icon.pixmap(24, 24))
                self.table.setCellWidget(r, 13, label)
                break
        self.status.setText("Ready")
        self.progress.setValue(0)

    # Batch processing logic
    def process_all_compress(self):
        folder = self.path_edit.text()
        #files = sorted(Path(folder).glob("*.avi"))
        files = self.get_files(folder, "compress")
        tasks = [(fp, 'compress') for fp in files]
        self.start_batch(tasks)

    def process_all_uncompress(self):
        folder = self.path_edit.text()
        #files = sorted(Path(folder).glob("*.mkv"))
        files = self.get_files(folder, "validate")
        tasks = [(fp, 'uncompress') for fp in files]
        self.start_batch(tasks)

    def process_all_validate(self):
        folder = self.path_edit.text()
        #files = sorted(Path(folder).glob("*.mkv"))
        files = self.get_files(folder, "validate")
        self.num_compress = 0
        tasks = [(fp, 'validate') for fp in files]
        self.start_batch(tasks)

    def process_all_compress_validate(self):
        # compress then validate
        folder = self.path_edit.text()
        #files = sorted(Path(folder).glob("*.avi"))
        files = self.get_files(folder, "compress")
        self.num_compress = len(files)
        compress_tasks = [(fp, 'compress') for fp in files]
        validate_tasks = [(fp.with_name(fp.stem + '.mkv'), 'validate') for fp, _ in compress_tasks]
        self.start_batch(compress_tasks + validate_tasks)

    def start_batch(self, tasks):
        if not tasks:
            return
        self.cancel_requested = False
        self.btn_cancel.setEnabled(True)
        self.batch_queue = list(tasks)
        self.total_tasks = len(tasks)
        self._run_next_batch()

    def cancel_batch(self):
        # request cancellation after current task
        self.cancel_requested = True
        self.btn_cancel.setEnabled(False)
        self.status.setText('Canceling after current task...')

    def _run_next_batch(self):
        stack = "".join(traceback.format_stack())
        current = threading.current_thread().name
        logger.debug(f"_run_next_batch ENTERED (thread={current}). Full call stack:\n{stack}")

        if self.cancel_requested:
            logger.debug("_run_next_batch: cancel_requested=True → clearing queue and returning")
            self.batch_queue.clear()
            self.status.setText("Canceled")
            self.progress.setValue(0)
            self.update_table(self.path_edit.text())
            self.btn_cancel.setEnabled(False)
            return

        if not self.batch_queue:
            logger.debug("_run_next_batch: queue is empty → setting Ready and returning")
            self.status.setText("Ready")
            self.progress.setValue(0)
            self.btn_cancel.setEnabled(False)
            self.update_table(self.path_edit.text())
            self.flash_window()
            return

        # If there is an existing worker still running, refuse to start a new one.
        if self.worker is not None and self.worker.isRunning():
            logger.error("_run_next_batch: Detected self.worker is still running → ABORT starting a second worker! queue length remains %d",
                         len(self.batch_queue))
            return

        next_fp, next_mode = self.batch_queue[0]
        idx = self.total_tasks - len(self.batch_queue) + 1
        # Update the table if we had >0 compresss, and we're about to run the first validate:
        if self.num_compress > 0 and next_mode == 'validate' and idx == self.num_compress + 1:
            logger.debug("All compress tasks finished → calling update_table() before starting validation")
            self.update_table(self.path_edit.text())

        fp, mode = self.batch_queue.pop(0)
        idx = self.total_tasks - len(self.batch_queue)
        logger.debug(f"_run_next_batch: dispatching task {idx}/{self.total_tasks}: {mode} '{fp.name}'")
        self.status.setText(f"{mode.title() if mode == 'compress' else mode.title()[:-1]}ing {fp.name} ({idx}/{self.total_tasks})")
        nframes = self.frames[fp.name]
        # Create our new worker
        if mode in ("compress", "uncompress"):
            out_fp = (fp.with_suffix(".mkv") if mode == "compress" else fp.with_suffix(".avi"))
            worker = FFmpegConverter(fp, out_fp, nframes, mode, self.track_storage)
            logger.debug(f"_run_next_batch: created FFmpegConverter for {fp.name} → out {out_fp.name}")
            worker.result.connect(self.on_conversion_result)
            worker.finished.connect(self._run_next_batch)
        else:  # mode == "validate"
            orig = find_original_file(fp)
            nframes = self.frames[orig.name]
            worker = FrameValidator(orig, fp, nframes)
            logger.debug(f"_run_next_batch: created FrameValidator for {fp.name}")
            worker.result.connect(self.on_validation_result)
            worker.finished.connect(self._run_next_batch)

        # Hold a reference so it won’t be garbage‐collected
        self.worker = worker
        worker.progress.connect(self.progress.setValue)
        logger.debug("_run_next_batch: Starting worker thread")
        worker.start()

# region ─────────── Video Scanners ───────────

def get_video_info(fp: Path) -> dict[str, str]:
    """
    Returns a dict with keys:
      duration, codec_name, pix_fmt, width, height, codec_tag_string, color, fps
    Guaranteed to return strings, with sensible defaults on error/missing fields.
    """
    # 1) default return values
    default = {
        "duration": "0",
        "codec_name": "Unknown",
        "pix_fmt": "Unknown",
        "width": "0",
        "height": "0",
        "codec_tag_string": "Unknown",
        "color": "Unknown",
        "fps": "Unknown",
        "frames": "0"
    }

    try:
        suffix = fp.suffix.lower()
        if suffix in (".tif", ".tiff"):
            info = inspect_tiff(fp)
        else:
            info = inspect_ffprobe(fp)

        # 3) coerce everything to string and fill in missing keys
        return {k: str(info.get(k, default[k])) for k in default}

    except Exception as e:
        logger.error(f"get_video_info failed for {fp.name}: {e}", exc_info=True)
        return default

def inspect_tiff(fp: Path) -> dict[str, any]:

    with TiffFile(fp) as tif:
        pages = tif.pages
        n_pages = len(pages)
        first = pages[0]
        h, w = first.shape[:2]
        bits = first.dtype.itemsize * 8
        phot = getattr(first, "photometric", None)
        phot = phot.name if phot else None

        # --- photometric interpretation via page.photometric ---
        phot = getattr(first, "photometric", None)
        phot_name = phot.name if phot else "UNKNOWN"

        # map to pix_fmt + color
        if phot_name in ("MINISBLACK", "MINISWHITE"):
            pix_fmt, color = f"gray{bits}", "gray"
        elif phot_name == "RGB":
            pix_fmt, color = f"rgb{bits}", "color"
        else:
            pix_fmt = f"{phot_name.lower()}{bits}"
            color = phot_name.lower()

        # --- compression via page.compression ---
        comp_tag = first.tags.get('Compression')
        if comp_tag is not None:
            comp_value = comp_tag.value
            # map the most common TIFF compression codes
            compression_map = {
                1: "raw",  # no compression
                5: "lzw",
                6: "jpeg",
                7: "jpeg",  # sometimes JPEG is 6 or 7
                8: "deflate",
                32773: "packbits",
            }
            codec_tag = compression_map.get(comp_value, str(comp_value))
        else:
            codec_tag = ""

        # --- duration & fps logic ---
        imgj = tif.imagej_metadata or {}
        duration_sec = 0.0
        fps = 0

        if imgj:
            # 1) try Labels list
            labels = imgj.get("Labels")
            if isinstance(labels, (list, tuple)) and labels:
                last = labels[-1]  # e.g. "347.50 s"
                try:
                    duration_sec = float(last.rstrip(" s"))
                    # derive fps if possible
                    fps = imgj.get("fps", 0)
                except ValueError:
                    duration_sec = 0.0
                    fps = imgj.get("fps", 0) or 0

            # 2) fallback to fps header
            if duration_sec == 0.0 and imgj.get("fps"):
                fps = imgj["fps"]
                duration_sec = (n_pages - 1) / fps if fps > 0 else 0.0

        else:
            # 3) no ImageJ metadata → scan deviceTime
            times: list[float] = []
            for pg in pages:
                desc = pg.tags.get("ImageDescription")
                if not desc:
                    continue
                try:
                    info = json.loads(desc.value)
                except Exception:
                    continue
                t = info.get("deviceTime")
                if isinstance(t, (int, float)):
                    times.append(t)
                    if len(times) >= 2:
                        break

            if len(times) >= 2:
                duration_sec = times[-1] - times[0]
                fps = round((len(times) - 1) / duration_sec) if duration_sec > 0 else 0

        # ensure non-negative
        duration_sec = max(duration_sec, 0.0)
        fps = max(int(fps), 0)

        return {
            "duration": duration_sec,
            "codec_name": "tiff",
            "pix_fmt": pix_fmt,
            "width": w,
            "height": h,
            "codec_tag_string": codec_tag,
            "color": color,
            "fps": fps,
            "frames": n_pages,
        }

def inspect_ffprobe(fp: Path) -> dict[str, any]:

    args = [
        FFPROBE, "-v", "error",
        "-show_format", "-show_streams", "-of", "json", str(fp)
    ]
    raw = subprocess.check_output(args, text=True)
    data = json.loads(raw)

    fmt = data.get("format", {})
    # pick the first video stream
    vs = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})

    return {
        "duration": fmt.get("duration", 0),
        "codec_name": vs.get("codec_name", "Unknown"),
        "pix_fmt": vs.get("pix_fmt", "Unknown"),
        "width": vs.get("width", 0),
        "height": vs.get("height", 0),
        "codec_tag_string": vs.get("codec_tag_string", "Unknown"),
        "fps": format_decimal(frac_to_float(vs.get("avg_frame_rate", "Unknown"))),
        "frames": vs.get("nb_frames", 0),
    }

# endregion

# region ─────────── Networking ───────────

def list_network_drives():
    """
    Return a list of dicts for each partition that looks like a network volume.
    On Windows, we detect network drives by checking part.device.startswith("\\\\").
    On macOS/Linux, we detect via fstype ∈ NETWORK_FS_TYPES or "remote" ∈ opts.
    """
    drives = []
    for part in psutil.disk_partitions(all=True):
        # ==== Windows: UNC‐style device path (e.g. '\\\\SERVER\\Share') ====
        if sys.platform.startswith("win") and part.device.startswith(r"\\"):
            drives.append({
                "device": part.device,
                "mountpoint": part.mountpoint,
                "fstype": part.fstype,
                "opts": part.opts,
            })

        # ==== macOS/Linux (and also catches any other network fs that psutil knows) ====
        else:
            ft = part.fstype.upper()
            is_network_fstype = ft in {"CIFS", "SMBFS","NFS", "AFPFS", "WEBDAV", "DAVFS"}
            is_remote_flag = "remote" in part.opts.lower()
            if is_network_fstype or is_remote_flag:
                drives.append({
                    "device": part.device,
                    "mountpoint": part.mountpoint,
                    "fstype": part.fstype,
                    "opts": part.opts,
                })

    return drives

def find_file_on_network_drives(relative_path: str):
    """
    Given a path like 'folder/subfolder/file.ext' (POSIX style),
    this function returns a list of absolute‐normalized strings where that file exists
    on any network‐mounted drive.
    """
    found_paths = []
    rel = PurePath(relative_path)        # Treat input as “POSIX‐style” components
    for d in list_network_drives():
        mount = d["mountpoint"]
        # Build the candidate path via pathlib
        candidate = Path(mount) / rel
        # candidate.resolve() will normalize things: e.g. convert "C:/" or resolve symlinks
        try:
            candidate = candidate.resolve(strict=False)
        except Exception:
            # In some rare cases (e.g. the network drive is offline), .resolve() might fail.
            # We'll just keep the un‐resolved Path then.
            pass
        # Convert to string in OS-native form:
        candidate_str = str(candidate)
        if candidate.is_file():
            found_paths.append(candidate_str)
    return found_paths

def find_original_file(base_fp: Path, exts = ('avi', 'tif', 'tiff')):
    """
    Given a base filepath (e.g. /some/dir/video.mkv), look for
    /some/dir/video.<ext> in order, returning the first one that exists.
    If none exist, returns None.
    """
    # strip any suffix, so video.mkv → video
    stem = base_fp.with_suffix("")
    for ext in exts:
        candidate = stem.with_suffix(f".{ext}")
        if candidate.exists():
            print(f"Found original file {candidate}")
            logger.info(f"Found original file {candidate}")
            return candidate
    return None

# endregion ───────────────────────────────────────────

# region ─────────── Formatting ───────────

def frac_to_float(frac_str):
    """
    Turn a string fraction like "30000/1001" or "10/1"
    (or even "29.97") into a float.
    """
    try:
        return float(Fraction(frac_str))
    except ValueError:
        try:
            return float(frac_str)
        except ValueError:
            return "Unknown"

def format_decimal(value: float | None, max_decimals: int = 2) -> str:
    """
    Round to at most `max_decimals` places.
    If the result ends in .0, drop the decimal entirely.
    Returns "N/A" on None.
    """
    if value is None:
        return "N/A"
    # round and format as fixed-point
    s = f"{value:.{max_decimals}f}"
    # strip trailing zeros and then a trailing dot if present
    s = s.rstrip("0").rstrip(".")
    return s

# endregion

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(icon_path))
    app.setStyle('Fusion')

    # Splash screen
    splash_pix = QPixmap(400, 200)
    splash_pix.fill(app.palette().color(QPalette.Window))
    splash = QSplashScreen(splash_pix)
    splash.showMessage(f"{APP_NAME} Loading...\n\n v{__version__}", Qt.AlignCenter | Qt.AlignCenter, app.palette().color(QPalette.Text))
    splash.show()

    app.processEvents()
    win = MainWindow()
    win.show()
    splash.finish(win)    # Close splash when ready

    sys.exit(app.exec())
