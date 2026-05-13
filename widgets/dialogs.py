import datetime
import json
import logging
import os
import re
import subprocess
import sys
import cv2
import numpy as np
import threading
import traceback
from fractions import Fraction
from logging.handlers import RotatingFileHandler
from pathlib import Path, PurePath
from statistics import median
import urllib.request

# Third-Party Imports
from PySide6.QtCore import QSettings, Qt, QThread, Signal, QTimer, QPoint, QRect, QPointF, QRectF, QUrl
from PySide6.QtGui import QBrush, QColor, QIcon, QPalette, QPixmap, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QDialog, QFileDialog, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar,
    QPushButton, QSizePolicy, QSplashScreen, QStyle, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget, QGraphicsView, QGraphicsScene, QGraphicsRectItem, QListWidget
)
from tifffile import TiffFile, imread
from widgets.file_scanning import find_file_on_network_drives, find_original_file, _look_for_csv
from widgets.resources import setup_logging, icon_path, FFMPEG, FFPROBE, get_log_file
from widgets.inspecting import get_video_info
from config import *
import certifi
import ssl


logger = logging.getLogger(__name__)

log_dir = get_log_file().parent  # or however you store LOGFILE
log_dir_url = QUrl.fromLocalFile(str(log_dir)).toString()


instructions = (
    "Use this tool to reduce the size of AVI/TIF files and save storage space.<br><br>"

    "<b>Usage Instructions:</b><br>"
    "1. Compress large AVI/TIF files for archiving on server<br>"
    "2. Decompress files on your computer for use with ImageJ<br>"

    "<b>File Functions:</b><br>"
    "<i>Compress:</i> Compress video file.<br>"
    "<i>Validate:</i> Compare AVI/TIF and MKV files to verify no data loss.<br>"
    "<i>Decompress:</i> Convert compressed video back to AVI/TIF.<br>"
    "<i>Inspect:</i> Shows the pixel values for the top left 10x10 pixels of the first frame."
    "<i>Crop:</i> Loads frame for the user to draw a crop box. Creates a new cropped video.<br><br>"

    "• Can run directly on server folders/files, but will be slower.<br>"
    "• Compressed videos cannot be opened directly in ImageJ.<br>"
    "• VLC can play the MKV files: <a href='https://www.videolan.org/vlc/'>VLC Video Player</a><br>"
    "• This app will never delete data, it will only create new files.<br>"
    "• Validation compares videos frame by frame for any pixel mismatches.<br><br>"

    "This program takes files in raw/decompressed format, and uses FFMPEG<br>"
    "to convert to .MKV with the FFV1 codec.<br>"
    "FFV1 is a modern lossless codec which is used by the Library of Congress,<br>"
    "the US National Archives, and universities for video archiving.<br>"
    "FFV1 ensures no data loss, so the MKV files can be decompressed back to the<br>"
    "original AVI/TIF if necessary for compatibility on older software or hardware.<br><br>"

    f"v{version}<br><br>"
    
    "<b>Issues</b><br>"
    "If you encounter any issues or bugs with the program, contact the developer.<br>"
    "Be sure to include the app version and the log file.<br>"
    f"<a href='{log_dir_url}'>Open log folder</a><br>"
    "<i>Windows:</i> C:/Users/(username)/AppData/Local/VideoVise/videovise_debug.log<br>"
    "<i>Mac:</i> ~/Library/Logs/VideoVise/videovise_debug.log<br><br>"


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
        try:
            if self.fp.suffix.lower() in (".tif", ".tiff"):
                # Load the array - we remove the .astype("uint8") to allow 16-bit
                arr = imread(self.fp, key=0)
                if arr.ndim == 2:
                    frame = arr.tolist()
                else:
                    frame = [[tuple(pixel) for pixel in row] for row in arr.tolist()]

            else:
                # Determine if we should ask for 16-bit or 8-bit output
                is_16bit = "16" in self.pix_fmt or "48" in self.pix_fmt

                # Use 'pgm' for grayscale, 'ppm' for RGB
                # This is the "Intuitive" approach: FFmpeg handles all color math
                is_gray = self.pix_fmt.lower().startswith("gray")
                vcodec = "pgm" if is_gray else "ppm"

                cmd = [
                    str(FFMPEG), "-hide_banner", "-loglevel", "error",
                    "-i", str(self.fp),
                    "-frames:v", "1",
                    "-f", "image2pipe",
                    "-vcodec", vcodec,
                    "pipe:1"
                ]

                # Use communicate to prevent the deadlock/hang you saw earlier
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout_data, stderr_data = proc.communicate()

                if proc.returncode != 0:
                    print(f"FFmpeg Error: {stderr_data.decode()}")
                    return

                # --- DYNAMIC NETPBM PARSING ---
                # We don't hardcode stdout.readline() because communicate() gives us one big blob
                lines = stdout_data.split(b'\n', 3)
                magic = lines[0].strip()  # P5 (Gray) or P6 (RGB)

                # Handle potential comments in header
                header_offset = 1
                if lines[header_offset].startswith(b'#'):
                    header_offset += 1

                dims = lines[header_offset].split()
                w, h = int(dims[0]), int(dims[1])
                maxval = int(lines[header_offset + 1])

                # Find where the raw data actually starts
                # Netpbm headers end with a single whitespace character after MaxVal
                data_start = stdout_data.find(str(maxval).encode()) + len(str(maxval)) + 1
                raw_bytes = stdout_data[data_start:]

                # Interpret bytes based on MaxVal (8-bit if 255, 16-bit if 65535)
                dtype = np.dtype('>u2') if maxval > 255 else np.uint8  # Netpbm 16-bit is Big-Endian
                flat_arr = np.frombuffer(raw_bytes, dtype=dtype)

                # Reshape and convert to list
                if magic == b'P5':  # Grayscale
                    arr = flat_arr.reshape((h, w))
                    frame = arr.tolist()
                else:  # RGB (P6)
                    arr = flat_arr.reshape((h, w, 3))
                    frame = [[tuple(pixel) for pixel in row] for row in arr.tolist()]

            # Slice to top-left 10×10 for the UI
            h_slice = min(10, self.h)
            w_slice = min(10, self.w)
            small_frame = [r[:w_slice] for r in frame[:h_slice]]

            self.finished.emit(self.fp, small_frame)

        except Exception as e:
            import traceback
            traceback.print_exc()

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

            ssl_context = ssl.create_default_context(cafile=certifi.where())

            with urllib.request.urlopen(REPO_URL, context=ssl_context, timeout=5) as response:
                # GitHub redirects 'latest' to the specific tag URL.
                # Example: .../releases/tag/v0.1
                final_url = response.geturl()

                # Extract the tag (the last part of the URL)
                latest_tag = final_url.split('/')[-1]

                if self._is_version_newer(latest_tag, version):
                    self.update_available.emit(latest_tag)
        except Exception as e:
            # Log silently; we don't want to annoy the user if they are offline
            logger.warning(f"Update check failed: {e}")

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
            logger.warning(f"Could not parse version tags for comparison: {remote_tag} vs {current_version}")
            return False


class CropLabel(QGraphicsView):
    """
        A QGraphicsView canvas for drawing a crop box on a zoomable image.
        Automatically handles scroll bars, coordinate transformations, and zooming.
        """
    crop_completed = Signal(list)  # Emits: [x, y, width, height]

    def __init__(self, parent=None):
        super().__init__(parent)

        # --- Scene and View Setup ---
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)

        self.setBackgroundBrush(QColor("#222222"))

        # --- State Variables ---
        self._image_item = None
        self._crop_rect_item = None
        self._start_point = None

        # --- Drawing Styles ---
        self._pen = QPen(QColor(0, 255, 255), 2)
        self._pen.setCosmetic(True)  # Keeps the line 2px thick regardless of zoom

    def set_background(self, pixmap: QPixmap):
        """Clears the scene and sets a new background image."""
        self.clear()
        self._image_item = self._scene.addPixmap(pixmap)
        self.reset_view()

    def reset_view(self):
        """Resets the view to fit the entire image within the viewport."""
        if self._image_item:
            self.fitInView(self._image_item, Qt.KeepAspectRatio)

    def clear(self):
        """Clears all items from the canvas."""
        self._scene.clear()
        self._image_item = None
        self._crop_rect_item = None
        self._start_point = None

    def _clamp_to_image(self, pos: QPointF) -> QPointF:
        """Forces the coordinates to stay strictly within the image boundaries."""
        if not self._image_item:
            return pos
        rect = self._image_item.boundingRect()
        x = max(rect.left(), min(pos.x(), rect.right()))
        y = max(rect.top(), min(pos.y(), rect.bottom()))
        return QPointF(x, y)

    def _zoom(self, factor):
        """Applies a zoom factor, centered on the mouse cursor."""
        if self._image_item is None:
            return
        if factor < 1.0:
            h_bar = self.horizontalScrollBar()
            v_bar = self.verticalScrollBar()
            if h_bar.maximum() <= 0 and v_bar.maximum() <= 0:
                return
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.scale(factor, factor)
        self.setTransformationAnchor(QGraphicsView.NoAnchor)

    # ——————————————
    # Mouse & Key Events

    def wheelEvent(self, event):
        """Handles zooming and panning via the mouse wheel."""
        if self._image_item is None:
            return

        angle = event.angleDelta().y()

        if event.modifiers() == Qt.ControlModifier:
            if angle > 0:
                self._zoom(1.15)
            else:
                self._zoom(1 / 1.15)
        elif event.modifiers() == Qt.ShiftModifier:
            h_bar = self.horizontalScrollBar()
            h_bar.setValue(h_bar.value() - angle)
        else:
            super().wheelEvent(event)

    def mouseDoubleClickEvent(self, event):
        """Resets the view on double-click."""
        if event.button() == Qt.LeftButton:
            self.reset_view()
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):
        """Starts drawing the crop box."""
        if event.button() != Qt.LeftButton or self._image_item is None:
            super().mousePressEvent(event)
            return

        # 1. Map viewport click to native image coordinates
        scene_pos = self.mapToScene(event.pos())
        self._start_point = self._clamp_to_image(scene_pos)

        # 2. Create or reset the crop rectangle
        if self._crop_rect_item is None:
            self._crop_rect_item = QGraphicsRectItem(QRectF(self._start_point, self._start_point))
            self._crop_rect_item.setPen(self._pen)
            self._scene.addItem(self._crop_rect_item)
        else:
            self._crop_rect_item.setRect(QRectF(self._start_point, self._start_point))

    def mouseMoveEvent(self, event):
        """Updates the crop box as the user drags."""
        if self._start_point and self._crop_rect_item:
            scene_pos = self.mapToScene(event.pos())
            current_point = self._clamp_to_image(scene_pos)

            # .normalized() safely handles drawing backwards/upwards
            new_rect = QRectF(self._start_point, current_point).normalized()
            self._crop_rect_item.setRect(new_rect)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Finalizes the crop box and extracts the native coordinates."""
        if event.button() == Qt.LeftButton and self._start_point and self._crop_rect_item:
            scene_pos = self.mapToScene(event.pos())
            end_point = self._clamp_to_image(scene_pos)

            final_rect = QRectF(self._start_point, end_point).normalized()
            self._crop_rect_item.setRect(final_rect)
            self._start_point = None

            # Extract exact 1:1 image coordinates
            self.crop_completed.emit([
                int(final_rect.x()),
                int(final_rect.y()),
                int(final_rect.width()),
                int(final_rect.height())
            ])

        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        """Handles keyboard shortcuts."""
        if event.key() in (Qt.Key_Equal, Qt.Key_Plus):
            self._zoom(1.5)
            event.accept()
        elif event.key() in (Qt.Key_Minus, Qt.Key_Underscore):
            self._zoom(1 / 1.5)
            event.accept()
        else:
            super().keyPressEvent(event)

    def get_crop_data(self):
        """Allows external classes to fetch the coordinates manually."""
        if self._crop_rect_item:
            r = self._crop_rect_item.rect()
            if r.width() > 0 and r.height() > 0:
                return int(r.x()), int(r.y()), int(r.width()), int(r.height())
        return None


class CropDialog(QDialog):
    def __init__(self, video_path: Path, total_frames: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Select Crop Region - {video_path.name}")
        self.resize(1000, 1000)  # Give the dialog a nice default size

        self.video_path = video_path
        self.total_frames = total_frames
        self.crop_coords = None

        self.init_ui()
        self.load_frame(total_frames)

    def init_ui(self):
        self.layout = QVBoxLayout(self)

        # Instructions
        self.info_label = QLabel(
            "Click and drag to draw a crop box. "
            "Use <b>Ctrl + Mouse Wheel</b> to zoom. Double-click to reset view."
        )
        self.layout.addWidget(self.info_label)

        # The interactive QGraphicsView canvas
        self.canvas = CropLabel()
        self.layout.addWidget(self.canvas)

        # Buttons
        self.btn_layout = QHBoxLayout()
        self.btn_confirm = QPushButton("Confirm")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_confirm.setDefault(True)
        self.btn_confirm.setAutoDefault(True)
        self.btn_cancel.setDefault(False)
        self.btn_cancel.setAutoDefault(False)

        self.btn_layout.addStretch()
        self.btn_layout.addWidget(self.btn_cancel)
        self.btn_layout.addWidget(self.btn_confirm)
        self.layout.addLayout(self.btn_layout)

        # Connections
        self.btn_confirm.clicked.connect(self.accept_crop)
        self.btn_cancel.clicked.connect(self.reject)

    def load_frame(self, total_frames: int):
        target_frame_index = 0
        csv_path = _look_for_csv(self.video_path)

        if csv_path:
            try:
                # names=True reads the header row; dtype=float converts the data
                data = np.genfromtxt(csv_path, delimiter=',', names=True, encoding='utf-8-sig', dtype=float)

                # Validate parity/alignment
                if len(data) == self.total_frames:
                    # Check if 'distance' column exists
                    if data.dtype.names and 'distance' in data.dtype.names:
                        target_frame_index = int(np.argmax(data['distance']))
                        logger.info(f"Target frame found at index {target_frame_index} (max distance).")
                    # Check if 'pressure' column exists
                    elif data.dtype.names and 'pressure' in data.dtype.names:
                        target_frame_index = int(np.argmax(data['pressure']))
                        logger.info(f"Target frame found at index {target_frame_index} (max pressure).")
                    else:
                        logger.warning(
                            "CSV loaded, but 'distance' or 'pressure' column not found. Falling back to frame 0.")
                else:
                    logger.warning(
                        f"Row count mismatch: CSV ({len(data)}) vs Video ({self.total_frames}). Falling back to frame 0.")
            except Exception as e:
                logger.error(f"Failed to process CSV for frame selection: {e}. Falling back to frame 0.")

        # Open video and navigate to the target frame
        cap = cv2.VideoCapture(str(self.video_path))

        if target_frame_index > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_index)
            self.info_label.setText(
                "Located maximum deformation frame. Click and drag to draw a crop box. "
                "<b>Ctrl + Wheel</b> to zoom."
            )
            ret, frame = cap.read()
        else:
            # Average 3 frames to get a good composite if no CSV is found
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames / 4))
            ret1, frame1 = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames / 2))
            ret2, frame2 = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames - 1))
            ret3, frame3 = cap.read()

            if ret1 and ret2 and ret3:
                frame = (
                                frame1.astype(np.float32) +
                                frame2.astype(np.float32) +
                                frame3.astype(np.float32)
                        ) / 3.0
                frame = np.clip(frame, 0, 255).astype(np.uint8)
                ret = True
            else:
                ret = False

        cap.release()

        if ret:
            # OpenCV loads in BGR, PySide needs RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)

            # Pass the full-resolution pixmap directly to the canvas
            self.canvas.set_background(pixmap)
        else:
            self.info_label.setText("Failed to load video frame.")
            self.btn_confirm.setEnabled(False)

    def accept_crop(self):
        raw_coords = self.canvas.get_crop_data()

        # Scenario 1: The user clicked "Confirm" without drawing a box,
        # or accidentally clicked without dragging (0x0 area).
        if not raw_coords:
            QMessageBox.warning(
                self,
                "No Crop Selected",
                "No valid crop area detected. Please click and drag to draw a box on the image."
            )
            return  # Halts the process, keeps dialog open

        # Scenario 2: A box was drawn, but it's suspiciously small.
        x, y, w, h = raw_coords
        if w <= 10 or h <= 10:
            QMessageBox.warning(
                self,
                "Crop Too Small",
                "You submitted a very small window for cropping (10 pixels or less). Please draw a larger box."
            )
            return  # Halts the process, keeps dialog open

        # Scenario 3: Valid crop box. Proceed with closing.
        self.crop_coords = raw_coords
        self.accept()

    def showEvent(self, event):
        """
        Triggered right when the dialog appears on screen.
        This ensures the layout geometry is fully calculated before we attempt to fit the image.
        """
        super().showEvent(event)
        self.canvas.reset_view()

    def resizeEvent(self, event):
        """Keeps the image fitted to the window if the user resizes the dialog."""
        super().resizeEvent(event)
        self.canvas.reset_view()


class CompressibleResultsDialog(QDialog):
    # Optional: emit the chosen folder so the main window can load it immediately
    folder_selected = Signal(str)

    def __init__(self, directories, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Compressible Folders Found")
        self.resize(650, 400)

        layout = QVBoxLayout(self)

        label = QLabel(f"Found {len(directories)} folders containing .tif files or .avi files larger than 1GB. "
                       f"<br><i>Double-click a folder to load it into the main app.</i>")
        layout.addWidget(label)

        self.list_widget = QListWidget()
        self.list_widget.addItems(directories)
        layout.addWidget(self.list_widget)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)

        # Connect double-click to our custom action
        self.list_widget.itemDoubleClicked.connect(self.on_item_double_clicked)

    def on_item_double_clicked(self, item):
        self.folder_selected.emit(item.text())
        self.accept()  # Close the dialog automatically