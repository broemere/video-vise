import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path
import gc

# Third-Party Imports
from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QIcon, QPalette, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QFileDialog, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar,
    QPushButton, QSplashScreen, QStyle, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget
)
from widgets.file_scanning import find_file_on_network_drives, find_original_file, get_files, load_storage_jsonl
from widgets.resources import setup_logging, icon_path
from widgets.inspecting import get_video_info, sample_tail_zeros
from config import *
from widgets.dialogs import PixelDialog, PixelLoaderThread, instructions, UpdateChecker
from widgets.converter import FFmpegConverter
from widgets.validator import FrameValidator
from widgets.tracker import StorageTracker


logger = setup_logging()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.settings = QSettings(ORG, APP_NAME.replace(" ",""))
        self.worker = None
        self.batch_queue = []
        self.total_tasks = 0
        self.lossless_results = {}
        self.durations = {}
        self.frames = {}
        self.codecs = {}
        self.cancel_requested = False
        self.num_compress = 0
        self.tracker = StorageTracker()
        self.init_ui()
        geometry = self.settings.value("windowGeometry")
        if geometry:
            self.restoreGeometry(geometry)  # Restore size AND position
        else:
            self.resize(1300, 750)  # Fallback to default size
        last = self.settings.value("lastFolder", "")
        if last and os.path.isdir(last):
            self.path_edit.setText(last)
            self.update_table(last)

        self.update_checker = UpdateChecker()
        self.update_checker.update_available.connect(self.on_update_available)
        self.update_checker.start()  # Runs in background, won't freeze app

    def closeEvent(self, event):
        """
        Called when the window is closing.
        """
        self.settings.setValue("windowGeometry", self.saveGeometry())
        self.settings.setValue("lastFolder", self.path_edit.text())
        super().closeEvent(event)

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
        brightness = (base.red() * 0.299 + base.green() * 0.587 + base.blue() * 0.114)  # Calculate perceived brightness
        if brightness < 128:
            alt = base.lighter(50)  # Dark background: lighten alternate rows
        else:
            alt = base.darker(110)  # Light background: darken alternate rows
        pal.setColor(QPalette.AlternateBase, alt)
        self.table.setPalette(pal)
        cols = [
            " ", "Filename","Size (GB)","Created","Duration","Codec","PixelFmt",
            "Resolution","Tag","Color/Gray","FPS","Frames","Compress","Validate","Lossless", "Relative Size", "Uncompress", "Inspect"
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
        if self.tracker.track_storage:
            self.gb_saved.setText(f"{self.tracker.get_gb_saved()} GB saved")
        self.progress = QProgressBar(); self.progress.setRange(0, 100); self.progress.setTextVisible(False)
        self.progress.setStyleSheet("QProgressBar{border:1px solid gray; border-radius:5px;}"+
                                   "QProgressBar::chunk{background-color:#4CAF50}")
        self.progress.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.status); status_layout.addStretch(); status_layout.addWidget(self.gb_saved)
        layout.addLayout(status_layout)
        layout.addWidget(self.progress)

    def flash_window(self):
        QApplication.alert(self, 0) # default platform timeout

    def show_help(self):
        msg = QMessageBox(self)
        msg.setWindowTitle("Help")
        msg.setTextFormat(Qt.RichText)
        msg.setText(instructions)
        # Allow clicking links
        msg.setTextInteractionFlags(Qt.TextBrowserInteraction)
        msg.setStandardButtons(QMessageBox.Close)
        msg.setDefaultButton(QMessageBox.Close)
        msg.exec()

    def browse_folder(self):
        last = self.settings.value("lastFolder", type=str)
        if last:
            parent = Path(last).parent  # Always try to use the parent of the last folder
            if parent.exists() and parent.is_dir():
                initial_dir = str(parent)
            else:
                initial_dir = str(Path.home())
        else:
            initial_dir = str(Path.home())

        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder",
            initial_dir,
            QFileDialog.ShowDirsOnly | QFileDialog.ReadOnly
        )

        if folder:
            self.path_edit.setText(folder)
            self.settings.setValue("lastFolder", folder)
            self.lossless_results.clear()
            self.update_table(folder)

    def update_progress(self, val: int):
        self.progress.setValue(val)

    def update_table(self, folder: str):
        """Updates the table with files, grouping them by subdirectory."""
        if not folder or not os.path.isdir(folder):
            return

        base_folder = Path(folder)
        self.status.setText("Refreshing...")
        self.progress.setValue(0)
        self.table.setRowCount(0)
        QApplication.processEvents()
        files = get_files(folder)  #  gets files recursively

        # --- Group files by directory ---
        files_by_dir = {}
        for fp in files:
            # Get the parent directory
            parent_dir = fp.parent
            if parent_dir not in files_by_dir:
                files_by_dir[parent_dir] = []
            files_by_dir[parent_dir].append(fp)

        # --- Create the list of items to render (files + dir headers) ---
        all_items_to_render = []

        # 1. Add top-level files first (if any)
        if base_folder in files_by_dir:
            # Sort top-level files
            top_level_files = sorted(files_by_dir[base_folder], key=lambda p: p.name.lower())
            all_items_to_render.extend(top_level_files)
            del files_by_dir[base_folder]  # Remove from dict so no repeats

        # 2. Add subdirectories and their files
        # Sort directories by path for consistent order
        sorted_dirs = sorted(files_by_dir.keys(), key=str)

        for dir_path in sorted_dirs:
            # Add the directory itself as a "header" row
            relative_dir = dir_path.relative_to(base_folder)
            all_items_to_render.append(str(relative_dir))
            # Sort files within this directory
            files_in_dir = sorted(files_by_dir[dir_path], key=lambda p: p.name.lower())
            all_items_to_render.extend(files_in_dir)

        # --- Populate Table ---
        self.table.setRowCount(len(all_items_to_render))
        QApplication.processEvents()

        # Keep track of file progress (not row progress)
        file_progress_count = 0
        num_files_total = len(files)

        # Get style for folder icon
        folder_icon = self.style().standardIcon(QStyle.SP_DirIcon)
        # Define a background color for directory rows
        dir_bg_color = QColor("#eeeeee")
        warning_color = QColor("#f54927")
        pal = self.table.palette()
        if pal.color(QPalette.Base).lightness() < 128:  # Dark theme
            dir_bg_color = QColor("#424242")
        for r, item in enumerate(all_items_to_render):
            # --- Handle Directory Header Rows ---
            if isinstance(item, str):
                dir_path_str = item

                # Column 0: Color swatch (empty)
                color_item = QTableWidgetItem()
                color_item.setBackground(dir_bg_color)
                color_item.setFlags(Qt.ItemIsEnabled)  # Not selectable
                self.table.setItem(r, 0, color_item)

                # Column 1: Directory Name
                dir_item = QTableWidgetItem(f" {dir_path_str}{os.sep}")  # Add trailing slash
                dir_item.setIcon(folder_icon)
                dir_item.setBackground(dir_bg_color)
                dir_item.setFlags(Qt.ItemIsEnabled)
                dir_item.setForeground(pal.color(QPalette.Text))
                self.table.setItem(r, 1, dir_item)

                # Span the directory name across remaining columns
                self.table.setSpan(r, 1, 1, self.table.columnCount() - 1)
                continue  # Skip to next row

            # --- Handle File Rows (item is a Path object) ---
            fp = item

            # Use full path string as key to prevent name collisions
            fp_key = str(fp)

            # Get relative path for display in filename column
            display_name = fp.name  # Fallback

            info = get_video_info(fp)
            codec = info.get("codec_name", "-")
            if codec == "Unknown":
                codec = "-"
            self.codecs[fp_key] = codec  # Cache global

            # — Column 0: Color swatch
            color_item = QTableWidgetItem()
            hexcol = EXT_COLOR.get(fp.suffix.lower())
            if hexcol:
                color_item.setBackground(QBrush(QColor(hexcol)))
            self.table.setItem(r, 0, color_item)

            # — Column 1: Filename (now relative path)
            display_item = QTableWidgetItem(display_name)
            display_item.setData(Qt.UserRole, fp_key) # <-- ADD THIS LINE
            self.table.setItem(r, 1, display_item)

            # — Column 2: Size
            size_logical = fp.stat().st_size
            size_gb = size_logical / 1024 ** 3
            size_item = QTableWidgetItem(f"{size_gb:.2f}")
            size_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(r, 2, size_item)

            # — Column 3: Created
            created = datetime.datetime.fromtimestamp(fp.stat().st_ctime)
            self.table.setItem(r, 3, QTableWidgetItem(created.strftime("%Y-%m-%d %H:%M:%S")))

            # — Column 4: Duration
            dur = float(info.get("duration", "0"))
            # *** Use fp_key for dictionary ***
            frames = int(info.get("frames", 0))  # Capture frames locally
            self.frames[fp_key] = frames  # Cache global
            #print(fp_key, info.get("frames", 0), print(dur))
            dur_str = str(datetime.timedelta(seconds=int(dur))) if dur else "-"
            self.table.setItem(r, 4, QTableWidgetItem(dur_str))

            # — Column 5-10: Video Info
            self.table.setItem(r, 5, QTableWidgetItem(codec))
            pix = info.get("pix_fmt", "-")
            if pix == "Unknown":
                pix = "-"
            self.table.setItem(r, 6, QTableWidgetItem(pix))
            w = info.get("width", "0")
            h = info.get("height", "0")
            res = f"{w}x{h}" if w and h else "N/A"
            self.table.setItem(r, 7, QTableWidgetItem(res))
            tag = info.get("codec_tag_string", "")
            if tag == "Unknown":
                tag = "-"
            if not re.match(r"^[A-Za-z0-9_.]+$", tag): tag = "-"
            self.table.setItem(r, 8, QTableWidgetItem(tag))
            color = "-"
            if "gray" in pix.lower():
                color = "Gray"
            elif "rgb" in pix.lower():
                color = "RGB"
            elif "bgr" in pix.lower():
                color = "BGR"
            elif "yuv" in pix.lower():
                color = "YUV"
            self.table.setItem(r, 9, QTableWidgetItem(color))
            fps = str(int(round(float(info.get("fps", "0")))))
            self.table.setItem(r, 10, QTableWidgetItem(fps))

            # — Column 11: Frames (NEW COLUMN)
            frames_item = QTableWidgetItem(str(frames))
            #frames_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(r, 11, frames_item)


            # — Column 12: Compress Button
            if codec != "ffv1":
                # Determine widget based on status
                widget = None

                # Check for corruption first (Frames=0 or Duration/Res=0)
                is_corrupt = (frames == 0) or (dur == 0 and float(w) == 0 and float(h) == 0)

                if is_corrupt:
                    widget = QLabel("Corrupt")
                    widget.setStyleSheet("color: red; font-weight: bold;")
                    widget.setAlignment(Qt.AlignCenter)
                elif codec == "mjpeg":
                    widget = QLabel("Compressed")
                    widget.setAlignment(Qt.AlignCenter)
                    widget.setStyleSheet("color: #777; font-style: italic;")  # Optional styling
                elif frames == 1:
                    widget = QLabel("Single Frame")
                    widget.setAlignment(Qt.AlignCenter)
                    widget.setStyleSheet("color: #777; font-style: italic;")  # Optional styling
                else:
                    # Normal Compress Button
                    widget = QPushButton("Compress")
                    widget.clicked.connect(lambda _, p=fp: self.start_convert(p, "compress"))

                self.table.setCellWidget(r, 12, widget)

            # — Column 13 & 16: Validate & Uncompress Buttons
            if codec == "ffv1":
                # 1. Check for original source file
                orig_fp = find_original_file(fp, True)

                # — Column 13: Validate (Only if source exists)
                if orig_fp is not None:
                    btnv = QPushButton("Validate")
                    btnv.clicked.connect(lambda _, p=fp: self.start_validate(p))
                    self.table.setCellWidget(r, 13, btnv)
                else:
                    lbl = QLabel("Source not found")
                    lbl.setAlignment(Qt.AlignCenter)
                    lbl.setStyleSheet("color: #777; font-style: italic;")
                    self.table.setCellWidget(r, 13, lbl)

                # — Column 16: Uncompress (Always available for FFV1)
                btnd = QPushButton("Uncompress")
                btnd.clicked.connect(lambda _, p=fp: self.start_convert(p, "uncompress"))
                self.table.setCellWidget(r, 16, btnd)

            # — Column 14: Lossless Icon
            label = QLabel()
            label.setAlignment(Qt.AlignCenter)
            # *** Use fp_key for dictionary ***
            if fp_key in self.lossless_results:
                ok = self.lossless_results[fp_key]
                icon = self.style().standardIcon(QStyle.SP_DialogApplyButton if ok else QStyle.SP_DialogCancelButton)
                label.setPixmap(icon.pixmap(24, 24))
            if codec == "rawvideo":
                s = sample_tail_zeros(fp)
                #print(fp.name, s)
                if s["tail_zero"] >= 0.999 and s["near_tail_zero"] >= 0.999 and s["headish_zero"] <= 0.95:
                # if size_physical / size_logical < 0.95:
                    icon = self.style().standardIcon(QStyle.SP_MessageBoxWarning)
                    label.setPixmap(icon.pixmap(24, 24))
                    label.setToolTip("File is missing data/incomplete. Compressing will process all viable data.")
            self.table.setCellWidget(r, 14, label)

            # — Column 15: Relative Size
            pct_item = QTableWidgetItem()
            pct_str = ""
            if fp.suffix.lower() == ".mkv":
                # Check for source files in the *same directory*
                avi_fp = fp.with_suffix(".avi")
                tif_fp = fp.with_suffix(".tif")
                tiff_fp = fp.with_suffix(".tiff")

                source_fp = None
                if avi_fp.exists():
                    source_fp = avi_fp
                elif tif_fp.exists():
                    source_fp = tif_fp
                elif tiff_fp.exists():
                    source_fp = tiff_fp

                if source_fp:
                    source_size = source_fp.stat().st_size
                    pct = (size_logical / source_size * 100) if source_size else 0
                    pct_str = f"{pct:.0f}%"

                    # ---- NEW: Frame mismatch warning shading for Column 11 ----
                    source_key = str(source_fp)
                    src_frames = self.frames.get(source_key)

                    # Only warn if we actually know the source frame count
                    if isinstance(src_frames, int) and src_frames > 0:
                        if frames != src_frames:
                            frames_item.setBackground(QBrush(warning_color))
                            frames_item.setToolTip(
                                f"Frame mismatch vs source:\n"
                                f"Source: {source_fp.name} = {src_frames}\n"
                                f"Output: {fp.name} = {frames}"
                            )
                else:
                    pct_str = "N/A"

            pct_item.setText(pct_str)
            pct_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(r, 15, pct_item)

            # — Column 17: Inspect Button
            btni = QPushButton()
            btni.setIcon(self.style().standardIcon(QStyle.SP_FileDialogContentsView))
            btni.setProperty("fp", fp)
            btni.setProperty("width", w)
            btni.setProperty("height", h)
            btni.setProperty("pix_fmt", pix)
            btni.clicked.connect(self.inspect_pixels)
            btni.setStyleSheet(
                "QPushButton { padding-top: 2px; padding-bottom: 2px; padding-left: 4px; padding-right: 4px; }")
            self.table.setCellWidget(r, 17, btni)

            # Update progress based on files processed
            file_progress_count += 1
            if num_files_total > 0:
                self.progress.setValue(int(100 * (file_progress_count / num_files_total)))

        self.table.resizeColumnsToContents()
        # Ensure the spanned directory row also resizes
        for r in range(self.table.rowCount()):
            if self.table.columnSpan(r, 1) > 1:
                self.table.setRowHeight(r, self.table.rowHeight(r))  # Trigger redraw/resize

        self.status.setText("Ready")
        self.progress.setValue(0)
        if self.tracker.track_storage:
            self.gb_saved.setText(f"{int(self.tracker.get_gb_saved())} GB saved")
        QApplication.processEvents()

    def inspect_pixels(self):
        btn = self.sender()
        fp = btn.property("fp")
        w = btn.property("width")
        h = btn.property("height")
        pix_fmt = btn.property("pix_fmt")
        loader = PixelLoaderThread(fp, w, h, pix_fmt, parent=self)
        loader.finished.connect(self.on_pixels_ready)
        loader.finished.connect(loader.deleteLater)
        loader.start()

    def on_pixels_ready(self, fp, small_frame):
        dlg = PixelDialog(fp, small_frame, parent=self)
        dlg.show()

    def start_convert(self, fp: Path, mode: str):
        if str(fp) in self.frames:
            nframes = self.frames[str(fp)]
        else:
            logger.debug(f"Frame count not cached for {fp}, running inspection...")
            try:
                info = get_video_info(fp)
                nframes = info.get("frames", 0)
                self.frames[str(fp)] = nframes
            except Exception as e:
                logger.error(f"Failed to inspect frames: {e}", exc_info=True)
                nframes = 0

        if mode == "compress":
            out_fp = fp.with_name(fp.stem + ".mkv")
        else:
            out_fp = fp.with_name(fp.stem + "_RAW.avi")

        self.status.setText(f"{mode.title()}ing {fp}")
        self.progress.setValue(0)
        self.worker = FFmpegConverter(fp, out_fp, nframes, mode, self.tracker.track_storage)
        self.worker.progress.connect(self.update_progress)
        self.worker.result.connect(self.on_conversion_result)
        self.worker.finished.connect(lambda: self.update_table(self.path_edit.text()))
        self.worker.start()

    def on_conversion_result(self, key_name: str, result: int):
        if result > 0:
            self.tracker.filesizes_reduced = load_storage_jsonl(self.tracker.storage_path)
            if key_name not in self.tracker.filesizes_reduced.keys():
                self.tracker.filesizes_reduced[key_name] = result
                self.gb_saved.setText(f"{self.tracker.get_gb_saved()} GB saved")
                self.tracker.add_member_jsonl(key_name, result)

    def on_conversion_complete_and_continue(self, key_name: str, result: int):
        self.on_conversion_result(key_name, result)
        if self.worker:
            self.worker.quit()
            self.worker.wait()
        self.worker = None
        self.status.setText("Loading...")
        gc.collect()
        self._run_next_batch()
        # QTimer.singleShot(1000, self._run_next_batch) # cooldown

    def start_validate(self, fp: Path):
        orig_fp = find_original_file(fp)
        if orig_fp is not None:
            nframes = self.frames[str(orig_fp)]
            self.status.setText(f"Validating {fp} against {orig_fp}")
            self.progress.setValue(0)
            self.worker = FrameValidator(orig_fp, fp, nframes)
            self.worker.progress.connect(self.update_progress)
            self.worker.result.connect(self.on_validation_result)
            self.worker.start()
        else:
            logger.error(f"File does not exist: {orig_fp}")

    def on_validation_result(self, filename: str, ok: bool):
        self.lossless_results[filename] = ok
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 1)
            if item and item.data(Qt.UserRole) == filename:
                label = QLabel()
                label.setAlignment(Qt.AlignCenter)
                icon = self.style().standardIcon(QStyle.SP_DialogApplyButton if ok else QStyle.SP_DialogCancelButton)
                label.setPixmap(icon.pixmap(24, 24))
                self.table.setCellWidget(r, 14, label)
                QApplication.processEvents()
                break
        self.status.setText("Ready")
        self.progress.setValue(0)

    def on_validation_complete_and_continue(self, filename: str, ok: bool):
        self.on_validation_result(filename, ok)
        if self.worker:
            self.worker.quit()
            self.worker.wait()
        self.worker = None
        self.status.setText("Loading...")
        gc.collect()
        self._run_next_batch()

    # Batch processing logic
    def process_all_compress(self):
        folder = self.path_edit.text()
        files = get_files(folder, "compress")
        tasks = [(fp, 'compress') for fp in files if self.frames.get(str(fp), 0) > 1 and self.codecs.get(str(fp), "") != "mjpeg"]
        self.start_batch(tasks)

    def process_all_uncompress(self):
        folder = self.path_edit.text()
        files = get_files(folder, "validate")
        tasks = [(fp, 'uncompress') for fp in files]
        self.start_batch(tasks)

    def process_all_validate(self):
        folder = self.path_edit.text()
        files = get_files(folder, "validate")
        self.num_compress = 0
        tasks = [(fp, 'validate') for fp in files]
        self.start_batch(tasks)

    def process_all_compress_validate(self):
        folder = self.path_edit.text()
        files = get_files(folder, "compress")
        compress_tasks = [(fp, 'compress') for fp in files if self.frames.get(str(fp), 0) > 1 and self.codecs.get(str(fp), "") != "mjpeg"]
        self.num_compress = len(compress_tasks)
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
        #stack = "".join(traceback.format_stack())  # Only for serious worker stack debugging
        #current = threading.current_thread().name
        #logger.debug(f"_run_next_batch ENTERED (thread={current}). Full call stack:\n{stack}")

        if self.cancel_requested:
            logger.debug("_run_next_batch: cancel_requested=True → clearing queue and returning")
            self.batch_queue.clear()
            self.status.setText("Canceled")
            self.progress.setValue(0)
            self.update_table(self.path_edit.text())
            self.btn_cancel.setEnabled(False)
            self.flash_window()
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
        logger.debug(f"_run_next_batch: dispatching task {idx}/{self.total_tasks}: {mode} '{fp}'")
        self.progress.setValue(0)
        self.status.setText(f"{mode.title() if mode.endswith('compress') else mode.title()[:-1]}ing {fp} ({idx}/{self.total_tasks})")
        worker = None
        nframes = 0
        try:
            nframes = self.frames[str(fp)]
        except:
            logger.warning("Could not find frame count for "+str(fp))

        # Create new worker
        if mode in ("compress", "uncompress"):
            if nframes > 1:
                if mode == "compress":
                    out_fp = fp.with_suffix(".mkv")
                else:
                    out_fp = fp.with_name(fp.stem + "_RAW.avi")
                #out_fp = (fp.with_suffix(".mkv") if mode == "compress" else fp.with_suffix(".avi"))
                worker = FFmpegConverter(fp, out_fp, nframes, mode, self.tracker.track_storage)
                logger.debug(f"_run_next_batch: created FFmpegConverter for {fp} → out {out_fp}")
                worker.result.connect(self.on_conversion_complete_and_continue)
                #worker.finished.connect(self._run_next_batch)
        else:  # mode == "validate"
            orig = find_original_file(fp)
            if orig is None:
                logger.warning(f"Could not find original file for '{fp.name}'. Skipping validation.")
            else:
                nframes = self.frames[str(orig)]
                if nframes > 1:
                    worker = FrameValidator(orig, fp, nframes)
                    logger.debug(f"_run_next_batch: created FrameValidator for {fp}")
                    worker.result.connect(self.on_validation_complete_and_continue)
                    #worker.finished.connect(self._run_next_batch)

        # Hold a reference so it won’t be garbage‐collected
        if worker:
            self.worker = worker
            worker.progress.connect(self.progress.setValue)
            logger.debug("_run_next_batch: Starting worker thread")
            worker.start()
        else:
            # No worker was created (e.g., validation skipped because orig=None)
            # call _run_next_batch() again to continue the queue.
            logger.debug("_run_next_batch: No worker created, moving to next task.")
            # This posts the call to the event loop instead of calling it directly, which prevents a "recursion"
            # error if many files are skipped in a row.
            QTimer.singleShot(0, self._run_next_batch)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(icon_path))
    app.setStyle('Fusion')

    # Splash screen
    splash_pix = QPixmap(400, 200)
    splash_pix.fill(app.palette().color(QPalette.Window))
    splash = QSplashScreen(splash_pix)
    splash.showMessage(f"{APP_NAME} Loading...\n\n v{version}", Qt.AlignCenter | Qt.AlignCenter, app.palette().color(QPalette.Text))
    splash.show()

    app.processEvents()
    win = MainWindow()
    win.show()
    splash.finish(win)    # Close splash when ready

    sys.exit(app.exec())
