from pathlib import Path
import logging
from PySide6.QtCore import QSettings, Qt, QThread, Signal, QTimer
logger = logging.getLogger(__name__)
import re
import os
from widgets.inspecting import get_video_info, get_tiff_summary
from widgets.resources import FFMPEG, FFPROBE
import subprocess
from tifffile import TiffFile
from statistics import median
import json
from config import DEFAULT_FPS

class FFmpegConverter(QThread):
    progress = Signal(int)
    result = Signal(str, int)
    failed = Signal(Path, str)

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
        try:
            frame_re = re.compile(r"frame=\s*(\d+)\b")
            cpu_count = os.cpu_count() or 1
            threads = 3 if cpu_count == 4 else min(max(cpu_count - 2, 1), 16) # Changed from 4 to 2 for 6 core devices
            # Maybe improve this logic again for different size CPUs (small and large)
            slices = self.choose_slices(threads)

            if self._is_tiff():
                self._process_tiff(frame_re, threads, slices)
            else:
                self._process_video(frame_re, threads, slices)

            stat = self.input_path.stat()
            os.utime(self.output_path, (stat.st_atime, stat.st_mtime))

            # finish up
            self.progress.emit(100)
            if self.track:
                self._emit_size_diff()
        except Exception as e:
            # This block will catch any crashes and log the traceback
            logger.error(f"WORKER CRASHED processing {self.input_path}!", exc_info=True)
            self.failed.emit(self.input_path, str(e))

    def _is_tiff(self) -> bool:
        return self.input_path.suffix.lower() in [".tif", ".tiff"]

    # def _process_tiff(self, frame_re, threads: int, slices: int):
    #     # 1) extract metadata
    #     fps, pix_fmt, w, h = self._extract_tiff_metadata()
    #
    #     # 2) build & spawn ffmpeg
    #     cmd = [
    #         FFMPEG, "-y",
    #         "-f", "rawvideo",
    #         "-pix_fmt", pix_fmt,
    #         "-s", f"{w}x{h}",
    #         "-r", f"{fps:.3f}",
    #         "-i", "pipe:0",
    #         "-an",
    #         "-c:v", "ffv1",
    #         "-level", "3",
    #         "-threads", str(threads),
    #         "-coder", "1", "-context", "1",
    #         "-g", "1", "-slices", str(slices),
    #         "-slicecrc", "1",
    #         str(self.output_path)
    #     ]
    #     logger.info("Spawning FFmpeg process: " + " ".join(cmd))
    #     proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    #     logger.debug(f"Launched FFmpeg (PID={proc.pid})")
    #
    #     # 3) stream frames & report progress
    #     self._stream_tiff_frames(proc)
    #
    #     if proc.wait() != 0:
    #         raise RuntimeError("FFmpeg TIFF→video failed")

    def _process_tiff(self, frame_re, threads: int, slices: int):
        # --- 1. LEVERAGE THE INSPECTOR ---
        # No more manual parsing here. We trust the robust inspector logic.
        info = get_video_info(self.input_path)

        # Parse result (inspector returns strings, so we cast as needed)
        fps = float(info.get("raw_fps", DEFAULT_FPS))  # Use raw_fps if available for precision
        if fps == 0: fps = DEFAULT_FPS  # Fallback

        pix_fmt = info.get("pix_fmt", "gray")
        w = info.get("width", "0")
        h = info.get("height", "0")

        # --- 2. BUILD FFMPEG CMD ---
        cmd = [
            FFMPEG, "-y",
            "-f", "rawvideo",
            "-pix_fmt", pix_fmt,  # Inspector gives us the correct le/be format
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

        logger.info(f"Spawning FFmpeg TIF conversion: {w}x{h} @ {fps}fps fmt:{pix_fmt}")
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

        # --- 3. STREAM FRAMES ---
        self._stream_tiff_frames(proc)

        if proc.wait() != 0:
            raise RuntimeError("FFmpeg TIFF→video failed (non-zero exit code)")
    #
    # def _find_tiff_pages(self, tif):
    #     """
    #     Helper function to analyze the TIF and return a universal
    #     frame iterator and the total frame count, handling all
    #     three of your defined scenarios.
    #     """
    #
    #     # --- CASE 3: Multi-Series (Each series is one frame) ---
    #     # e.g., 7354 series, each is a single (2048, 2592) image
    #     if len(tif.series) > 1:
    #         total_frames = len(tif.series)
    #         frame_iterator = (series.asarray() for series in tif.series)
    #         return frame_iterator, total_frames
    #
    #     # --- CASES 1 & 2: Single-Series ---
    #     # If we are here, len(tif.series) == 1
    #
    #     main_series = tif.series[0]
    #
    #     if not main_series.pages:
    #         logger.warning(f"TIF file {self.input_path} series[0] has no pages.")
    #         # Check if the series as a whole is a stack
    #         stack_data = main_series.asarray()
    #         if stack_data.ndim > 2:  # It's a hyperstack (N, Y, X)
    #             total_frames = stack_data.shape[0]
    #             frame_iterator = (frame for frame in stack_data)
    #             return frame_iterator, total_frames
    #         elif stack_data.ndim == 2:  # It's a single 2D image
    #             total_frames = 1
    #             frame_iterator = [stack_data]
    #             return frame_iterator, total_frames
    #         else:
    #             return [], 0  # Empty/unsupported
    #
    #     n_pages_in_series = len(main_series.pages)
    #
    #     # --- CASE 1: Standard Multi-Page ---
    #     # (e.g., 1 series, 3475 pages, shape (3475, 800, 800))
    #     # Note: We check > 1 because Case 2 has 1 page.
    #     if n_pages_in_series > 1:
    #         total_frames = n_pages_in_series
    #         # This is your original, efficient iterator
    #         frame_iterator = (page.asarray() for page in main_series.pages)
    #         return frame_iterator, total_frames
    #
    #     # --- CASE 2: Single-Page Hyperstack (or just a single image) ---
    #     # (e.g., 1 series, 1 page, but series.shape is (3475, 800, 800))
    #     if n_pages_in_series == 1:
    #
    #         # Must load the whole page to know what's in it.
    #         # This is the big memory load for this file type.
    #         stack_data = main_series.asarray()
    #
    #         # Check if it's a stack (N, Y, X) or just one frame (Y, X)
    #         if stack_data.ndim > 2:
    #             # It's a Hyperstack (N, Y, X)
    #             total_frames = stack_data.shape[0]
    #             frame_iterator = (frame for frame in stack_data)
    #             return frame_iterator, total_frames
    #         else:
    #             # It's just a single 2D image (Y, X)
    #             total_frames = 1
    #             frame_iterator = [stack_data]  # A list containing the single frame
    #             return frame_iterator, total_frames
    #
    #     # Fallback for empty/weird files
    #     return [], 0
    #
    # def _get_metadata_source(self, tif):
    #     """
    #     Analyzes the TIF and returns the source for the first frame's
    #     metadata and the correct list of all pages/series.
    #
    #     Returns:
    #         (first_frame_source, all_pages_source)
    #     """
    #
    #     # --- CASE 3: Multi-Series (Each series is one frame) ---
    #     if len(tif.series) > 1:
    #         logger.info("Metadata Source: Case 3 (Multi-Series)")
    #         # The first frame is the first series
    #         # The list of all "pages" is the list of all series
    #         return tif.series[0], tif.series
    #
    #     # --- CASES 1 & 2: Single-Series ---
    #     elif len(tif.series) == 1:
    #         main_series = tif.series[0]
    #
    #         if not main_series.pages:
    #             raise ValueError("TIF file series[0] contains no pages.")
    #
    #         if len(main_series.pages) == 1:
    #             logger.info("Metadata Source: Case 2 (Hyperstack / Single Page)")
    #         else:
    #             logger.info("Metadata Source: Case 1 (Standard Multi-Page)")
    #
    #         # The first frame is the first page of the main series
    #         # The list of all "pages" is the list of pages in the main series
    #         return main_series.pages[0], main_series.pages
    #
    #     # Fallback for empty/unreadable TIF
    #     else:
    #         raise ValueError("TIF file contains no series.")
    #
    # def _extract_tiff_metadata(self):
    #     with TiffFile(self.input_path) as tif:
    #
    #         # --- Robustly find the first frame and page list ---
    #         first_page_source, all_pages = self._get_metadata_source(tif)
    #
    #         # --- Get pixel format, width, and height ---
    #
    #         # This .asarray() call is still the "danger zone" for Case 2.
    #         # It will load the entire hyperstack into memory.
    #         # But this is unavoidable if we need to check its dimensions.
    #         first_frame_data = first_page_source.asarray()
    #
    #         h, w = (0, 0)
    #
    #         if first_frame_data.ndim == 2:
    #             # Case 1 or 3: (Y, X)
    #             pix_fmt = "gray"
    #             h, w = first_frame_data.shape
    #             logger.info(f"Found grayscale tif data (shape: {first_frame_data.shape})")
    #
    #         elif first_frame_data.ndim == 3 and first_frame_data.shape[-1] == 3:
    #             # Case 1 or 3: (Y, X, 3)
    #             pix_fmt = "rgb24"
    #             h, w = first_frame_data.shape[:2]
    #             logger.info(f"Found rgb tif data (shape: {first_frame_data.shape})")
    #
    #         elif first_frame_data.ndim > 3 and first_frame_data.shape[-1] == 3:
    #             # Case 2: Hyperstack (Z, Y, X, 3)
    #             pix_fmt = "rgb24"
    #             h, w = first_frame_data.shape[-3:-1]  # Get Y and X
    #             logger.info(f"Found RGB hyperstack (shape: {first_frame_data.shape})")
    #
    #         elif first_frame_data.ndim > 2:
    #             # Case 2: Hyperstack (Z, Y, X)
    #             pix_fmt = "gray"
    #             h, w = first_frame_data.shape[-2:]  # Get Y and X
    #             logger.info(f"Found grayscale hyperstack (shape: {first_frame_data.shape})")
    #
    #         else:
    #             raise ValueError(f"Unsupported shape: {first_frame_data.shape}")
    #
    #         # --- fps logic (no change needed for ImageJ part) ---
    #         imgj = tif.imagej_metadata or {}
    #         fps = DEFAULT_FPS
    #
    #         if imgj:
    #             fps = imgj.get("fps", 0)
    #             logger.info(f"Acquired frame rate from tif metadata: {fps}")
    #         else:
    #             # --- THIS IS THE SECOND FIX ---
    #             # We now iterate over the *correct* list (all_pages)
    #             logger.debug(f"Scanning {len(all_pages)} pages/series for deviceTime...")
    #             times: list[float] = []
    #
    #             # This loop now works for Case 1, 2, and 3
    #             for item in all_pages:
    #                 page_with_tags = None
    #                 # 'item' can be a TiffPage (Cases 1, 2) or TiffPageSeries (Case 3)
    #                 if hasattr(item, 'tags'):
    #                     # This is a TiffPage (Cases 1 & 2)
    #                     page_with_tags = item
    #                 elif hasattr(item, 'pages') and item.pages:
    #                     # This is a TiffPageSeries (Case 3). Get its first page.
    #                     page_with_tags = item.pages[0]
    #
    #                 if not page_with_tags:
    #                     continue  # Skip if we couldn't find a page with tags
    #
    #                 # Now we safely access .tags on a valid TiffPage object
    #                 desc = page_with_tags.tags.get("ImageDescription")
    #                 if not desc:
    #                     continue
    #                 try:
    #                     desc_val = desc.value
    #                     if isinstance(desc_val, bytes):
    #                         desc_val = desc_val.decode('utf-8')
    #                     info = json.loads(desc_val)
    #                 except Exception:
    #                     continue
    #
    #                 t = info.get("deviceTime")
    #                 if isinstance(t, (int, float)):
    #                     times.append(t)
    #
    #             # Stop scanning if we are in Case 2 (Hyperstack) and have 0 times
    #             if len(all_pages) == 1 and not times:
    #                 logger.debug("Hyperstack has no deviceTime tag in its description.")
    #
    #             if len(times) >= 2:
    #                 deltas = [t2 - t1 for t1, t2 in zip(times, times[1:]) if (t2 - t1) > 0]
    #                 median_dt = median(deltas) if deltas else 0.1
    #                 fps = 1.0 / median_dt
    #                 logger.info(f"Derived frame rate from deviceTime: {fps}")
    #             else:
    #                 logger.debug(f"Cannot parse frame rate from this tif, falling back to default {DEFAULT_FPS}")
    #
    #         # ensure non-negative
    #         fps = max(fps, 1)
    #
    #     return fps, pix_fmt, w, h

    def _stream_tiff_frames(self, proc):
        """
        Uses the shared 'get_tiff_summary' to identify the file structure,
        then iterates and pipes data to FFmpeg.
        """
        step = 3
        next_emit = step

        try:
            with TiffFile(self.input_path) as tif:
                # --- LEVERAGE INSPECTOR STRUCTURE LOGIC ---
                _, access_strategy, total_frames, _, _ = get_tiff_summary(tif)

                if total_frames == 0:
                    logger.error("No frames found to process.")
                    return

                logger.info(f"Beginning stream of {total_frames} frames...")

                # Detect if this is Case 2 (Hyperstack / Memory Mapping required)
                is_hyperstack = (len(access_strategy) == 1 and total_frames > 1)

                if is_hyperstack:
                    # CASE 2: Hyperstack
                    series = access_strategy[0]
                    # 'memmap' ensures we don't load 50GB into RAM at once
                    data_stack = series.asarray(out='memmap')

                    for idx, frame in enumerate(data_stack, start=1):
                        self._write_frame(proc, frame, idx, total_frames, next_emit)

                        # --- PROGRESS UPDATE ---
                        pct = min(int(idx / total_frames * 100), 100)
                        if pct >= next_emit:
                            self.progress.emit(pct)  # <--- THIS WAS MISSING
                            next_emit = (pct // step + 1) * step
                else:
                    # CASES 1 & 3: Page/Series Iteration
                    for idx in range(total_frames):
                        # Lazy load the page
                        page_obj = access_strategy[idx]
                        frame = page_obj.asarray()

                        self._write_frame(proc, frame, idx + 1, total_frames, next_emit)

                        # --- PROGRESS UPDATE ---
                        pct = min(int((idx + 1) / total_frames * 100), 100)
                        if pct >= next_emit:
                            self.progress.emit(pct)  # <--- THIS WAS MISSING
                            next_emit = (pct // step + 1) * step

        except Exception as e:
            logger.error(f"Error streaming TIFF: {e}")
            raise
        finally:
            if proc and proc.stdin:
                try:
                    proc.stdin.close()
                except IOError:
                    pass

    # def _stream_tiff_frames(self, proc):
    #     step = 3
    #     next_emit = step
    #
    #     try:
    #         with TiffFile(self.input_path) as tif:
    #             frame_iterator, total_frames = self._find_tiff_pages(tif)
    #             if total_frames == 0:
    #                 logger.error("No frames found to process.")
    #                 return
    #             logger.info(f"Beginning stream of {total_frames} total frames...")
    #
    #             # This loop now works for all cases
    #             for idx, frame_data in enumerate(frame_iterator, start=1):
    #                 # Your original logic
    #                 if frame_data.ndim > 3 or (frame_data.ndim == 3 and frame_data.shape[-1] != 3):
    #                     logger.error(f"Frame {idx} has unexpected shape {frame_data.shape}, skipping.")
    #                     continue
    #                 frame = frame_data.astype("uint8")
    #                 proc.stdin.write(frame.tobytes())
    #
    #                 # Use the *correct* total_frames for percentage
    #                 pct = min(int(idx / total_frames * 100), 100)
    #                 if pct >= next_emit:
    #                     try:
    #                         self.progress.emit(pct)  # Your original line
    #                     except Exception as e:
    #                         logger.error(f"Error emitting progress: {e}")
    #                     next_emit = (pct // step + 1) * step  # More robust step logic
    #
    #     except Exception as e:
    #         logger.error(f"Error streaming TIFF: {e}")
    #         # Ensure proc is closed on error
    #
    #     finally:
    #         if proc and proc.stdin:
    #             try:
    #                 proc.stdin.close()
    #                 logger.info("Stream closed.")
    #             except IOError as e:
    #                 logger.error(f"Error closing stream (may already be closed): {e}")

    def _write_frame(self, proc, frame, idx, total, next_emit):
        """Helper to handle the byte writing and shape checks"""
        # Sanity check shape (ignore alpha channel if present, handled by pix_fmt)
        if frame.ndim > 3 or (frame.ndim == 3 and frame.shape[-1] not in (3, 4)):
            logger.warning(f"Frame {idx} has weird shape {frame.shape}, attempting write anyway.")

        # Ensure uint8
        if frame.dtype != 'uint8':
            # Fast cast if needed, though usually handled by TiffFile
            frame = frame.astype("uint8")

        proc.stdin.write(frame.tobytes())

        # Progress emission is handled in the loop to keep context of 'next_emit'
        # but could be moved here if we pass 'self' and 'step'.
        # Kept in loop for performance (function call overhead).

    def _process_video(self, frame_re, threads: int, slices: int):
        cmd = self._build_video_cmd(threads, slices)
        logger.info(f"Spawning FFmpeg {'compress' if self.mode=='compress' else 'uncompress'} process: " + " ".join(cmd))

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
        if proc.wait() != 0:
            raise RuntimeError("FFmpeg conversion failed")

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
        if self.mode == "uncompress":
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
