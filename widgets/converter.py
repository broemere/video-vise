import logging
logger = logging.getLogger(__name__)
import re
import os
import sys
import subprocess
import numpy as np
from pathlib import Path
from tifffile import TiffFile, TiffWriter
from PySide6.QtCore import QSettings, Qt, QThread, Signal, QTimer
from config import DEFAULT_FPS
from widgets.inspecting import get_video_info, get_tiff_summary
from widgets.resources import FFMPEG, FFPROBE

class FFmpegConverter(QThread):
    progress = Signal(int)
    result = Signal(str, int)
    failed = Signal(Path, str)

    def __init__(self, input_path: Path, output_path: Path, frames: int, mode: str, track: bool, in_fmt: str):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.frames = frames
        self.mode = mode
        self.track = track
        self.in_fmt = in_fmt

    def run(self):
        logger.debug(
            f"FFmpegConverter thread STARTED: input={self.input_path}, "
            f"output={self.output_path}, frames={self.frames}, mode={self.mode!r}, input_format={self.in_fmt}"
        )
        try:
            cpu_count = os.cpu_count() or 1
            threads = 3 if cpu_count == 4 else min(max(cpu_count - 2, 1), 16) # Changed from 4 to 2 for 6 core devices
            # Maybe improve this logic again for different size CPUs (small and large)
            slices = self.choose_slices(threads)

            if self.input_path.suffix.lower() in [".tif", ".tiff"]:
                self._process_tiff(threads, slices)
            else:
                self._process_video(threads, slices)

            logger.debug("Process function returned. Finalizing metadata...")

            stat = self.input_path.stat()
            os.utime(self.output_path, (stat.st_atime, stat.st_mtime))

            self.progress.emit(100)  # Finish
            if self.track:
                self._emit_size_diff()
            else:
                self.result.emit("", 0)  # Empty result in case of no tracking

        except ValueError as ve:
            # This catches your "16-bit RGB not supported" error
            logger.warning(f"Conversion skipped: {ve}")
            self.failed.emit(self.input_path, str(ve))
        except Exception as e:
            logger.error(f"WORKER CRASHED processing {self.input_path}!", exc_info=True)  # Catch crashes and log traceback
            self.failed.emit(self.input_path, str(e))

    def _process_tiff(self, threads: int, slices: int):
        info = get_video_info(self.input_path)
        fps = float(info.get("raw_fps", DEFAULT_FPS))  # Use raw_fps if available for precision
        if fps == 0: fps = DEFAULT_FPS  # Fallback

        pix_fmt = info.get("pix_fmt", "gray")
        w = info.get("width", "0")
        h = info.get("height", "0")

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
        self._stream_tiff_frames(proc)
        if proc.wait() != 0:
            raise RuntimeError("FFmpeg TIFF→video failed (non-zero exit code)")

    def _stream_tiff_frames(self, proc):
        """
        Uses the shared 'get_tiff_summary' to identify the file structure,
        then iterates and pipes data to FFmpeg.
        """
        step = 3
        next_emit = step

        try:
            with TiffFile(self.input_path) as tif:
                _, access_strategy, total_frames, _, _ = get_tiff_summary(tif)

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
                        pct = min(int(idx / total_frames * 100), 100)
                        if pct >= next_emit:
                            self.progress.emit(pct)
                            next_emit = (pct // step + 1) * step
                else:
                    # CASES 1 & 3: Page/Series Iteration
                    for idx in range(total_frames):
                        # Lazy load the page
                        page_obj = access_strategy[idx]
                        frame = page_obj.asarray()
                        self._write_frame(proc, frame, idx + 1, total_frames, next_emit)
                        pct = min(int((idx + 1) / total_frames * 100), 100)
                        if pct >= next_emit:
                            self.progress.emit(pct)
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

    def _write_frame(self, proc, frame, idx, total, next_emit):
        """Helper to handle the byte writing and shape checks"""
        # Sanity check shape (ignore alpha channel if present, handled by pix_fmt)
        # Get the original bit depth
        dtype = frame.dtype

        # If it's already 8-bit, just send it
        if dtype == 'uint8':
            proc.stdin.write(frame.tobytes())
            return

        # If it's 16-bit (uint16), send the full 2-byte-per-pixel data
        if dtype == 'uint16' or dtype == '<u2' or dtype == '>u2':
            # Ensure Little Endian for FFmpeg 'gray16le'
            if frame.dtype.byteorder == '>' or (frame.dtype.byteorder == '=' and sys.byteorder == 'big'):
                frame = frame.byteswap().newbyteorder('<')

            proc.stdin.write(frame.tobytes())
            return

        # Fallback for unexpected types (like float or int32)
        logger.warning(f"Unexpected dtype {dtype} on frame {idx}. Casting to uint8 (Lossy!)")
        proc.stdin.write(frame.astype('uint8').tobytes())

    def _process_video(self, threads: int, slices: int):
        out_ext = self.output_path.suffix.lower()
        if self.mode == "decompress" and out_ext in [".tif", ".tiff"]:
            logger.info(f"Using TiffWriter for: {self.output_path.name}")
            self._decompress_to_tiff_stack(threads)
            return

        cmd = self._build_video_cmd(threads, slices)
        frame_re = re.compile(r"frame=\s*(\d+)\b")
        logger.info(f"Spawning FFmpeg {'compress' if self.mode=='compress' else 'decompress'} process: " + " ".join(cmd))
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

            is_16bit = "16" in self.in_fmt or "48" in self.in_fmt

            if is_16bit:
                # Final Conclusion: AVI cannot handle raw 16-bit without
                # falling back to broken RGB555.
                raise ValueError(
                    f"High bit-depth ({self.in_fmt}) is incompatible with raw AVI. "
                    "Decompression aborted to prevent file corruption."
                )

            # Standard 8-bit Logic
            is_color = any(x in self.in_fmt for x in ["rgb", "bgr", "0", "yuv"])
            target = "bgr24" if is_color else "gray"

            return base + [
                "-vcodec", "rawvideo",
                "-vf", f"format={target}",
                "-pix_fmt", target,
                str(self.output_path)
            ]

    def _decompress_to_tiff_stack(self, threads: int):
        # 1. Inspect the source to get dimensions and format
        info = get_video_info(self.input_path)
        w = int(info.get("width", 0))
        h = int(info.get("height", 0))
        pix_fmt = info.get("pix_fmt", "gray")  # Default
        is_16bit = "16" in pix_fmt or "48" in pix_fmt or "64" in pix_fmt

        # Check if the source is grayscale or color
        # YUV, PAL8, and RGB/BGR are all 'color' for our TIFF purposes
        if any(x in pix_fmt for x in ["rgb", "gbr", "bgr", "yuv", "pal8", "0"]):
            channels = 3
            target_pix_fmt = "rgb48le" if is_16bit else "rgb24"
        else:
            channels = 1
            target_pix_fmt = "gray16le" if is_16bit else "gray"

        dtype = np.uint16 if is_16bit else np.uint8
        bpp = 2 if is_16bit else 1

        frame_size = w * h * channels * bpp

        # 2. Command: Output RAW pixels to stdout (pipe:1)
        cmd = [
            FFMPEG, "-y",
            "-i", str(self.input_path),
            "-vf", f"format={target_pix_fmt}",  # Force the flip here
            "-f", "rawvideo",
            "-pix_fmt", target_pix_fmt,
            "-threads", str(threads),
            "-loglevel", "error",  # Minimize stderr noise
            "pipe:1"
        ]

        logger.info(f"Decompressing to TIFF Stack: {w}x{h}, {channels}ch, {dtype.__name__}")

        # 3. Stream and Write
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10 ** 8)

        count = 0
        try:
            # bigtiff=True is critical for >4GB files
            with TiffWriter(self.output_path, bigtiff=True) as tif:
                while True:
                    raw_frame = proc.stdout.read(frame_size)
                    if len(raw_frame) < frame_size:
                        break

                    # Reshape raw bytes to 2D image
                    if channels == 3:
                        frame_array = np.frombuffer(raw_frame, dtype=dtype).reshape((h, w, 3))
                    else:
                        frame_array = np.frombuffer(raw_frame, dtype=dtype).reshape((h, w))
                    tif.write(frame_array, contiguous=True)

                    # Update progress every 5 frames
                    count += 1
                    if self.frames and count % 5 == 0:
                        self.progress.emit(min(int(count / self.frames * 100), 100))

            proc.wait()
            logger.info(f"TiffWriter finished. Wrote {count} frames.")

            # This is why the UI didn't progress! You must emit the result
            # so MainWindow knows this worker is finished.
            #self.progress.emit(100)

        except Exception as e:
            logger.error(f"Error in TIFF decompression: {e}")
            raise  # Pass to the 'run' method's catch block

        finally:
            if proc.poll() is None:
                proc.terminate()
            ret_code = proc.wait()
            if ret_code != 0:
                err = proc.stderr.read().decode()
                logger.error(f"FFmpeg Pipe Error (Code {ret_code}): {err}")

    def _emit_size_diff(self):
        try:
            input_size = self.input_path.stat().st_size
            output_size = self.output_path.stat().st_size
            diff = int(round((input_size - output_size) / (1024 ** 3)))
            if diff == input_size:
                diff = 0
        except Exception:
            diff = 0
        if self.mode == "decompress":
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
