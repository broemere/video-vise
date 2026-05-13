import os
import re
import subprocess
import logging
from pathlib import Path
from PySide6.QtCore import QThread, Signal
from widgets.resources import FFMPEG

logger = logging.getLogger(__name__)


class FFmpegCropper(QThread):
    progress = Signal(int)
    result = Signal(str, str, str, int)  # Emits original filename and new output path
    failed = Signal(Path, str)

    def __init__(self, input_path: Path, output_path: Path, frames: int, x: int, y: int, w: int, h: int, orig_w: int,
                 orig_h: int, track: bool):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.frames = frames
        self.track = track

        # Original video dimensions bounds checking
        self.orig_w = orig_w
        self.orig_h = orig_h

        # Sanitize and strictly clamp the crop parameters
        self.x, self.y, self.w, self.h = self._sanitize_dimensions(x, y, w, h)

    def _sanitize_dimensions(self, x: int, y: int, w: int, h: int):
        """
        Ensures dimensions are even numbers and do not exceed the original video bounds.
        """

        def pad_axis(pos, size, max_size, target_multiple=16):
            # 1. How many pixels are we short of the next multiple?
            remainder = size % target_multiple
            if remainder == 0:
                return pos, size

            pad_needed = target_multiple - remainder

            # 2. Split the padding evenly to keep the crop box centered
            pad_before = pad_needed // 2
            pad_after = pad_needed - pad_before

            # 3. Apply the padding
            new_pos = pos - pad_before
            new_size = size + pad_needed

            # 4. Strict Boundary Checks
            # Did we push the start position below 0?
            if new_pos < 0:
                new_pos = 0

            # Did we push the end position past the right/bottom edge?
            if new_pos + new_size > max_size:
                # Push the starting coordinate back to the left/up
                new_pos = max_size - new_size

                # If pushing it back made it negative again, the video itself
                # is too small to support this multiple of 16.
                if new_pos < 0:
                    new_pos = 0
                    # Fallback to the largest valid even number within bounds
                    new_size = max_size if max_size % 2 == 0 else max_size - 1

            return new_pos, new_size

        # Process both axes through the helper function
        new_x, new_w = pad_axis(x, w, self.orig_w)
        new_y, new_h = pad_axis(y, h, self.orig_h)

        logger.debug(f"Original Crop: {w}x{h} @ ({x},{y}) -> Sanitized Crop: {new_w}x{new_h} @ ({new_x},{new_y})")

        return new_x, new_y, new_w, new_h


    def choose_slices(self, threads: int) -> int:
        """Shared logic for FFV1 slicing efficiency."""
        allowed = [4, 6, 9, 12, 16, 24, 30]
        ideal = threads * 2
        valid = [s for s in allowed if s <= ideal]
        return max(valid) if valid else min(allowed)

    def run(self):
        logger.debug(f"FFmpegCropper STARTED: {self.input_path} -> {self.output_path}")
        try:
            cpu_count = os.cpu_count() or 1
            threads = 3 if cpu_count == 4 else min(max(cpu_count - 2, 1), 16)
            slices = self.choose_slices(threads)

            # Note: We cannot use "-c:v copy" here because applying a video filter (-vf)
            # requires re-encoding the raw pixel data. We encode back to lossless FFV1.
            cmd = [
                FFMPEG, "-y",
                "-i", str(self.input_path),
                "-vf", f"crop={self.w}:{self.h}:{self.x}:{self.y}",
                "-c:v", "ffv1",
                "-level", "3",
                "-threads", str(threads),
                "-coder", "1",
                "-context", "1",
                "-g", "1",
                "-slices", str(slices),
                "-slicecrc", "1",
                str(self.output_path)
            ]

            logger.info("Spawning FFmpeg crop process: " + " ".join(cmd))

            proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True)
            frame_re = re.compile(r"frame=\s*(\d+)\b")

            # Parse stderr for progress updates
            for line in proc.stderr:
                if self.isInterruptionRequested():
                    logger.info("Interruption requested. Terminating FFmpeg process...")
                    proc.terminate()  # Kill the child process
                    break
                if self.frames > 0:
                    m = frame_re.search(line)
                    if m:
                        frame_num = int(m.group(1))
                        pct = min(int(frame_num / self.frames * 100), 100)
                        self.progress.emit(pct)

            if proc.wait() != 0:
                raise RuntimeError("FFmpeg cropping failed (non-zero exit code).")

            # Preserve metadata timestamps
            stat = self.input_path.stat()
            os.utime(self.output_path, (stat.st_atime, stat.st_mtime))

            self.progress.emit(100)

            if self.track:
                key_name, result = self._get_size_diff()
                if result == 0:
                    self.result.emit(str(self.input_path), str(self.output_path), "", 0)
                else:
                    self.result.emit(str(self.input_path), str(self.output_path), key_name, result)

        except Exception as e:
            logger.error(f"WORKER CRASHED cropping {self.input_path}!", exc_info=True)
            self.failed.emit(self.input_path, str(e))

    def _get_size_diff(self):
        try:
            input_size = self.input_path.stat().st_size
            output_size = self.output_path.stat().st_size
            diff = int(round((input_size - output_size) / (1024 ** 3)))
            if diff == input_size:
                diff = 0
        except Exception:
            diff = 0

        name = str(self.input_path.name)[:-4].replace(" ", "")
        return [name + str(input_size), diff]