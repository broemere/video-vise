from pathlib import Path
import logging
from PySide6.QtCore import QSettings, Qt, QThread, Signal, QTimer
import re
import os
import numpy as np
import subprocess
from tifffile import TiffFile
from statistics import median
import json
import threading
from widgets.resources import FFMPEG, FFPROBE
from widgets.inspecting import inspect_tiff, get_video_info, get_tiff_summary


logger = logging.getLogger(__name__)


class FrameValidator(QThread):
    progress = Signal(int)
    result   = Signal(str, bool)

    def __init__(self, orig: Path, comp: Path, frames: int):
        super().__init__()
        self.orig   = orig
        self.comp   = comp
        self.frames = frames
        self.step_pct = 3
        self.PREFIX = 50
        if orig.suffix.lower() in ['.tif', '.tiff']:
            info = get_video_info(orig)
        else:
            info = get_video_info(orig)

        self.target_pix_fmt = info.get("pix_fmt", "gray")
        logger.debug(f"Validator: Target format {self.target_pix_fmt}, Expecting Frames: {self.frames}")

        logger.debug(f"Validator determined target format for {orig.name}: {self.target_pix_fmt}")

    def run(self):
        logger.debug(f"FrameValidator thread STARTED: orig={self.orig}, comp={self.comp}")
        try:

            # Check that frame counts match
            comp_info = get_video_info(self.comp)
            actual_comp_frames = int(comp_info.get("frames", 0))

            if actual_comp_frames != self.frames:
                logger.warning(
                    f"Frame count mismatch detected! Header Expected: {self.frames}, "
                    f"MKV Actual: {actual_comp_frames}. Initiating Deep Scan..."
                )

                # --- NEW LOGIC START ---
                # The header might be lying (truncated source). Check the PHYSICAL source frames.
                true_orig_frames = self._deep_scan_frame_count(self.orig)

                if actual_comp_frames == true_orig_frames:
                    logger.info(
                        f"Deep Scan Rescue: Source file is truncated/corrupt but conversion preserved all valid data. "
                        f"Updating expected frames from {self.frames} to {true_orig_frames}."
                    )
                    # Update our expectation to match reality so the hashing logic works correctly
                    self.frames = true_orig_frames
                else:
                    # Genuine failure: The output has different frames than the source (even after deep scanning)
                    logger.error(
                        f"Validation Failed: Mismatch persists after deep scan. "
                        f"Source Physical: {true_orig_frames}, MKV Actual: {actual_comp_frames}."
                    )
                    self.result.emit(str(self.comp), False)
                    return

            comp_hashes = self._hash_file(self.comp, 0, 50)
            orig_hashes = self._hash_file(self.orig, 50, 100, is_original=True)

            logger.debug(f"Hashes: {comp_hashes[:5]}")
            logger.debug(f"Hashes: {orig_hashes[:5]}")

            self.progress.emit(100)
            if len(orig_hashes) != len(comp_hashes):
                logger.warning(f"Hash count mismatch! Orig: {len(orig_hashes)}, Comp: {len(comp_hashes)}")
                is_lossless = False
            else:
                is_lossless = (orig_hashes == comp_hashes)
            logger.info(
                f"FrameValidator result: is_lossless={is_lossless} "
                f"(orig_frames={len(orig_hashes)} vs comp_frames={len(comp_hashes)})"
            )
            self.result.emit(str(self.comp), is_lossless)

        except Exception:
            logger.error("Exception in FrameValidator.run()", exc_info=True)
            # ensure UI doesn’t hang
            self.result.emit(str(self.comp), False)

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
        hashes = []
        vf_chain = []

        if not is_original:
            vf_chain.append(f"format={self.target_pix_fmt}")

        def run_hash_cmd(args, current_start_pct, current_end_pct, expected_frames):
            logger.info(f"Hash cmd: {' '.join(args)}")
            proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

            chunk_hashes = []
            count = 0

            self.progress.emit(current_start_pct)

            for line in proc.stdout:
                if line.startswith("#"): continue
                chunk_hashes.append(line.split(",")[-1].strip())
                count += 1

                if expected_frames > 0:
                    frac = count / expected_frames
                    pct = current_start_pct + int(frac * (current_end_pct - current_start_pct))
                    self.progress.emit(min(pct, current_end_pct))

            proc.wait()
            self.progress.emit(current_end_pct)
            return chunk_hashes

        # limit ffmpeg to PREFIX frames.
        # If the file is shorter than PREFIX, ffmpeg stops automatically at EOF.
        cmd = [
            FFMPEG,
            "-i", str(path),
            "-map", "0:v:0",
            "-frames:v", str(self.PREFIX)
        ]

        if vf_chain: cmd.extend(["-vf", ",".join(vf_chain)])
        cmd.extend(["-f", "framemd5", "pipe:1"])

        # Important: For the progress bar math, we need to know if we expect 50 frames
        # or fewer (if the file is short).
        frames_to_expect = min(self.frames, self.PREFIX)

        return run_hash_cmd(cmd, start_pct, end_pct, frames_to_expect)

    def _hash_tiff(self, path: Path, start_pct: int, end_pct: int) -> list[str]:
        hashes = []
        try:
            with TiffFile(path) as tif:
                # 1. Use Shared Inspector Logic
                _, access_strategy, total_frames, series_shape, _ = get_tiff_summary(tif)

                if total_frames == 0: return []
                count_to_check = min(total_frames, self.PREFIX)
                indices = list(range(count_to_check))

                #first_page = access_strategy[0]
                is_hyperstack = (len(access_strategy) == 1 and total_frames > 1)

                h, w = (0, 0)

                # Logic copied/adapted from your inspect_tiff to handle shapes
                # series_shape might be (T, Y, X), (Y, X), (T, Y, X, 3), etc.

                # Remove T dimension if present
                shape_check = list(series_shape)
                if is_hyperstack and len(shape_check) > 2:
                    # (9000, 800, 800) -> (800, 800)
                    # Remove the first dimension (Time/Frames)
                    shape_check.pop(0)

                # Now handle Color vs Gray
                if len(shape_check) == 2:
                    h, w = shape_check
                elif len(shape_check) >= 3:
                    if shape_check[-1] in (3, 4):
                        h, w = shape_check[-3:-1]
                    else:
                        h, w = shape_check[-2:]

                cmd = [
                    FFMPEG, "-f", "rawvideo",
                    "-pix_fmt", self.target_pix_fmt,
                    "-s", f"{w}x{h}",
                    "-i", "pipe:0",
                    "-vf", f"format={self.target_pix_fmt}",
                    "-f", "framemd5", "pipe:1"
                ]

                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

                reader_thread = threading.Thread(target=self._drain_stdout, args=(proc.stdout, hashes))
                reader_thread.daemon = True
                reader_thread.start()

                processed = 0
                next_emit = start_pct + self.step_pct

                # Hyperstack Optimization (Memmap)
                if is_hyperstack:
                    series = access_strategy[0]
                    data_stack = series.asarray(out='memmap')

                for idx in indices:
                    try:
                        # A. Fetch Frame (Random Access)
                        if is_hyperstack:
                            frame = data_stack[idx]
                        else:
                            frame = access_strategy[idx].asarray()

                        # B. Restore Shape Safety Checks (FROM ORIGINAL)
                        # Ensure we don't pipe (Y, X, 1) when ffmpeg expects (Y, X)
                        if self.target_pix_fmt in ("gray", "gray8", "gray16le") and frame.ndim == 3:
                            frame = np.squeeze(frame)
                            # If squeezing made it (Y, X, 3), take channel 0
                            if frame.ndim == 3:
                                frame = frame[..., 0]

                        # Ensure we don't pipe (Y, X) when ffmpeg expects RGB
                        elif "rgb" in self.target_pix_fmt and frame.ndim == 2:
                            logger.warning(f"Frame {idx} is grayscale, but TIF target is RGB. Skipping.")
                            continue

                        # C. Write
                        frame = frame.astype("uint8")
                        proc.stdin.write(frame.tobytes())

                        # D. Progress
                        processed += 1
                        frac = processed / count_to_check
                        scaled = start_pct + int(frac * (end_pct - start_pct))
                        if scaled >= next_emit:
                            self.progress.emit(scaled)
                            next_emit += self.step_pct

                    except Exception as e:
                        logger.error(f"Error hashing tiff frame {idx}: {e}")
                        break

                proc.stdin.close()
                reader_thread.join()
                proc.wait()

                return hashes

        except Exception as e:
            logger.error(f"Error in _hash_tiff: {e}", exc_info=True)
            return []

    def _drain_stdout(self, pipe, hashes):
        for raw in pipe:
            line = raw.decode("utf-8", "ignore")
            if not line.startswith("#"):
                hashes.append(line.rsplit(",", 1)[-1].strip())

    def _deep_scan_frame_count(self, path: Path) -> int:
        """
        Runs a full decode scan on the file to count actual physically present frames.
        Used when header metadata contradicts the converted output.
        """
        try:
            # -count_frames forces decoding of the entire stream
            # -show_entries stream=nb_read_frames gets the result
            cmd = [
                FFPROBE, "-v", "error",
                "-select_streams", "v:0",
                "-count_frames",
                "-show_entries", "stream=nb_read_frames",
                "-of", "csv=p=0",
                str(path)
            ]

            # NOTE: This operation is blocking and may take time for large files,
            # but it is only triggered in error states.
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, encoding="utf-8"
            )

            # Output should be a single number like "145"
            count = result.stdout.strip()
            return int(count) if count.isdigit() else 0

        except Exception as e:
            logger.error(f"Deep scan failed for {path}: {e}")
            return 0