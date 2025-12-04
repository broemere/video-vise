import json
from pathlib import Path
import subprocess
import logging
from tifffile import TiffFile
from statistics import median
from widgets.formatting import format_decimal, frac_to_float
from widgets.resources import *

# Map base format to ffmpeg pixel format string (partial), append 'le' or 'be' dynamically for 16-bit types
BASE_TIFF_FMT = {
    "gray8": "gray",
    "rgb8": "rgb24",
    "rgba8": "rgba",
    "gray16": "gray16", # Will append le/be
    "rgb16": "rgb48",   # Will append le/be
    "rgba16": "rgba64", # Will append le/be
}

logger = logging.getLogger(__name__)


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
        "fps": "0",
        "raw_fps": "0",
        "frames": "0"
    }

    try:
        if fp.suffix.lower() in (".tif", ".tiff"):
            info = inspect_tiff(fp)
        else:
            info = inspect_ffprobe(fp)

        # Merge info into default, converting all values to strings
        return {k: str(info.get(k, default[k])) for k in default}

    except subprocess.CalledProcessError as e:
        indent = "  "
        # Pre-process the stderr: replace every newline with a newline + indent
        indented_stderr = e.stderr.strip().replace("\n", f"\n{indent}")
        logger.error(
            f"get_video_info failed for {fp} (ffprobe error).\n"
            f"  Return Code: {e.returncode}\n"
            f"  Stderr: {indented_stderr}"  # <-- Use the new indented string
        )
        return default

    except json.JSONDecodeError as e:
        # This catches if ffprobe succeeds but outputs non-JSON garbage
        logger.error(f"get_video_info failed for {fp} (JSON parse error): {e}")
        return default

    except Exception as e:
        logger.error(f"get_video_info failed for {fp}: {e}", exc_info=True)
        return default


def get_tiff_summary(tif: TiffFile):
    """
    Efficiently inspects a TIF file (for all 3 cases)
    and returns its vital stats *without* loading pixel data.
    """
    if not tif.series:
        raise ValueError("TIF file contains no series.")

    # --- CASE 3: Multi-Series (OMEX / One frame per series) ---
    if len(tif.series) > 1:
        if not tif.series[0] or not tif.series[0].pages:
            raise ValueError("TIF file series[0] contains no pages.")

        # Optimization: Don't list-comp all pages. Use the series itself as the container.
        # assume series[i].pages[0] is the frame.
        first_page_obj = tif.series[0].pages[0]

        # Create a proxy wrapper to treat series like a page list
        class SeriesProxy:
            def __init__(self, series_seq):
                self.series = series_seq

            def __getitem__(self, index):
                # Lazy access: only parses the specific series when requested
                return self.series[index].pages[0]

            def __len__(self):
                return len(self.series)

        access_strategy = SeriesProxy(tif.series)
        total_frames = len(tif.series)
        series_shape = tif.series[0].shape
        series_dtype = tif.series[0].dtype

        return first_page_obj, access_strategy, total_frames, series_shape, series_dtype

    # --- CASES 1 & 2: Single-Series ---
    main_series = tif.series[0]
    series_shape = main_series.shape
    series_dtype = main_series.dtype

    # Identify dimensionality
    ndim = len(series_shape)
    has_color = (ndim > 0 and series_shape[-1] in (3, 4))
    spatial_dims = ndim - 1 if has_color else ndim

    def calc_frames_from_shape():
        if spatial_dims > 2:
            return series_shape[0]  # (T, Y, X)
        return 1

    calculated_frames = calc_frames_from_shape()
    page_count = len(main_series.pages) if main_series.pages else 0

    # Logic: If we have many frames in the Shape, but only 1 Page (or 0),
    # it is a Case 2 Hyperstack. We MUST use the Series as the access object.

    # Identify the first page for metadata (tags)
    # Even a hyperstack usually has at least 1 page containing the tags.
    first_page_obj = main_series.pages[0] if page_count > 0 else None

    # Check for Hyperstack condition:
    # We have many frames in the shape, but few pages (usually 1) in the file structure.
    is_hyperstack = (page_count <= 1 and calculated_frames > 1)

    if is_hyperstack:
        # Case 2: Hyperstack (memory mapping / internal decoding)
        # CRITICAL FIX: Return 'first_page_obj' (a Page) for metadata,
        # but '[main_series]' (a Series list) for pixel access.
        return first_page_obj, [main_series], calculated_frames, series_shape, series_dtype

    # Case 1: Standard Multi-Page
    if page_count > 1:
        total_frames = page_count
    else:
        total_frames = 1

    return first_page_obj, main_series.pages, total_frames, series_shape, series_dtype

def inspect_tiff(fp: Path) -> dict[str, any]:
    """
    Robustly and efficiently inspects a TIFF file,
    handling all 3 common stack structures.
    """
    with TiffFile(fp) as tif:
        try:
            # pages_proxy supports __getitem__ efficiently
            first_obj, pages_proxy, n_pages, series_shape, series_dtype = get_tiff_summary(tif)
        except ValueError as e:
            logger.error(f"Failed to inspect TIF {fp}: {e}")
            return {}

        # --- Dimensions & Color ---
        h, w = (0, 0)
        bits = series_dtype.itemsize * 8
        phot = getattr(first_obj, "photometric", None)
        phot_name = phot.name if phot else "UNKNOWN"

        color_type = "gray"
        # Simplistic shape logic
        if len(series_shape) == 2:
            h, w = series_shape
        elif len(series_shape) == 3:
            if series_shape[-1] in (3, 4):
                h, w = series_shape[:2]
                color_type = "color"
            else:
                h, w = series_shape[1:]
        elif len(series_shape) >= 4:
            if series_shape[-1] in (3, 4):
                h, w = series_shape[-3:-1]
                color_type = "color"
            else:
                h, w = series_shape[-2:]

        # --- Pixel Format & Endianness ---
        # Check byte order: '>' is BigEndian, '<' is LittleEndian
        endian_suffix = "be" if tif.byteorder == ">" else "le"

        if phot_name in ("MINISBLACK", "MINISWHITE"):
            base_fmt = f"gray{bits}"
            color = "gray"
        elif phot_name == "RGB":
            base_fmt = f"rgb{bits}"
            color = "color"
        else:
            base_fmt = f"{color_type}{bits}"
            color = color_type

            # Construct final ffmpeg string
        if bits > 8:
            # Append le/be for 16-bit+ types
            pix_fmt = BASE_TIFF_FMT.get(base_fmt, "gray")
            # If the dict returned a generic base like 'gray16', append suffix
            if pix_fmt.endswith("16") or pix_fmt.endswith("48") or pix_fmt.endswith("64"):
                pix_fmt += endian_suffix
        else:
            pix_fmt = BASE_TIFF_FMT.get(base_fmt, "gray")

        # --- 3. Compression (from first_obj) ---
        comp_tag = first_obj.tags.get('Compression')
        codec_tag = ""
        if comp_tag is not None:
            comp_value = comp_tag.value
            compression_map = {
                1: "raw", 5: "lzw", 6: "jpeg", 7: "jpeg",
                8: "deflate", 32773: "packbits",
            }
            codec_tag = compression_map.get(comp_value, str(comp_value))

        # --- 4. Duration & FPS logic ---
        imgj = tif.imagej_metadata or {}
        duration_sec = 0.0
        fps = 0.0

        if imgj and imgj.get("fps"):
            # Trust ImageJ metadata if present
            fps = imgj.get("fps", 0)
            if fps > 0:
                duration_sec = (n_pages - 1) / fps

        else:
            # Fallback: Scan deviceTime, but SAMPLE instead of linear scan
            # Strategy: Grab first 5 frames for FPS delta, and Last frame for Duration
            # This makes the complexity O(1) instead of O(N)
            indices_to_check = list(range(min(n_pages, 5)))
            if n_pages > 5:
                indices_to_check.append(n_pages - 1)

            sorted_times = {}  # Map index -> time

            for idx in indices_to_check:
                try:
                    pg = pages_proxy[idx]  # Lazy access
                    if not hasattr(pg, 'tags'): continue

                    desc = pg.tags.get("ImageDescription")
                    if not desc: continue

                    val = desc.value
                    if isinstance(val, bytes): val = val.decode('utf-8', errors='ignore')

                    # Quick string search to avoid full JSON parse if possible?
                    # JSON parse is safer though.
                    d_info = json.loads(val)
                    t = d_info.get("deviceTime")
                    if t is None:
                        t = d_info.get("time_s")
                    if isinstance(t, (int, float)):
                        sorted_times[idx] = t
                except Exception:
                    continue

            # Calculate from sampled data
            if sorted_times:
                # 1. Total Duration: Last Time - First Time
                start_idx = min(sorted_times.keys())
                end_idx = max(sorted_times.keys())

                if start_idx != end_idx:
                    duration_sec = sorted_times[end_idx] - sorted_times[start_idx]

                # 2. Estimate FPS from the initial consecutive burst
                deltas = []
                sorted_keys = sorted(k for k in sorted_times.keys() if k < 5)
                for k1, k2 in zip(sorted_keys, sorted_keys[1:]):
                    # Only accept consecutive frames for FPS calc
                    if k2 == k1 + 1:
                        dt = sorted_times[k2] - sorted_times[k1]
                        if dt > 0: deltas.append(dt)

                median_dt = median(deltas) if deltas else 0
                fps = (1.0 / median_dt) if median_dt > 0 else 0

            # Sanity Checks
        fps = max(fps, 0)  # Return int FPS? Or float? User code used int(round())
        duration_sec = max(duration_sec, 0.0)

        # --- 5. Return the robust data ---
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
    # Run command (Fast, header read only)
    args = [
        "ffprobe", "-v", "error",
        "-show_format", "-show_streams",
        "-of", "json",
        str(fp)
    ]

    try:
        result = subprocess.run(
            args, capture_output=True, text=True, check=True, encoding="utf-8"
        )
        data = json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        # Return a "safe" empty dict if the file is totally corrupt
        logger.error(f"FFprobe failed for {fp}: {e}")
        return {"duration": 0, "frames": 0, "fps": 0.0, "width": 0, "height": 0}

    fmt = data.get("format", {})
    # Get first video stream safely
    vs = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})

    # --- HELPER: Safely parse numbers from "N/A", "100", or None ---
    def safe_float(val):
        try:
            return float(val) if val and val != "N/A" else 0.0
        except ValueError:
            return 0.0

    def safe_int(val):
        try:
            return int(val) if val and val != "N/A" else 0
        except ValueError:
            return 0

    # GET DURATION (Check Format first, then Stream)
    duration_s = safe_float(fmt.get("duration"))
    if duration_s == 0:
        duration_s = safe_float(vs.get("duration"))

    # GET FPS
    avg_fps = frac_to_float(vs.get("avg_frame_rate"))
    r_fps = frac_to_float(vs.get("r_frame_rate"))
    # Prefer average, fall back to real/closest
    fps = avg_fps if avg_fps > 0 else (r_fps if r_fps > 0 else 0.0)

    # GET FRAMES
    # Try explicit header first
    nb_frames = safe_int(vs.get("nb_frames"))

    # If header is 0 or missing, CALCULATE it
    if nb_frames == 0 and duration_s > 0 and fps > 0:
        nb_frames = int(round(duration_s * fps))

    # 6. Return Clean Data
    return {
        "duration": duration_s,
        "codec_name": vs.get("codec_name", "Unknown"),
        "pix_fmt": vs.get("pix_fmt", "Unknown"),
        "width": safe_int(vs.get("width")),
        "height": safe_int(vs.get("height")),
        "codec_tag_string": vs.get("codec_tag_string", "Unknown"),
        "fps": format_decimal(fps),  # formatted for display
        "raw_fps": fps,  # kept raw for math if needed
        "frames": nb_frames,
    }
