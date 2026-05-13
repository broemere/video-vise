import sys
from pathlib import Path, PurePath
import logging
from config import supported_extensions
import psutil
import json
import subprocess
import os
from typing import Iterable
from PySide6.QtCore import QThread, Signal
from widgets.inspecting import get_video_info, sample_tail_zeros

logger = logging.getLogger(__name__)


class FileScannerThread(QThread):
    # Signals to communicate with the main GUI thread
    progress = Signal(int)
    status_text = Signal(str)
    # Emit a tuple containing all the dictionaries and lists we built
    scan_finished = Signal(tuple)

    def __init__(self, folder, known_lossless, parent=None):
        super().__init__(parent)
        self.folder = folder
        self.known_lossless = known_lossless  # Pass existing cache if needed

    def run(self):
        self.status_text.emit("Discovering files...")
        base_folder = Path(self.folder)

        # 1. Get files recursively
        files = get_files(self.folder)

        files_by_dir = {}
        for fp in files:
            if self.isInterruptionRequested(): return
            parent_dir = fp.parent
            if parent_dir not in files_by_dir:
                files_by_dir[parent_dir] = []
            files_by_dir[parent_dir].append(fp)

        # 2. Build rendering order
        all_items_to_render = []
        if base_folder in files_by_dir:
            top_level_files = sorted(files_by_dir[base_folder], key=lambda p: p.name.lower())
            all_items_to_render.extend(top_level_files)
            del files_by_dir[base_folder]

        sorted_dirs = sorted(files_by_dir.keys(), key=str)
        for dir_path in sorted_dirs:
            if self.isInterruptionRequested(): return
            relative_dir = dir_path.relative_to(base_folder)
            all_items_to_render.append(str(relative_dir))
            files_in_dir = sorted(files_by_dir[dir_path], key=lambda p: p.name.lower())
            all_items_to_render.extend(files_in_dir)

        # 3. Gather data
        files_only = [x for x in all_items_to_render if isinstance(x, Path)]
        total_files = len(files_only)

        info_map = {}
        tail_data = {}
        source_map = {}
        stat_map = {}
        sizes = {}
        formats = {}
        codecs = {}
        frames = {}

        for i, fp in enumerate(files_only, 1):
            if self.isInterruptionRequested(): return

            self.status_text.emit(f"Scanning file {i}/{total_files}...")

            fp_key = str(fp)
            info = get_video_info(fp)  # This blocks, but now we are in the background!
            info_map[fp_key] = info

            codec = info.get("codec_name", "-")
            if codec == "Unknown": codec = "-"
            codecs[fp_key] = codec

            if codec == "rawvideo":
                tail_data[fp_key] = sample_tail_zeros(fp)

            if codec in ("ffv1", "tiff"):
                known = files_by_dir.get(fp.parent, files_only)
                source_map[fp_key] = find_original_file(fp, silence=True, known_files=known)
            else:
                source_map[fp_key] = None

            frames[fp_key] = int(info.get("frames", 0) or 0)
            st = fp.stat()
            stat_map[fp_key] = st
            sizes[fp_key] = st.st_size
            formats[fp_key] = info.get("pix_fmt", "")

            # Emit progress calculation safely
            self.progress.emit(int(100 * i / total_files) if total_files else 100)

        if not self.isInterruptionRequested():
            # Package everything up and send it to the main thread
            result_data = (
                all_items_to_render, info_map, tail_data, source_map,
                stat_map, sizes, formats, codecs, frames
            )
            self.scan_finished.emit(result_data)

def list_network_drives():
    """
    Return a list of dicts for each partition that looks like a network volume.
    On Windows, we detect network drives by checking part.device.startswith("\\\\").
    On macOS/Linux, we detect via fstype ∈ NETWORK_FS_TYPES or "remote" ∈ opts.
    """
    drives = []
    try:
        for part in psutil.disk_partitions(all=True):
            # ==== Windows: UNC‐style device path (e.g. '\\\\SERVER\\Share') ====
            if sys.platform.startswith("win") and part.device.startswith(r"\\"):
                if part.device.startswith(r"\\") or 'remot' in part.opts:
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
    except Exception as e:
        logger.error(f"Error scanning system partitions: {e}")

    # Add Windows Network Locations (Shortcuts)
    if sys.platform.startswith("win"):
        drives.extend(get_windows_network_locations())

    # Remove duplicates (in case a location is both mapped AND has a shortcut)
    # We key by the 'mountpoint' (path)
    unique_drives = {d['mountpoint']: d for d in drives}.values()

    return list(unique_drives)

def find_file_on_network_drives(relative_path: str):
    """
    Given a path like 'folder/subfolder/file.ext' (POSIX style),
    this function returns a list of absolute‐normalized strings where that file exists
    on any network‐mounted drive.
    """
    found_paths = []
    rel = PurePath(relative_path)        # Treat input as “POSIX‐style” components
    for d in list_network_drives():
        try:
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
        except OSError as e:
            # This will catch [WinError 1326] (which is an OSError),
            # "Permission denied" on macOS/Linux, and other OS-level access errors.
            # We will simply "disregard the drive" by continuing to the next one.
            logger.info(f"Skipping {d.get('mountpoint')}, due to access error: {e}")
            continue
    return found_paths


def find_original_file(
    base_fp: Path,
    silence: bool = False,
    exts: tuple[str, ...] = ("avi", "tif", "tiff"),
    known_files: Iterable[Path] | None = None,
) -> Path | None:
    """
    Find the original/source file for base_fp by matching same directory + same stem
    and trying extensions in `exts` order.

    If `known_files` is provided, search is done in-memory (no disk hits).
    Otherwise falls back to filesystem existence checks.
    """
    base_fp = Path(base_fp)
    parent = base_fp.parent
    stem_cf = base_fp.stem.casefold()

    # Fast path: use in-memory file list
    if known_files is not None:
        # Build a tiny lookup for just this stem in just this dir
        # ext -> Path (ext is "avi"/"tif"/"tiff" lowercased)
        matches: dict[str, Path] = {}

        for fp in known_files:
            # If caller passes all files in the folder tree, this keeps it correct
            # and avoids false matches from other directories.
            if fp.parent != parent:
                continue
            if fp.stem.casefold() != stem_cf:
                continue

            ext = fp.suffix[1:].casefold() if fp.suffix.startswith(".") else fp.suffix.casefold()
            if ext:  # ignore files with no extension
                matches[ext] = fp

        for ext in exts:
            hit = matches.get(ext.casefold())
            if hit is not None:
                if not silence:
                    logger.info(f"Found original file {hit}")
                return hit
        return None

    # Fallback: disk checks (standalone usage)
    base_name = base_fp.stem  # keeps dots that are not the final suffix
    for ext in exts:
        candidate = parent / f"{base_name}.{ext}"
        if candidate.exists():
            if not silence:
                logger.info(f"Found original file {candidate}")
            return candidate
    return None

def get_files(folder=None, mode=None):
    folder_path = Path(folder)
    if not folder_path.is_dir():
        return []
    exts = list(supported_extensions)
    if mode == "compress":
        exts.remove("mkv")
    elif mode == "validate":
        exts = ["mkv"]
    files = [
        p for p in folder_path.rglob('*')
        if p.is_file() and p.suffix.lower().lstrip('.') in exts
    ]
    return sorted(
        files,
        key=lambda p: (
            exts.index(p.suffix.lower().lstrip('.')),
            p.name.lower()     # or p.name if you want case‐sensitive name sort
        )
    )

def load_storage_jsonl(path: Path) -> dict:
    """
    Reads a .jsonl file and merges all lines into a single dictionary.
    """
    result = {}
    if not path.exists():
        return result # Return empty dict, not None

    try:
        with path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue # skip invalid lines

                if isinstance(obj, dict):
                    result.update(obj)
    except (IOError, OSError) as e:
        # Log this error instead of crashing
        # You'll need to pass 'logger' in or import it
        print(f"Error reading storage file {path}: {e}")

    return result


def resolve_shortcut_target(shortcut_path):
    """
    Uses PowerShell to resolve the target of a Windows .lnk file.
    This avoids needing the heavy 'pywin32' library.
    """
    try:
        # PowerShell command to load WScript.Shell and read the target path
        cmd = [
            "powershell", "-NoProfile", "-Command",
            f"(New-Object -ComObject WScript.Shell).CreateShortcut('{str(shortcut_path)}').TargetPath"
        ]
        # Run command and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        target = result.stdout.strip()

        # Only return if it looks like a network path (UNC)
        if target.startswith(r"\\"):
            return target
    except Exception as e:
        logger.debug(f"Could not resolve shortcut {shortcut_path}: {e}")
    return None


def get_windows_network_locations():
    """
    Scans the Windows 'Network Shortcuts' folder for 'Add Network Location' items.
    Returns a list of dicts similar to psutil structure.
    """
    locations = []
    if not sys.platform.startswith("win"):
        return locations

    # The standard location where Windows stores "Network Locations"
    shortcut_dir = Path(os.environ["APPDATA"]) / "Microsoft" / "Windows" / "Network Shortcuts"

    if not shortcut_dir.exists():
        return locations

    for item in shortcut_dir.iterdir():
        unc_path = None

        # Case A: The item is a direct .lnk file (e.g., "MyShare.lnk")
        if item.suffix.lower() == ".lnk":
            unc_path = resolve_shortcut_target(item)

        # Case B: The item is a folder containing a 'target.lnk' (Common for WebDAV/older wizards)
        elif item.is_dir():
            target_lnk = item / "target.lnk"
            if target_lnk.exists():
                unc_path = resolve_shortcut_target(target_lnk)

        if unc_path:
            locations.append({
                "device": unc_path,  # The actual \\Server\Share path
                "mountpoint": unc_path,  # Treated as the mountpoint for search
                "fstype": "NetworkLocation",
                "opts": "rw",  # Assumed
                "origin": "shortcut"  # Just a tag to know where this came from
            })

    return locations

def _look_for_csv(path):
    p = Path(path)
    stems = [p.stem, p.stem.replace("_video", "_force"), p.stem.replace("_video", "_pressure")]
    ext = ".csv"

    for stem in stems:
        candidate = p.with_stem(stem).with_suffix(ext)
        if candidate.exists():
            logger.info(f"Found matching csv: {candidate}")
            return candidate

    return None


class CompressibleScannerThread(QThread):
    # Emits a sorted list of folder paths (strings)
    scan_finished = Signal(list)
    status_update = Signal(str)

    def __init__(self, start_dir, parent=None):
        super().__init__(parent)
        self.start_dir = start_dir

    def run(self):
        found_dirs = set()

        # os.walk is highly efficient here
        for root, dirs, files in os.walk(self.start_dir):
            if self.isInterruptionRequested():
                return  # Safely abort if user closes the app

            self.status_update.emit(f"Scanning: {root}")

            for file_name in files:
                if self.isInterruptionRequested():
                    return

                lower_file = file_name.lower()

                # Check for TIF
                if lower_file.endswith('.tif') or lower_file.endswith('.tiff'):
                    found_dirs.add(root)
                    break  # Found one! Move to the next directory

                # Check for >1GB AVI
                elif lower_file.endswith('.avi'):
                    file_path = Path(root) / file_name
                    try:
                        # St.size is in bytes. 1GB = 1024^3 bytes
                        if file_path.stat().st_size > (1024 * 1024 * 1024):
                            found_dirs.add(root)
                            break  # Found one! Move to the next directory
                    except OSError:
                        # Silently skip files we don't have permission to read
                        pass

                        # Emit the unique, sorted list of directories
        self.scan_finished.emit(sorted(list(found_dirs)))