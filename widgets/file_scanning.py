import sys
from pathlib import Path, PurePath
import logging
from config import supported_extensions
import psutil
import json
import subprocess
import os

logger = logging.getLogger(__name__)


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

def find_original_file(base_fp: Path, silence=False, exts = ('avi', 'tif', 'tiff')):
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
