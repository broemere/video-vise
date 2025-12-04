import os
import json
from pathlib import Path
from widgets.file_scanning import find_file_on_network_drives, find_original_file, get_files, load_storage_jsonl
import logging

logger = logging.getLogger(__name__)


class StorageTracker:
    def __init__(self):

        self.filesizes_reduced = {}
        self.tracker_file = "Lab/Software/storage_saved.jsonl"
        self.track_storage = True
        self.storage_path = None
        self.init_storage_tracking()

    def init_storage_tracking(self):
        logger.info(f"Looking for '{self.tracker_file}' on network drives…")
        matches = find_file_on_network_drives(self.tracker_file)
        if matches:
            logger.info(f"  → Found: {matches[0]}")
            self.tracker_file = matches[0]
            self.storage_path = Path(self.tracker_file)  # Set storage_path here
            self.filesizes_reduced = load_storage_jsonl(self.storage_path)
            self.track_storage = True
            logger.info("Tracking storage savings\n")
        else:
            logger.info("  → File not found on network, tracking disabled\n")
            self.filesizes_reduced = {}  # Still initialize the dict
            self.track_storage = False
            self.storage_path = None  # Explicitly set to None

    def get_gb_saved(self):
        """Return the sum of all numeric values in `data`."""
        self.total_gb_saved = sum(self.filesizes_reduced.values())
        return self.total_gb_saved

    def add_member_jsonl(self, key: str, value) -> None:
        """Append a new JSON line: {key: value}."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        entry = {key: value}
        with self.storage_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()  # push Python’s internal buffer out to the OS
            os.fsync(f.fileno())  # ask the OS to flush its buffers all the way to the physical file