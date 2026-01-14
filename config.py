APP_NAME = "VideoVise"
version = "1.6"
ORG = "TykockiLab"
EXT_COLOR = {
    ".tif": "#4e84af",  # FIJI/ImageJ color
    ".tiff": "#4e84af",
    ".avi": "#FFB900",  # burnt orange (#FF5500)
    ".mkv": "#9c27b0",  # deep purple (lighter: #9575cd)
}
supported_extensions = ["avi", "tif", "tiff", "mkv"]
DEFAULT_FPS = 10  # Make this a user settable option in the UI?
REPO_URL = "https://github.com/broemere/video-vise/releases/latest"