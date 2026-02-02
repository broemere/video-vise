# VideoVise

A simple GUI tool to batch-convert image sequences (AVI/TIFF) into lossless MKV videos using FFmpeg, with built-in validation and decompression back to AVI/TIF.

---

## Features

- **Batch conversion** of AVI or multi-page TIFF files into MKV  
- **Validation** step to confirm no pixel data was lost or altered  
- **Decompress** back to AVI or TIF for legacy workflows (e.g. ImageJ)  
- A clean GUI with comprehensive file metadata details/inspection

---

## Requirements

### Running app

* Windows
* Mac

### Building app

- **Python 3.7+** with the packages in `requirements.txt`
- **FFmpeg & FFprobe** binaries  
  - Either on your system `PATH` (via your OS package manager)  
  - Or placed in a local `ffmpeg/` folder (see Installation)  

---

## Installation

### Direct run

* Download and run the exe/dmg from https://github.com/broemere/video-vise/releases
* Your OS may block the program the first time it is run.
  `"Windows protected your PC` → `More info` → `Run anyway`

### Building from source code

### 1. Clone the repository

```bash
git clone https://github.com/broemere/video-vise.git
```
or

Direct download: https://github.com/broemere/video-vise/releases


### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```
   
### 3. **Set up FFmpeg**

1. Download the appropriate **FFmpeg** build for your OS from 
https://ffmpeg.org/download.html
2. Copy the FFmpeg and FFprobe executables to the ```ffmpeg/``` directory in the project root so you have:
    ```
    ffmpeg/ffmpeg(.exe on Windows)
    ffmpeg/ffprobe(.exe on Windows)
    ```

### 4. Run the app
You can either:

* Run directly from source
    ```bash
    python video_vise.py
    ```
  
or

* Build a standalone executable
    ```bash
    python build.py
    ```
    Run ```video_vise_<version>.exe``` in ```/dist```
    * This executable is fully self-contained, can be copied/moved, and no longer requires the source code or build environment.

---

## Usage
1. Launch the GUI
2. Select a folder containing your image data (avi or tiff files)
3. Click Convert to compress the raw data into a lossless mkv video
4. Click Validate to double-check that the conversion was successful and there was no pixel data loss
5. Safely delete your original image data
6. Click Decompress to convert the mkv back to an avi/tif if necessary (for legacy hardware/software like ImageJ)

---

## License

**MIT License** -- see ```LICENSE.md``` for details.

Third-party components are distributed under their own terms, see ```LICENSES/```:

* **psutil**: BSD-3-Clause 
* **tifffile**: BSD-3-Clause 
* **PySide6 / Qt**: LGPL-3.0
* **FFmpeg / FFprobe**: LGPL-2.1+
