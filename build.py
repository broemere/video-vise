#!/usr/bin/env python3
import subprocess, platform, shutil, os, re

# Clean up
# for d in ("build", "dist"):
#     shutil.rmtree(d, ignore_errors=True)
# for f in [f for f in os.listdir() if f.endswith(".spec")]:
#     os.remove(f)

VERSION_RE  = r'^__version__\s*=\s*[\'"]([^\'"]+)[\'"]'

py = "pyinstaller"
base = "video_vise"
script = f"{base}.py"

# Read version from top of your script without importing it
with open(script, "r", encoding="utf-8") as f:
    for line in f:
        m = re.match(VERSION_RE, line)
        if m:
            version = m.group(1)
            break
    else:
        version = "0.0.0"  # fallback

out_name = f"{base}_v{version}"

cmd = [
    "pyinstaller",
    "--onefile",
    "--name", out_name
]

# Platform‐specific data and icons
system = platform.system()
if system == "Windows":
    cmd += [
        "--add-data", "icons/app.ico;icons",
        "--add-binary", "ffmpeg/ffmpeg.exe;.",
        "--add-binary", "ffmpeg/ffprobe.exe;.",
        "--icon", "icons/app.ico"
    ]
elif system == "Darwin":
    cmd += [
        "--add-data", "icons/app.icns:icons",
        "--icon", "icons/app.icns"
    ]
else:  # assume Linux
    cmd += [
        "--add-data", "icons/app.png:icons",
        "--icon", "icons/app.png"
    ]

# Finally, specify the script to bundle
cmd.append(script)

print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True)
print("Done — results in dist/")