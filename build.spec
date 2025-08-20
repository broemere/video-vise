# -*- mode: python ; coding: utf-8 -*-

import sys
import os

# --- Read Version from Environment Variable ---
# Get the version set by build.py. Fallback to '0.0.0' if not set.
app_version = os.environ.get('APP_VERSION', '0.0.0')
app_name = f'video_vise_v{app_version}'

# --- Platform-specific settings ---
if sys.platform == 'darwin':
    platform_binaries = [('ffmpeg/ffmpeg', '.'), ('ffmpeg/ffprobe', '.')]
    icon_file = os.path.join('icons', 'app.icns')
else:
    platform_binaries = [('ffmpeg/ffmpeg.exe', '.'), ('ffmpeg/ffprobe.exe', '.')]
    icon_file = os.path.join('icons', 'app.ico')

platform_datas = [(icon_file, 'icons')]


# --- PyInstaller Analysis ---
a = Analysis(
    ['video_vise.py'],
    pathex=[],
    binaries=platform_binaries,
    datas=platform_datas,
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        'PySide6.QtSql',
        'PySide6.QtNetwork',
        'PySide6.QtXml'
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)


# --- Platform-specific Build Process ---
# All 'name' parameters now use the dynamic app_name variable.
if sys.platform == 'darwin':
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name=app_name, # <-- DYNAMIC
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,
        icon=icon_file,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name=app_name, # <-- DYNAMIC
    )
    app = BUNDLE(
        coll,
        name=f"{app_name}.app", # <-- DYNAMIC
        icon=icon_file,
        bundle_identifier=None,
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name=app_name, # <-- DYNAMIC
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=True,
        icon=icon_file,
    )