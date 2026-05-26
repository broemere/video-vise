#!/usr/bin/env python3
import argparse
import os
import re
import shutil
import subprocess
import sys

# --- Configuration ---
CONFIG_SCRIPT = 'config.py'
SPEC_FILE = 'build.spec'
APP_BASE_NAME = 'video_vise'


def get_version():
    """Reads the __version__ from the main application script."""
    print("--- Reading version number ---")
    version_re = r'^version\s*=\s*[\'"]([^\'"]+)[\'"]'
    with open(CONFIG_SCRIPT, "r", encoding="utf-8") as f:
        for line in f:
            match = re.match(version_re, line)
            if match:
                version = match.group(1)
                print(f"Version found: {version}\n")
                return version
    raise RuntimeError("Could not find version in the script.")


def clean():
    """Removes previous build artifacts."""
    print("--- Cleaning old build directories ---")
    for folder in ['build', 'dist', '__pycache__']:
        if os.path.exists(folder):
            shutil.rmtree(folder)
    for f in os.listdir():
        if f.startswith(APP_BASE_NAME) and f.endswith(".zip"):
            os.remove(f)
    print("Clean complete.\n")


def build(version, mac_arch=None):
    """Runs the PyInstaller command after setting the version environment variable."""
    print("--- Running PyInstaller ---")

    # Set the version in an environment variable for the spec file to read (only for this process, not system)
    env = os.environ.copy()
    env['APP_VERSION'] = version
    if sys.platform == 'darwin' and mac_arch:
        env['APP_TARGET_ARCH'] = mac_arch

    command = ['pyinstaller', SPEC_FILE, '--clean']

    print(f"Executing: {' '.join(command)}")
    # Pass the modified environment to the subprocess
    subprocess.run(command, check=True, env=env)
    print("PyInstaller build successful!\n")


def archive(version, mac_arch=None):
    """Creates a distributable ZIP archive of the build."""
    print("--- Creating distributable archive ---")
    platform = 'mac' if sys.platform == 'darwin' else 'win'
    app_versioned_name = f"{APP_BASE_NAME}_v{version}"

    # --- macOS DMG Creation ---
    if platform == 'mac':
        print("Platform is macOS. Creating .dmg...")
        source_app_path = os.path.join('dist', f"{app_versioned_name}.app")
        # Check that the .app bundle exists
        if not os.path.exists(source_app_path):
            print(f"Error: Cannot create DMG. Source app not found at:")
            print(f"{os.path.abspath(source_app_path)}")
            print("Ensure your .spec file is set to create a windowed .app bundle.")
            return

        arch_suffix = f"_{mac_arch}" if mac_arch else ""
        final_dmg_path = os.path.join('dist', f"{app_versioned_name}_{platform}{arch_suffix}.dmg")
        print(f"Creating {final_dmg_path}...")

        command = [
            'hdiutil', 'create',
            '-volname', f"{APP_BASE_NAME} v{version}",  #Name of the volume when the user opens the .dmg
            '-srcfolder', source_app_path,  # Path to the .app to include
            '-ov',
            '-format', 'UDZO',
            final_dmg_path
        ]

        print(f"Executing: {' '.join(command)}")
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"Successfully created archive: {final_dmg_path}\n")
        except subprocess.CalledProcessError as e:
            print("--- HDIUTIL FAILED ---")
            print("STDERR:", e.stderr)
            raise
        except FileNotFoundError:
            print("--- HDIUTIL FAILED ---")
            print("Error: 'hdiutil' command not found.")
            return

    # --- Windows ZIP Creation ---
    elif platform == 'win':
        print("Platform is Windows. Creating .zip...")

        # Look for the versioned folder or .exe file
        source_dir = os.path.join('dist', app_versioned_name)
        source_file = os.path.join('dist', f"{app_versioned_name}.exe")

        base_dir_to_zip = None
        if os.path.isdir(source_dir):
            base_dir_to_zip = app_versioned_name  # e.g., 'proper_v1.2.3'
            print(f"Found --onedir build: {source_dir}")
        elif os.path.isfile(source_file):
            base_dir_to_zip = f"{app_versioned_name}.exe"  # e.g., 'proper_v1.2.3.exe'
            print(f"Found --onefile build: {source_file}")

        if not base_dir_to_zip:
            print(f"Error: Could not find '{source_dir}' or '{source_file}'.")
            print("PyInstaller build may have failed or produced unexpected output.")
            return

        archive_path_without_ext = os.path.join('dist', f"{app_versioned_name}_{platform}")
        print(f"Zipping '{base_dir_to_zip}' into '{archive_path_without_ext}.zip'...")

        shutil.make_archive(
            archive_path_without_ext,
            'zip',
            root_dir='dist',
            base_dir=base_dir_to_zip
        )
        print(f"Successfully created archive: {archive_path_without_ext}.zip\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build VideoVise distributables.")
    parser.add_argument(
        "--mac-arch",
        choices=["arm64", "x86_64", "universal2"],
        default="arm64" if sys.platform == "darwin" else None,
        help="macOS target architecture for PyInstaller. Defaults to arm64 on macOS.",
    )
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        #clean()
        app_version = get_version()
        build(app_version, args.mac_arch)
        archive(app_version, args.mac_arch)
        print("✅ Build process complete!")
    except Exception as e:
        print(f"\n--- ❌ BUILD FAILED ---")
        print(f"An error occurred: {e}")
        sys.exit(1)
