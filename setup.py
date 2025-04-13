from setuptools import setup, find_packages, Extension
import os
import sys
import subprocess
from setuptools.command.build_ext import build_ext

# Custom build for CMake
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        # Ensure nanobind is installed
        try:
            import nanobind
            print(f"Found nanobind at {nanobind.__file__}")
        except ImportError:
            print("Installing nanobind")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "nanobind"])
        
        # Get nanobind CMAKE directory
        try:
            import nanobind
            nanobind_root = subprocess.check_output(
                [sys.executable, "-m", "nanobind", "--cmake_dir"],
                text=True
            ).strip()
            print(f"Using nanobind CMAKE dir: {nanobind_root}")
        except Exception as e:
            print(f"Error finding nanobind CMAKE dir: {e}")
            raise
        
        # Build with CMake
        build_temp = os.path.abspath(self.build_temp)
        os.makedirs(build_temp, exist_ok=True)
        
        # Configure
        cmake_args = [
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-Dnanobind_DIR={nanobind_root}",
            f"-DCMAKE_PREFIX_PATH={nanobind_root}"
        ]
        
        print(f"Running CMake with args: {cmake_args}")
        
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args,
            cwd=build_temp
        )
        
        # Build
        subprocess.check_call(
            ["cmake", "--build", ".", "--config", "Release"],
            cwd=build_temp
        )
        
        # Get the right directory structure
        # Make sure the extension directory exists
        ext_fullpath = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(ext_fullpath), exist_ok=True)
        
        # Try to find and copy the built extension file
        import glob
        built_files = glob.glob(os.path.join(build_temp, "**/*.so"), recursive=True)
        
        if not built_files:
            print("Warning: Could not find built extension file.")
            print("Files in build directory:")
            for root, dirs, files in os.walk(build_temp):
                for file in files:
                    print(os.path.join(root, file))
            raise RuntimeError("Could not find built extension file")
        
        # Copy the file
        import shutil
        for built_file in built_files:
            print(f"Copying {built_file} -> {ext_fullpath}")
            shutil.copy(built_file, ext_fullpath)

setup(
    name="optimizing_wfc",
    version="0.1.0",
    author="Your Name",
    packages=find_packages(),
    ext_modules=[CMakeExtension("wfc_cpp", sourcedir="cpp")],
    cmdclass=dict(build_ext=CMakeBuild),
    install_requires=[
        "numpy",
        "gymnasium",
        "stable_baselines3",
        "torch",
        "nanobind",  # Add nanobind as a dependency
    ],
    python_requires=">=3.8",
)