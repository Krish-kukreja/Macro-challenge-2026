"""
Environment setup helper for the bundled DREAMPlace.

The bundled DREAMPlace .so files were compiled against CUDA 11.8 toolkit
(libcudart.so.11.0). The eval Docker runs PyTorch 2.5.1+cu124 (which has
libcudart.so.12). We ship libcudart.so.11.0 in `lib/` next to the
`dreamplace/` package and make it discoverable via LD_LIBRARY_PATH.

Usage from the placer:
    from dreamplace_bundle import _bundle_env
    env = _bundle_env.subprocess_env()  # use this for subprocess.run(env=env)
    placer_py = _bundle_env.placer_py_path()
"""

import os
import sys


_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_DIR = os.path.join(_HERE, "lib")
_PLACER_PY = os.path.join(_HERE, "dreamplace", "Placer.py")


def bundle_root() -> str:
    """Return absolute path to the bundle root."""
    return _HERE


def lib_dir() -> str:
    """Return absolute path to the bundled .so libraries (libcudart, etc.)."""
    return _LIB_DIR


def placer_py_path() -> str:
    """Return absolute path to dreamplace/Placer.py."""
    return _PLACER_PY


def subprocess_env(base_env: dict = None) -> dict:
    """
    Return an env dict suitable for subprocess.run/Popen that has the
    bundled lib dir prepended to LD_LIBRARY_PATH and PYTHONPATH set so
    `import dreamplace` works inside the subprocess.
    """
    env = dict(base_env if base_env is not None else os.environ)

    ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{_LIB_DIR}:{ld}" if ld else _LIB_DIR

    py = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{_HERE}:{py}" if py else _HERE

    return env
