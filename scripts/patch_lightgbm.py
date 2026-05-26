#!/usr/bin/env python
"""Make LightGBM importable on macOS without homebrew.

LightGBM's wheel links ``@rpath/libomp.dylib`` (the OpenMP runtime) but only searches
homebrew/macports rpaths, which this environment does not have. scikit-learn already
vendors a compatible ``libomp.dylib`` in ``sklearn/.dylibs/``, so we add an rpath to
LightGBM's library pointing at it (a ``@loader_path``-relative path, stable within the
venv). Idempotent and safe to re-run; invoked from ``make install`` after pip installs.

No-op on non-macOS or if LightGBM already imports.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path


def _imports_ok() -> bool:
    return subprocess.run([sys.executable, "-c", "import lightgbm"],
                          capture_output=True).returncode == 0


def _lightgbm_lib() -> Path | None:
    """Path to ``lib_lightgbm.dylib`` without importing lightgbm (its import is what fails)."""
    spec = importlib.util.find_spec("lightgbm")
    if spec is None or not spec.origin:
        return None
    matches = list(Path(spec.origin).parent.rglob("lib_lightgbm.dylib"))
    return matches[0] if matches else None


def main() -> int:
    if sys.platform != "darwin":
        print("patch_lightgbm: non-macOS, nothing to do")
        return 0
    if _imports_ok():
        print("patch_lightgbm: lightgbm already imports — no patch needed")
        return 0

    try:
        import sklearn  # vendors libomp.dylib
    except ImportError:
        print("patch_lightgbm: scikit-learn not installed; cannot source libomp", file=sys.stderr)
        return 1

    sk_dylibs = Path(sklearn.__file__).parent / ".dylibs"
    if not (sk_dylibs / "libomp.dylib").exists():
        print(f"patch_lightgbm: no libomp.dylib under {sk_dylibs}", file=sys.stderr)
        return 1

    lib = _lightgbm_lib()
    if lib is None:
        print("patch_lightgbm: could not locate lib_lightgbm.dylib", file=sys.stderr)
        return 1
    rel = os.path.relpath(sk_dylibs, lib.parent)
    rpath = f"@loader_path/{rel}"

    existing = subprocess.run(["otool", "-l", str(lib)], capture_output=True, text=True).stdout
    if rpath in existing:
        print(f"patch_lightgbm: rpath already present ({rpath})")
    else:
        subprocess.run(["install_name_tool", "-add_rpath", rpath, str(lib)], check=True)
        print(f"patch_lightgbm: added rpath {rpath} -> {lib}")

    if _imports_ok():
        print("patch_lightgbm: lightgbm imports OK")
        return 0
    print("patch_lightgbm: still failing after patch", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
