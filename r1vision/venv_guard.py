"""Make `python foo.py` work regardless of which interpreter is on PATH.

Anaconda puts its own python first, and that one has neither opencv nor
cyclonedds. Without this, a probe run under the wrong interpreter reports
"no DDS" -- an environment failure wearing the costume of a robot finding.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_SENTINEL = "R1VISION_REEXEC"


def venv_python() -> Path | None:
    root = Path(__file__).resolve().parent.parent / ".venv"
    # Windows venvs put the interpreter in Scripts\python.exe, not bin/python.
    # Checking only the POSIX layout meant this reported "no .venv was found"
    # on Windows even with a complete, working .venv sitting right there --
    # an environment problem disguised as a missing one. Hit for real: running
    # from a different interpreter (a conda env with no cyclonedds) took the
    # branch that needs this function, which every same-process check this
    # session had quietly skipped by already being cyclonedds-equipped.
    for candidate in (root / "bin" / "python", root / "Scripts" / "python.exe"):
        if candidate.exists():
            return candidate
    return None


def missing_modules(required: tuple[str, ...]) -> list[str]:
    missing = []
    for name in required:
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    return missing


def ensure_venv(required: tuple[str, ...] = ("cv2",)) -> None:
    """Re-exec under the project venv if this interpreter lacks `required`."""
    missing = missing_modules(required)
    if not missing:
        return

    target = venv_python()
    if target is None:
        # && is a PowerShell parse error, and bin/pip does not exist on
        # Windows -- this message sent someone straight into a second
        # failure right after the first. One command per line works in
        # every shell this project actually gets run from.
        sys.exit(
            f"error: this interpreter ({sys.executable}) is missing "
            f"{', '.join(missing)} and no .venv was found.\n"
            "Create it with:\n"
            "  python3.12 -m venv .venv\n"
            "  .venv\\Scripts\\pip install -r requirements.txt   "
            "(Windows)\n"
            "  .venv/bin/pip install -r requirements.txt          "
            "(macOS/Linux)"
        )

    if Path(sys.executable).resolve() == target.resolve() or os.environ.get(_SENTINEL):
        # Already the venv (or we just came from it) and still short -- so the
        # venv itself is incomplete. Re-execing again would spin forever.
        sys.exit(
            f"error: {target} is missing {', '.join(missing)}.\n"
            f"Install it with:  {target} -m pip install -r requirements.txt"
        )

    print(
        f"[r1vision] {Path(sys.executable).name} lacks {', '.join(missing)}; "
        f"switching to {target}",
        file=sys.stderr,
    )
    os.environ[_SENTINEL] = "1"
    os.execv(str(target), [str(target), *sys.argv])
