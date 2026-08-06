#!/usr/bin/env python3
"""Dump a vendor SDK's real API. Run this ON the workstation. READ ONLY.

    python tools/inspect_sdk.py neurapy
    python tools/inspect_sdk.py neurapy --deep
    python tools/inspect_sdk.py neurapy > maira-sdk.txt

It imports the package, walks its classes and prints every public method with
its real signature and first docstring line. Nothing is instantiated and nothing
is called, so no robot moves and no connection is opened -- importing is the
only side effect, and that is what `--no-import-submodules` is for if even that
is unwelcome.

The point is to replace a conversation about what the SDK might be called with
the actual names, so the adapter can be written against real signatures instead
of guesses.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import pkgutil
import sys

# Method names worth finding, grouped by what the adapter needs them for. Purely
# a highlighting aid -- everything public is printed regardless.
INTERESTING = {
    "connect":  ("connect", "init", "login", "open", "start", "enable",
                 "switch_to_automatic", "power_on"),
    "motion":   ("move", "movej", "movel", "move_joint", "move_linear",
                 "move_to", "goto", "set_pose", "plan", "execute", "run"),
    "gripper":  ("grip", "gripper", "grasp", "pick", "place", "open", "close",
                 "release", "suction", "tool"),
    "stop":     ("stop", "halt", "abort", "estop", "emergency", "pause",
                 "cancel", "brake"),
    "state":    ("state", "status", "get_", "read", "pose", "joint", "tcp",
                 "position", "is_", "error"),
    "reset":    ("reset", "clear", "recover", "ack"),
}


def bucket(name: str) -> str | None:
    low = name.lower()
    for label, needles in INTERESTING.items():
        if any(n in low for n in needles):
            return label
    return None


def show_callable(obj, name: str, indent: str = "    ") -> None:
    try:
        sig = str(inspect.signature(obj))
    except (ValueError, TypeError):
        sig = "(...)"
    tag = bucket(name)
    marker = f"  <-- {tag}" if tag else ""
    print(f"{indent}{name}{sig}{marker}")
    doc = inspect.getdoc(obj)
    if doc:
        first = doc.strip().splitlines()[0]
        print(f"{indent}    {first[:100]}")


def describe_class(cls, deep: bool) -> None:
    print(f"\n  class {cls.__name__}")
    doc = inspect.getdoc(cls)
    if doc:
        for line in doc.strip().splitlines()[:3]:
            print(f"      {line[:100]}")
    members = [(n, m) for n, m in inspect.getmembers(cls)
               if not n.startswith("_") and callable(m)]
    if not members:
        print("      (no public methods)")
    for name, member in members:
        if deep or bucket(name):
            show_callable(member, name, indent="      ")


def sig_from_ast(node) -> str:
    """Rebuild a signature from an AST node, without ast.unparse.

    ast.unparse landed in 3.9 and this has to run under whatever interpreter
    the vendor's venv was built with -- 3.8 here.
    """
    a = node.args
    parts: list[str] = []
    positional = list(getattr(a, "posonlyargs", [])) + list(a.args)
    defaults = list(a.defaults)
    pad = len(positional) - len(defaults)
    for i, arg in enumerate(positional):
        if arg.arg == "self":
            continue
        text = arg.arg
        if i >= pad:
            text += "=..."
        parts.append(text)
    if a.vararg:
        parts.append("*" + a.vararg.arg)
    for arg, default in zip(a.kwonlyargs, a.kw_defaults):
        parts.append(arg.arg + ("=..." if default is not None else ""))
    if a.kwarg:
        parts.append("**" + a.kwarg.arg)
    return "(" + ", ".join(parts) + ")"


def scan_source(pkg_dir, deep: bool) -> None:
    """Parse every .py in the package and list what it defines.

    Reading the source beats importing for two reasons here. Nothing executes,
    so a package that opens sockets or probes hardware on import stays inert.
    And methods attached dynamically -- which `class Robot` for a "socket
    interface" almost certainly does -- are invisible to inspect.getmembers on
    the class but plain to see in the file.
    """
    import ast
    import os

    files = []
    for root, _dirs, names in os.walk(pkg_dir):
        if "__pycache__" in root:
            continue
        files += [os.path.join(root, n) for n in sorted(names) if n.endswith(".py")]

    print(f"\nscanning {len(files)} source files under {pkg_dir}\n")
    for path in files:
        try:
            tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
        except SyntaxError as exc:
            print(f"  (skip {os.path.basename(path)}: {exc})")
            continue

        rel = os.path.relpath(path, pkg_dir)
        blocks: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = []
                for item in node.body:
                    if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    if item.name.startswith("_") and item.name != "__init__":
                        continue
                    tag = bucket(item.name)
                    if not deep and not tag and item.name != "__init__":
                        continue
                    marker = f"  <-- {tag}" if tag else ""
                    line = f"      {item.name}{sig_from_ast(item)}{marker}"
                    doc = ast.get_docstring(item)
                    if doc:
                        line += f"\n          {doc.strip().splitlines()[0][:96]}"
                    methods.append(line)
                if methods:
                    head = f"  class {node.name}"
                    cdoc = ast.get_docstring(node)
                    if cdoc:
                        head += f"\n      {cdoc.strip().splitlines()[0][:96]}"
                    blocks.append(head + "\n" + "\n".join(methods))

        if blocks:
            print(f"{'-'*70}\n{rel}\n{'-'*70}")
            print("\n".join(blocks))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("package", help="e.g. neurapy, or neurapy.robot")
    ap.add_argument("--deep", action="store_true",
                    help="print every public method, not only likely-relevant ones")
    ap.add_argument("--source", action="store_true",
                    help="parse the package's .py files instead of importing. "
                         "Finds dynamically-attached methods and runs no vendor "
                         "code at all")
    ap.add_argument("--no-import-submodules", action="store_true")
    args = ap.parse_args()

    print(f"python   : {sys.version.split()[0]}  ({sys.executable})")
    print(f"platform : {sys.platform}")

    try:
        pkg = importlib.import_module(args.package)
    except Exception as exc:
        print(f"\nFAILED to import {args.package!r}: {type(exc).__name__}: {exc}")
        print("\nRun this with the same interpreter the NEURA SDK is installed in.")
        return 1

    print(f"package  : {args.package}")
    print(f"version  : {getattr(pkg, '__version__', '(no __version__)')}")
    pkg_file = getattr(pkg, "__file__", "") or ""
    print(f"path     : {pkg_file or '(namespace package)'}")

    # A package living under .../pythonX.Y/site-packages while this interpreter
    # is some other X.Y means the agent will not import what was inspected here.
    import re as _re
    m = _re.search(r"python(\d+\.\d+)", pkg_file)
    if m and m.group(1) != ".".join(sys.version.split()[0].split(".")[:2]):
        print(f"\n  !! MISMATCH: package sits in python{m.group(1)} but this is "
              f"python{'.'.join(sys.version.split()[0].split('.')[:2])}.")
        print("     Re-run with the venv's own interpreter:")
        print("       .venv/bin/python tools/inspect_sdk.py ...")

    if args.source and pkg_file:
        import os
        scan_source(os.path.dirname(pkg_file), args.deep)
        return 0

    modules = [(args.package, pkg)]
    if not args.no_import_submodules and hasattr(pkg, "__path__"):
        print("\nsubmodules:")
        for info in pkgutil.iter_modules(pkg.__path__):
            full = f"{args.package}.{info.name}"
            print(f"  {full}")
            try:
                modules.append((full, importlib.import_module(full)))
            except Exception as exc:
                print(f"      (import failed: {type(exc).__name__}: {exc})")

    seen: set[int] = set()
    for mod_name, mod in modules:
        classes = [(n, c) for n, c in inspect.getmembers(mod, inspect.isclass)
                   if getattr(c, "__module__", "").startswith(args.package.split(".")[0])
                   and id(c) not in seen]
        funcs = [(n, f) for n, f in inspect.getmembers(mod, inspect.isfunction)
                 if not n.startswith("_")
                 and getattr(f, "__module__", "").startswith(args.package.split(".")[0])]
        if not classes and not funcs:
            continue
        print(f"\n{'='*70}\n{mod_name}\n{'='*70}")
        for _n, cls in classes:
            seen.add(id(cls))
            describe_class(cls, args.deep)
        for name, fn in funcs:
            if args.deep or bucket(name):
                show_callable(fn, name, indent="  ")

    print(f"\n{'='*70}")
    print("Send this output back. The adapter is written against these exact")
    print("names -- see SDK-BINDING markers in agent/adapters/maira.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
