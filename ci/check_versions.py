#!/usr/bin/env python3
"""Require every QMBED distribution surface to share one version."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path


FILES = {
    "rust": (Path("Cargo.toml"), ("package", "version")),
    "capi": (Path("bindings/capi/Cargo.toml"), ("package", "version")),
    "python": (Path("bindings/python/pyproject.toml"), ("project", "version")),
    "julia": (Path("bindings/julia/Project.toml"), ("version",)),
}


def read_version(path: Path, keys: tuple[str, ...]) -> str:
    value = tomllib.loads(path.read_text())
    for key in keys:
        value = value[key]
    if not isinstance(value, str):
        raise TypeError(f"{path} version is not a string")
    return value


def check_versions(root: Path) -> str:
    versions = {
        name: read_version(root / path, keys)
        for name, (path, keys) in FILES.items()
    }
    if len(set(versions.values())) != 1:
        details = ", ".join(
            f"{name}={version}" for name, version in versions.items()
        )
        raise ValueError(f"QMBED versions differ: {details}")
    return next(iter(versions.values()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--tag")
    args = parser.parse_args()
    version = check_versions(args.root)
    if args.tag is not None and args.tag != f"v{version}":
        raise SystemExit(
            f"release tag {args.tag!r} does not match package version v{version}"
        )
    print(version)


if __name__ == "__main__":
    main()
