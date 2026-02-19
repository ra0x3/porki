#!/usr/bin/env python3
import sys

import tomllib


def get_version(filepath="pyproject.toml"):
    with open(filepath, "rb") as f:
        data = tomllib.load(f)

    version = data.get("project", {}).get("version") or data.get("tool", {}).get("poetry", {}).get(
        "version"
    )

    if not version:
        raise SystemExit("Could not determine package version from pyproject.toml")

    return version


def get_version_from_stdin():
    raw = sys.stdin.read()
    if not raw.strip():
        print("")
        sys.exit(0)

    try:
        data = tomllib.loads(raw)
    except Exception:
        print("")
        sys.exit(0)

    version = data.get("project", {}).get("version") or data.get("tool", {}).get("poetry", {}).get(
        "version"
    )
    print(version or "")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--stdin":
        get_version_from_stdin()
    else:
        print(get_version())
