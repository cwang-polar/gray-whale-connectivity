#!/usr/bin/env python3
"""Explicit study inspection and isolated configured reproduction entry point."""

from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from gray_whale_corridor.release.control import cli

if __name__ == "__main__":
    raise SystemExit(cli())
