#!/usr/bin/env python3
"""Hakken-style knowledge prediction toolkit - entry point (Python 3.8+, stdlib only).

    python3 scripts/hakken.py -h
"""
import os
import sys

# never write into the (possibly shared or read-only) skill directory
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hakkenlib.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
