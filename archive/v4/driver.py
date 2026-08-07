#!/usr/bin/env python3
"""Alias for drive.py so `python driver.py` works."""
from drive import *  # noqa: F401,F403
import drive as _drive

if __name__ == "__main__":
    _drive.main()
