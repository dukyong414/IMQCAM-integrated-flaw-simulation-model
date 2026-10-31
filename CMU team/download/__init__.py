# download/__init__.py

"""
Download module for Figshare data handling.

This package provides utilities for downloading and merging
datasets from public Figshare articles.
"""

from .fetch import downloaddata, mergefiles, getdata

__all__ = ["downloaddata", "mergefiles", "getdata"]
