"""
Evaluation module for building detection accuracy testing.

This module provides tools for:
- Fetching ground truth data from OpenStreetMap
- Comparing model results with reference data
- Computing accuracy metrics
- Generating evaluation reports
"""

__version__ = "1.0.0"
__author__ = "Building Detector Team"

from .overpass_client import OverpassClient

__all__ = ["OverpassClient"]
