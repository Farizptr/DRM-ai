"""
Building Detector Package

A tool for detecting buildings in satellite/aerial imagery using YOLOv8.
"""

__version__ = "1.0.0"
__author__ = "Building Detector Team"

# Main imports for easy access
from .core.polygon_detection import detect_buildings_in_polygon
from .core.detection import load_model, detect_buildings 