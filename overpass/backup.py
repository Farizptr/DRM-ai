import requests
import json
import os
import numpy as np
from PIL import Image, ImageDraw
import mercantile
from shapely.geometry import Polygon, box
import math
import io
import random
import time
import argparse
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description='Create YOLOv8 dataset from OpenStreetMap building data')
    parser.add_argument('--zoom', type=int, default=18, help='Zoom level for satellite imagery (default: 18)')
    parser.add_argument('--train', type=float, default=0.7, help='Ratio of training data (default: 0.7)')
    parser.add_argument('--val', type=float, default=0.15, help='Ratio of validation data (default: 0.15)')
    parser.add_argument('--test', type=float, default=0.15, help='Ratio of test data (default: 0.15)')
    parser.add_argument('--output', type=str, default='dataset', help='Output directory (default: dataset)')
    parser.add_argument('--min-size', type=float, default=0.01, help='Minimum normalized size of buildings (default: 0.01)')
    return parser.parse_args()

# Create necessary directories for YOLOv8 dataset
def create_dataset_directories(output_dir):
    os.makedirs(f"{output_dir}/images/train", exist_ok=True)
    os.makedirs(f"{output_dir}/images/val", exist_ok=True)
    os.makedirs(f"{output_dir}/images/test", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/train", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/val", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/test", exist_ok=True)

# Fetch OSM data using Overpass API
def fetch_osm_data(query, max_retries=3, retry_delay=5):
    overpass_url = "https://overpass-api.de/api/interpreter"
    
    for attempt in range(max_retries):
        try:
            response = requests.post(overpass_url, data={"data": query})
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:  # Too Many Requests
                print(f"Rate limited. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"Error: {response.status_code}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
        except Exception as e:
            print(f"Exception: {e}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    
    raise Exception(f"Failed to fetch OSM data after {max_retries} attempts")

# Extract building polygons from OSM data
def extract_buildings(osm_data):
    # Dictionary to store node coordinates
    nodes = {}
    for element in osm_data["elements"]:
        if element["type"] == "node":
            nodes[element["id"]] = (element["lon"], element["lat"])
    
    # Extract building polygons
    buildings = []
    for element in osm_data["elements"]:
        if element["type"] == "way" and "tags" in element and "building" in element["tags"]:
            # Get coordinates of all nodes in the way
            coords = []
            for node_id in element["nodes"]:
                if node_id in nodes:
                    coords.append(nodes[node_id])
            
            # Create a polygon if we have at least 3 points
            if len(coords) >= 3:
                try:
                    buildings.append(Polygon(coords))
                except Exception as e:
                    print(f"Error creating polygon: {e}")
    
    return buildings

# Determine which tiles at zoom level 18 we need to download
def get_required_tiles(buildings, zoom=18):
    tiles = set()
    
    # For each building polygon
    for building in buildings:
        # Get the bounds of the building
        minx, miny, maxx, maxy = building.bounds
        
        # Convert bounds to tile coordinates
        # Note: mercantile uses (lat, lon) order, but our coordinates are (lon, lat)
        min_tile = mercantile.tile(minx, miny, zoom)
        max_tile = mercantile.tile(maxx, maxy, zoom)
        
        # Add all tiles that cover the building
        for x in range(min_tile.x, max_tile.x + 1):
            for y in range(min_tile.y, max_tile.y + 1):
                tiles.add((x, y, zoom))
    
    return list(tiles)

# Download satellite imagery for a tile
def download_tile(x, y, zoom, max_retries=3, retry_delay=2):
    # We'll use the OpenStreetMap tile server
    # Note: For a production system, you should use a proper tile provider with appropriate attribution
    url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
    
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers={'User-Agent': 'YOLOv8DatasetCreator/1.0'})

            if response.status_code == 200:
                return Image.open(io.BytesIO(response.content))
            elif response.status_code == 429:  # Too Many Requests
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                time.sleep(retry_delay)
        except Exception:
            time.sleep(retry_delay)
    
    raise Exception(f"Failed to download tile after {max_retries} attempts")

# Convert building polygons to YOLOv8 annotation format for a specific tile
def create_yolo_annotations(buildings, tile_x, tile_y, zoom=18, min_size=0.01):
    # Get the bounds of the tile
    tile_bounds = mercantile.bounds(tile_x, tile_y, zoom)
    # Convert to shapely polygon
    tile_polygon = box(tile_bounds.west, tile_bounds.south, tile_bounds.east, tile_bounds.north)
    
    annotations = []
    
    # For each building polygon
    for building in buildings:
        # Check if the building intersects with the tile
        if building.intersects(tile_polygon):
            # Get the intersection of the building with the tile
            intersection = building.intersection(tile_polygon)
            
            # Get the bounds of the intersection
            minx, miny, maxx, maxy = intersection.bounds
            
            # Convert to normalized coordinates (0-1)
            x_min_norm = (minx - tile_bounds.west) / (tile_bounds.east - tile_bounds.west)
            y_min_norm = 1 - (maxy - tile_bounds.south) / (tile_bounds.north - tile_bounds.south)  # Flip Y axis
            x_max_norm = (maxx - tile_bounds.west) / (tile_bounds.east - tile_bounds.west)
            y_max_norm = 1 - (miny - tile_bounds.south) / (tile_bounds.north - tile_bounds.south)  # Flip Y axis
            
            # Calculate center and dimensions (YOLO format)
            x_center = (x_min_norm + x_max_norm) / 2
            y_center = (y_min_norm + y_max_norm) / 2
            width = x_max_norm - x_min_norm
            height = y_max_norm - y_min_norm
            
            # Class ID for buildings (assuming 0)
            class_id = 0
            
            # Add to annotations if the building is not too small
            if width > min_size and height > min_size:
                annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")
    
    return annotations

# Process tiles for a specific split (train/val/test)
def process_tiles(tiles, buildings, split, output_dir, zoom=18, min_size=0.01):
    successful_tiles = 0
    for i, (x, y, z) in enumerate(tqdm(tiles, desc=f"Processing {split} tiles")):
        try:
            # Download tile image
            img = download_tile(x, y, z)
            
            # Create YOLOv8 annotations
            annotations = create_yolo_annotations(buildings, x, y, z, min_size)
            
            # Skip if no annotations (no buildings in this tile)
            if not annotations:
                continue
            
            # Save image
            img_path = f"{output_dir}/images/{split}/{x}_{y}_{z}.png"
            img.save(img_path)
            
            # Save annotations
            label_path = f"{output_dir}/labels/{split}/{x}_{y}_{z}.txt"
            with open(label_path, "w") as f:
                f.write("\n".join(annotations))
                
            successful_tiles += 1
            
            # Add a small delay to avoid rate limiting
            time.sleep(0.1)
        except Exception as e:
            print(f"Error processing tile {x}, {y}, {z}: {e}")
    
    return successful_tiles

# Create dataset.yaml file for YOLOv8
def create_dataset_yaml(output_dir):
    yaml_content = f"""
path: {output_dir}
train: images/train
val: images/val
test: images/test

nc: 1
names: ['building']
"""
    with open(f"{output_dir}.yaml", "w") as f:
        f.write(yaml_content)

# Main function to create the YOLOv8 dataset
def create_yolov8_dataset(overpass_query, zoom=18, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, output_dir="dataset5", min_size=0.01):
    # Create dataset directories
    create_dataset_directories(output_dir)
    
    # Fetch OSM data
    print("Fetching OSM data...")
    osm_data = fetch_osm_data(overpass_query)
    
    # Extract building polygons
    print("Extracting building polygons...")
    buildings = extract_buildings(osm_data)
    print(f"Found {len(buildings)} buildings")
    
    # Determine required tiles
    print(f"Determining required tiles at zoom level {zoom}...")
    tiles = get_required_tiles(buildings, zoom)
    print(f"Need to download {len(tiles)} tiles")
    
    # Shuffle tiles for random train/val/test split
    random.shuffle(tiles)
    
    # Calculate split indices
    train_end = int(len(tiles) * train_ratio)
    val_end = train_end + int(len(tiles) * val_ratio)
    
    # Split tiles into train/val/test
    train_tiles = tiles[:train_end]
    val_tiles = tiles[train_end:val_end]
    test_tiles = tiles[val_end:]
    
    # Process tiles
    train_count = process_tiles(train_tiles, buildings, "train", output_dir, zoom, min_size)
    val_count = process_tiles(val_tiles, buildings, "val", output_dir, zoom, min_size)
    test_count = process_tiles(test_tiles, buildings, "test", output_dir, zoom, min_size)
    
    # Create dataset.yaml file for YOLOv8
    create_dataset_yaml(output_dir)
    
    print(f"\nYOLOv8 dataset created successfully!")
    print(f"Train images: {train_count}")
    print(f"Validation images: {val_count}")
    print(f"Test images: {test_count}")
    print(f"Total images: {train_count + val_count + test_count}")
    print(f"Dataset configuration saved to {output_dir}.yaml")

# Run the main function with the provided Overpass query
if __name__ == "__main__":
    args = parse_arguments()
    
    overpass_query = """
    [out:json];
    (
    way["building"](-6.9500,107.5900,-6.9000,107.6500); // Covers central Bandung, larger area
    );
    out body;
    >;
    out skel qt;


    """
    
    create_yolov8_dataset(
        overpass_query,
        zoom=args.zoom,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        output_dir=args.output,
        min_size=args.min_size
    )
