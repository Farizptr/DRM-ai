import os
import json
import sys
import tempfile
import time
from tqdm import tqdm
import concurrent.futures
from functools import partial
from shapely.geometry import box
from shapely.ops import unary_union # Added for merging
import threading # Added for thread lock
import glob

from .detection import load_model, detect_buildings
from ..utils.geojson_utils import load_geojson, extract_polygon, create_example_geojson
from .tile_utils import get_tile_bounds, get_tiles_for_polygon, get_tile_image, process_tile_detections
from ..visualization.visualization import visualize_polygon_detections
from ..utils.building_export import save_buildings_to_json, save_buildings_simple_format

# --- Incremental tile saving functions ---
def save_tile_results(tile_detections, output_dir, tile_id):
    """
    Save individual tile detection results to JSON file.
    Saves both detailed format (for recovery) and simple format (for final output).
    
    Args:
        tile_detections: Detection results for a single tile
        output_dir: Directory to save tile results
        tile_id: Tile identifier (z/x/y format)
    """
    # Create tiles subdirectory
    tiles_dir = os.path.join(output_dir, "tiles")
    os.makedirs(tiles_dir, exist_ok=True)
    
    # Clean tile_id for filename (replace / with _)
    safe_tile_id = tile_id.replace("/", "_")
    
    # Save detailed format for recovery
    tile_file_detailed = os.path.join(tiles_dir, f"tile_{safe_tile_id}.json")
    tile_data_detailed = {
        'tile': tile_detections['tile'],
        'bounds': tile_detections['bounds'],
        'detections': tile_detections['detections'],
        'boxes': tile_detections['boxes'],
        'confidences': tile_detections['confidences'],
        'class_ids': tile_detections['class_ids'],
        'processed_at': time.time()
    }
    
    # Save simple format for final output
    tile_file_simple = os.path.join(tiles_dir, f"tile_{safe_tile_id}_simple.json")
    tile_data_simple = convert_tile_to_simple_format(tile_detections)
    
    try:
        # Save detailed format
        with open(tile_file_detailed, 'w') as f:
            json.dump(tile_data_detailed, f, indent=2)
            
        # Save simple format
        with open(tile_file_simple, 'w') as f:
            json.dump(tile_data_simple, f, indent=2)
            
        print(f"Saved tile {tile_id} with {tile_detections['detections']} detections")
        print(f"  - Detailed: {tile_file_detailed}")
        print(f"  - Simple: {tile_file_simple}")
    except Exception as e:
        print(f"Error saving tile {tile_id}: {e}")

def convert_tile_to_simple_format(tile_detections):
    """
    Convert tile detections to simple format with id, longitude, latitude.
    
    Args:
        tile_detections: Detection results for a single tile
        
    Returns:
        List of buildings with id, longitude, latitude
    """
    tile_id_str = tile_detections.get('tile', 'unknown_tile')
    bounds = tile_detections['bounds']  # west, south, east, north
    west, south, east, north = bounds
    tile_width_deg = east - west
    tile_height_deg = north - south
    
    img_width, img_height = 256, 256  # Tile image size
    
    simple_buildings = []
    
    for i, bbox_coords in enumerate(tile_detections['boxes']):
        if len(bbox_coords) != 4:
            continue
            
        x1_pixel, y1_pixel, x2_pixel, y2_pixel = bbox_coords
        
        # Convert pixel coordinates to geographic coordinates
        x1_norm_tile = x1_pixel / img_width
        y1_norm_tile = y1_pixel / img_height
        x2_norm_tile = x2_pixel / img_width
        y2_norm_tile = y2_pixel / img_height
        
        # Convert to geographic coordinates
        geo_x1 = west + x1_norm_tile * tile_width_deg
        geo_y1 = south + (1 - y2_norm_tile) * tile_height_deg
        geo_x2 = west + x2_norm_tile * tile_width_deg
        geo_y2 = south + (1 - y1_norm_tile) * tile_height_deg
        
        # Calculate centroid
        centroid_lon = (geo_x1 + geo_x2) / 2
        centroid_lat = (geo_y1 + geo_y2) / 2
        
        # Create building ID
        building_id = f"{tile_id_str.replace('/', '_')}_{i}"
        
        simple_buildings.append({
            "id": building_id,
            "longitude": round(centroid_lon, 8),
            "latitude": round(centroid_lat, 8)
        })
    
    return simple_buildings

def load_saved_tile_results(output_dir):
    """
    Load all saved tile results from JSON files.
    
    Args:
        output_dir: Directory containing saved tile results
        
    Returns:
        List of tile detection results
    """
    tiles_dir = os.path.join(output_dir, "tiles")
    if not os.path.exists(tiles_dir):
        return []
    
    # Load detailed format files (not the simple ones)
    tile_files = glob.glob(os.path.join(tiles_dir, "tile_*.json"))
    # Filter out simple format files
    tile_files = [f for f in tile_files if not f.endswith('_simple.json')]
    
    all_tile_detections = []
    
    for tile_file in tile_files:
        try:
            with open(tile_file, 'r') as f:
                tile_data = json.load(f)
                all_tile_detections.append(tile_data)
        except Exception as e:
            print(f"Error loading tile file {tile_file}: {e}")
    
    print(f"Loaded {len(all_tile_detections)} saved tile results")
    return all_tile_detections

def load_all_simple_tile_results(output_dir):
    """
    Load all simple format tile results and combine them.
    
    Args:
        output_dir: Directory containing saved tile results
        
    Returns:
        List of buildings with id, longitude, latitude
    """
    tiles_dir = os.path.join(output_dir, "tiles")
    if not os.path.exists(tiles_dir):
        return []
    
    # Load simple format files
    simple_files = glob.glob(os.path.join(tiles_dir, "tile_*_simple.json"))
    all_buildings = []
    
    for simple_file in simple_files:
        try:
            with open(simple_file, 'r') as f:
                buildings = json.load(f)
                all_buildings.extend(buildings)
        except Exception as e:
            print(f"Error loading simple tile file {simple_file}: {e}")
    
    print(f"Loaded {len(all_buildings)} buildings from simple format files")
    return all_buildings

def save_incremental_simple_format(output_dir, output_path):
    """
    Save combined simple format results from all tile files.
    
    Args:
        output_dir: Directory containing tile results
        output_path: Path to save the combined simple format JSON
    """
    all_buildings = load_all_simple_tile_results(output_dir)
    
    if all_buildings:
        try:
            with open(output_path, 'w') as f:
                json.dump(all_buildings, f, indent=2)
            print(f"Saved {len(all_buildings)} buildings in simple format to {output_path}")
        except Exception as e:
            print(f"Error saving incremental simple format: {e}")
    else:
        print("No buildings found in simple format tiles")

def merge_all_tiles_to_simple_json(tiles_dir, output_path):
    """
    Merge all tiles JSON files into a single JSON with simple incremental ID format.
    Output format: [{"id": 1, "longitude": 106.123, "latitude": -6.456}, ...]
    
    Args:
        tiles_dir: Directory containing tile JSON files (or parent directory containing tiles/ folder)
        output_path: Path to save the merged simple JSON file
        
    Returns:
        Number of buildings merged
    """
    # Check if tiles_dir is the parent directory containing tiles/ folder
    if os.path.isdir(os.path.join(tiles_dir, "tiles")):
        actual_tiles_dir = os.path.join(tiles_dir, "tiles")
    else:
        actual_tiles_dir = tiles_dir
    
    if not os.path.exists(actual_tiles_dir):
        print(f"Tiles directory not found: {actual_tiles_dir}")
        return 0
    
    print(f"Merging tiles from: {actual_tiles_dir}")
    
    # Load simple format files
    simple_files = glob.glob(os.path.join(actual_tiles_dir, "tile_*_simple.json"))
    
    if not simple_files:
        print("No simple format tile files found")
        return 0
    
    print(f"Found {len(simple_files)} simple format tile files")
    
    all_buildings = []
    
    # Load all buildings from simple tiles
    for simple_file in simple_files:
        try:
            with open(simple_file, 'r') as f:
                buildings = json.load(f)
                all_buildings.extend(buildings)
                print(f"  Loaded {len(buildings)} buildings from {os.path.basename(simple_file)}")
        except Exception as e:
            print(f"Error loading simple tile file {simple_file}: {e}")
    
    # Convert to simple incremental format
    simple_buildings = []
    for i, building in enumerate(all_buildings, start=1):
        simple_buildings.append({
            "id": i,
            "longitude": building.get("longitude", 0.0),
            "latitude": building.get("latitude", 0.0)
        })
    
    # Save to output file
    try:
        with open(output_path, 'w') as f:
            json.dump(simple_buildings, f, indent=2)
        
        print(f"\n✅ Successfully merged {len(simple_buildings)} buildings")
        print(f"📁 Output saved to: {output_path}")
        print("📊 Format: ID starts from 1, increments by 1")
        
        # Show sample of output
        if simple_buildings:
            print("\n📋 Sample data (first 3 buildings):")
            for building in simple_buildings[:3]:
                print(f"  ID: {building['id']}, Lon: {building['longitude']}, Lat: {building['latitude']}")
        
        return len(simple_buildings)
        
    except (IOError, OSError, ValueError) as e:
        print(f"Error saving merged file: {e}")
        return 0

def cleanup_tile_files(output_dir):
    """
    Clean up individual tile JSON files after successful completion.
    
    Args:
        output_dir: Directory containing tile files
    """
    tiles_dir = os.path.join(output_dir, "tiles")
    if os.path.exists(tiles_dir):
        # Clean up both detailed and simple format files
        tile_files = glob.glob(os.path.join(tiles_dir, "tile_*.json"))
        for tile_file in tile_files:
            try:
                os.remove(tile_file)
            except (OSError, IOError):
                pass
        try:
            os.rmdir(tiles_dir)
            print("Cleaned up temporary tile files")
        except (OSError, IOError):
            pass

# --- Helper function to convert tile detections to geographic Shapely polygons ---
def convert_tile_detections_to_shapely_polygons(all_tile_detections):
    """
    Converts raw tile-based detections into a flat list of Shapely polygons 
    with geographic coordinates, associated confidence scores, and original tile ID.
    """
    shapely_polygons_with_attrs = []
    detection_id_counter = 0
    for tile_detection in all_tile_detections:
        # Each tile_detection should have 'tile': "z/x/y", 'bounds', 'boxes', 'confidences'
        tile_id_str = tile_detection.get('tile', 'unknown_tile') # z/x/y string
        bounds = tile_detection['bounds'] # west, south, east, north
        west, south, east, north = bounds
        tile_width_deg = east - west
        tile_height_deg = north - south
        
        img_width, img_height = 256, 256 # Assuming tile image size used for normalization

        for i, bbox_coords in enumerate(tile_detection['boxes']):
            _, _, _, _ = bbox_coords # Normalized 0-1 within tile (from model output)
            
            # Convert normalized to absolute pixel coordinates if they are not already
            # This step depends on how 'boxes' are stored. If they are [0-1], convert.
            # If they are already in pixel coords [0-255], this can be skipped or adjusted.
            # Assuming 'boxes' from process_tile_detections are [x1,y1,x2,y2] in pixel coords of the tile_image
            # For this example, let's assume bbox_coords are [x_min_pixel, y_min_pixel, x_max_pixel, y_max_pixel]
            # If they are normalized (0-1), then:
            # x1_pixel = x1_norm * img_width
            # y1_pixel = y1_norm * img_height
            # x2_pixel = x2_norm * img_width
            # y2_pixel = y2_norm * img_height
            # For now, assuming bbox_coords are already in pixel values [0-255] for a 256x256 tile
            x1_pixel, y1_pixel, x2_pixel, y2_pixel = bbox_coords

            # Convert pixel coordinates to normalized (0-1) within this tile
            x1_norm_tile = x1_pixel / img_width
            y1_norm_tile = y1_pixel / img_height
            x2_norm_tile = x2_pixel / img_width
            y2_norm_tile = y2_pixel / img_height

            # Convert normalized tile coordinates to geographic coordinates
            geo_x1 = west + x1_norm_tile * tile_width_deg
            geo_y1_bottom_up = south + (1 - y2_norm_tile) * tile_height_deg # y is often from top in images
            geo_x2 = west + x2_norm_tile * tile_width_deg
            geo_y2_bottom_up = south + (1 - y1_norm_tile) * tile_height_deg
            
            # Create Shapely box: box(minx, miny, maxx, maxy)
            # Ensure minx < maxx and miny < maxy
            current_shapely_box = box(
                min(geo_x1, geo_x2), 
                min(geo_y1_bottom_up, geo_y2_bottom_up), 
                max(geo_x1, geo_x2), 
                max(geo_y1_bottom_up, geo_y2_bottom_up)
            )
            
            confidence = tile_detection['confidences'][i] if i < len(tile_detection['confidences']) else 0.0
            
            shapely_polygons_with_attrs.append({
                'id': f"det_{detection_id_counter}",
                'polygon': current_shapely_box,
                'confidence': confidence,
                'tile_id': tile_id_str # Menyimpan ID tile asal
            })
            detection_id_counter += 1
            
    return shapely_polygons_with_attrs

def get_long_axis(polygon):
    """
    Menghitung sumbu panjang dari sebuah poligon.
    Mengembalikan vektor arah sumbu panjang (dinormalisasi).
    """
    # Dapatkan minimum bounding rectangle
    try:
        # Gunakan minimum_rotated_rectangle jika tersedia
        mbr = polygon.minimum_rotated_rectangle
    except AttributeError:
        # Fallback ke envelope jika minimum_rotated_rectangle tidak tersedia
        mbr = polygon.envelope
    
    # Dapatkan koordinat dari MBR
    coords = list(mbr.exterior.coords)
    
    # Hitung panjang sisi-sisi MBR
    sides = []
    for i in range(len(coords) - 1):
        dx = coords[i+1][0] - coords[i][0]
        dy = coords[i+1][1] - coords[i][1]
        sides.append(((dx, dy), (dx**2 + dy**2)**0.5))
    
    # Temukan sisi terpanjang
    longest_side = max(sides, key=lambda x: x[1])
    
    # Normalisasi vektor
    dx, dy = longest_side[0]
    length = longest_side[1]
    if length > 0:
        dx /= length
        dy /= length
    
    return (dx, dy)

def calculate_axis_alignment(axis1, axis2):
    """
    Menghitung keselarasan antara dua sumbu.
    Mengembalikan nilai antara 0 dan 1, di mana 1 berarti sejajar sempurna.
    """
    # Hitung dot product
    dot_product = abs(axis1[0] * axis2[0] + axis1[1] * axis2[1])
    
    # Dot product dari dua vektor unit adalah kosinus dari sudut antara mereka
    # Kita menggunakan nilai absolut untuk mengatasi orientasi yang berlawanan
    return dot_product

def parse_tile_id(tile_id_str):
    """Parse a tile ID string (z/x/y) into its components"""
    if tile_id_str == 'UNKNOWN':
        return None, None, None
    parts = tile_id_str.split('/')
    if len(parts) != 3:
        return None, None, None
    try:
        z, x, y = map(int, parts)
        return z, x, y
    except ValueError:
        return None, None, None

def calculate_boundary_proximity(poly, tile_id_str, other_poly, other_tile_id_str):
    """Calculate how close polygons are to their shared tile boundary"""
    z1, x1, y1 = parse_tile_id(tile_id_str)
    z2, x2, y2 = parse_tile_id(other_tile_id_str)
    
    if None in (z1, x1, y1, z2, x2, y2) or z1 != z2:
        return 0  # Invalid tile IDs or different zoom levels
    
    # Determine if tiles are adjacent
    is_adjacent = (abs(x1 - x2) <= 1 and abs(y1 - y2) <= 1)
    if not is_adjacent:
        return 0  # Tiles are not adjacent
    
    # Determine the direction of the shared boundary
    dx, dy = x2 - x1, y2 - y1
    
    # Get centroids
    c1 = poly.centroid
    c2 = other_poly.centroid
    
    # Calculate boundary proximity score based on how aligned the buildings are
    # with the direction of the tile boundary
    if dx != 0 and dy == 0:  # Horizontal boundary
        # Higher score if buildings are vertically aligned
        vertical_alignment = 1 - abs(c1.y - c2.y) / max(poly.bounds[3] - poly.bounds[1], other_poly.bounds[3] - other_poly.bounds[1])
        return vertical_alignment
    elif dx == 0 and dy != 0:  # Vertical boundary
        # Higher score if buildings are horizontally aligned
        horizontal_alignment = 1 - abs(c1.x - c2.x) / max(poly.bounds[2] - poly.bounds[0], other_poly.bounds[2] - other_poly.bounds[0])
        return horizontal_alignment
    else:  # Diagonal boundary (corner touching)
        # For diagonal, use the minimum of horizontal and vertical alignment
        vertical_alignment = 1 - abs(c1.y - c2.y) / max(poly.bounds[3] - poly.bounds[1], other_poly.bounds[3] - other_poly.bounds[1])
        horizontal_alignment = 1 - abs(c1.x - c2.x) / max(poly.bounds[2] - poly.bounds[0], other_poly.bounds[2] - other_poly.bounds[0])
        return min(vertical_alignment, horizontal_alignment)

class UnionFind:
    """
    Union-Find data structure for efficient component merging.
    Enables transitive connections: A→B→C→D all become one component.
    """
    def __init__(self, n):
        # Initialize each element as its own parent
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n
    
    def find(self, x):
        """Find the root of element x with path compression"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        """Union two elements with union by rank"""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False  # Already in same component
        
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        self.components -= 1
        return True
    
    def get_components(self):
        """Get all components as dictionary {root: [members]}"""
        components = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in components:
                components[root] = []
            components[root].append(i)
        return components

def _print_merging_progress(step, current, total, extra_info=""):
    """Print progress for merging operations with consistent formatting"""
    if total > 0:
        progress = (current / total) * 100
        base_msg = f"  {step} progress: {progress:.1f}% ({current}/{total})"
        if extra_info:
            print(f"{base_msg} - {extra_info}")
        else:
            print(base_msg)

def _precompute_polygon_axes(individual_detections):
    """Pre-compute long axes for all polygons with progress tracking"""
    print("Step 1/4: Pre-computing polygon axes...")
    long_axes = []
    
    for i, det in enumerate(individual_detections):
        polygon = det['polygon']
        long_axis = get_long_axis(polygon)
        long_axes.append(long_axis)
        
        # Progress tracking for axis computation
        if (i + 1) % max(1, len(individual_detections) // 10) == 0 or i == len(individual_detections) - 1:
            _print_merging_progress("Axis computation", i + 1, len(individual_detections))
    
    return long_axes

def _check_iou_connection(poly_i, poly_j, iou_thresh):
    """Check IoU overlap connection (Phase 1 - High confidence)
    
    Returns:
        tuple: (is_connected, connection_score, connection_type)
    """
    intersection = poly_i.intersection(poly_j).area
    if intersection > 0:
        union_val = poly_i.area + poly_j.area - intersection
        iou = intersection / union_val if union_val > 0 else 0
        if iou > iou_thresh:
            return True, -iou, 1  # Phase 1: High confidence, negative IoU (higher IoU = better score)
    return False, float('inf'), 1

def _check_boundary_connection(poly_i, poly_j, boundary_score, alignment_factor, 
                             touch_enabled, min_edge_distance_deg):
    """Check boundary-related connections (Phase 2 - Good evidence)
    
    Returns:
        tuple: (is_connected, connection_score, connection_type)
    """
    if boundary_score > 0.7:  # High boundary score
        # For touching polygons near boundaries
        if touch_enabled and poly_i.touches(poly_j):
            connection_score = -boundary_score * alignment_factor
            return True, connection_score, 2  # Phase 2: Boundary related
        
        # For proximal polygons near boundaries
        elif min_edge_distance_deg > 0:
            edge_dist = poly_i.distance(poly_j)
            if 0 < edge_dist < min_edge_distance_deg:
                norm_dist = edge_dist / min_edge_distance_deg
                connection_score = norm_dist - boundary_score
                return True, connection_score, 2  # Phase 2: Boundary related
    
    return False, float('inf'), 2

def _check_proximity_connection(poly_i, poly_j, touch_enabled, min_edge_distance_deg, 
                              alignment_factor, center_distance):
    """Check proximity/touching connections (Phase 3 - Other evidence)
    
    Returns:
        tuple: (is_connected, connection_score, connection_type)
    """
    # Check touches
    if touch_enabled and poly_i.touches(poly_j):
        boundary_i = poly_i.boundary
        boundary_j = poly_j.boundary
        touch_length = boundary_i.intersection(boundary_j).length
        connection_score = -touch_length * alignment_factor * 0.5  # Reduced weight
        return True, connection_score, 3  # Phase 3: Other
    
    # Check edge distance
    elif min_edge_distance_deg > 0:
        edge_dist = poly_i.distance(poly_j)
        if 0 < edge_dist < min_edge_distance_deg:
            epsilon = 1e-10  # Avoid division by zero
            # Higher center distance = worse score
            connection_score = edge_dist * (1 + center_distance) / (alignment_factor + epsilon)
            return True, connection_score, 3  # Phase 3: Other
    
    return False, float('inf'), 3

def _analyze_polygon_pair(i, j, individual_detections, long_axes, 
                        iou_thresh, touch_enabled, min_edge_distance_deg):
    """Analyze geometric relationship between two polygons
    
    Returns:
        tuple: (is_connected, connection_score, connection_type) or None if no connection
    """
    poly_i_obj = individual_detections[i]
    poly_j_obj = individual_detections[j]
    tile_i_id = poly_i_obj.get('tile_id', 'UNKNOWN')
    tile_j_id = poly_j_obj.get('tile_id', 'UNKNOWN')
    
    # Only process if from different tiles
    if tile_i_id == tile_j_id or tile_i_id == 'UNKNOWN' or tile_j_id == 'UNKNOWN':
        return None
    
    # Get polygon objects
    poly_i = poly_i_obj['polygon']
    poly_j = poly_j_obj['polygon']
    
    # Calculate center-to-center distance
    center_i = poly_i.centroid
    center_j = poly_j.centroid
    center_distance = center_i.distance(center_j)
    
    # Calculate boundary proximity score
    boundary_score = calculate_boundary_proximity(
        poly_i, tile_i_id, poly_j, tile_j_id
    )
    
    # Calculate axis alignment with reduced weight
    axis_alignment = calculate_axis_alignment(long_axes[i], long_axes[j])
    alignment_weight = 5.0  # Reduced from 10.0
    alignment_factor = axis_alignment ** alignment_weight
    
    # Check connections in priority order
    
    # 1. Check high-confidence connections (IoU)
    is_connected, score, conn_type = _check_iou_connection(poly_i, poly_j, iou_thresh)
    if is_connected:
        return is_connected, score, conn_type
    
    # 2. Check boundary-related connections
    is_connected, score, conn_type = _check_boundary_connection(
        poly_i, poly_j, boundary_score, alignment_factor,
        touch_enabled, min_edge_distance_deg
    )
    if is_connected:
        return is_connected, score, conn_type
    
    # 3. Check other geometric relationships
    is_connected, score, conn_type = _check_proximity_connection(
        poly_i, poly_j, touch_enabled, min_edge_distance_deg, 
        alignment_factor, center_distance
    )
    if is_connected:
        return is_connected, score, conn_type
    
    return None

def _find_all_connections(individual_detections, long_axes, iou_thresh, 
                        touch_enabled, min_edge_distance_deg):
    """Find all potential connections between detections from different tiles
    
    Returns:
        list: List of (i, j, score, connection_type) tuples
    """
    print("Step 2/4: Finding potential connections between detections...")
    all_connections = []
    
    total_comparisons = (len(individual_detections) * (len(individual_detections) - 1)) // 2
    comparison_count = 0
    
    for i in range(len(individual_detections)):
        for j in range(i+1, len(individual_detections)):
            comparison_count += 1
            
            # Progress tracking for comparisons
            if comparison_count % max(1, total_comparisons // 20) == 0 or comparison_count == total_comparisons:
                _print_merging_progress("Connection analysis", comparison_count, total_comparisons)
            
            # Analyze this polygon pair
            result = _analyze_polygon_pair(
                i, j, individual_detections, long_axes,
                iou_thresh, touch_enabled, min_edge_distance_deg
            )
            
            if result is not None:
                _, connection_score, connection_type = result
                all_connections.append((i, j, connection_score, connection_type))
    
    print(f"Found {len(all_connections)} potential connections")
    return all_connections

def _process_union_find_merging(all_connections, individual_detections, allowed_merge_phases):
    """Process connections using Union-Find algorithm
    
    Returns:
        dict: Components dictionary from Union-Find
    """
    print("Step 3/4: Processing connections and merging...")
    
    # Initialize Union-Find data structure for transitive merging
    n_detections = len(individual_detections)
    uf = UnionFind(n_detections)
    
    # Sort all connections by phase first, then by score within each phase
    all_connections.sort(key=lambda x: (x[3], x[2]))  # Sort by (phase, score)
    
    # Apply union operations for allowed phases only
    processed_connections = 0
    merged_count = 0
    
    for i, j, score, connection_type in all_connections:
        processed_connections += 1
        
        # Progress tracking for connection processing
        if processed_connections % max(1, len(all_connections) // 10) == 0 or processed_connections == len(all_connections):
            _print_merging_progress("Connection processing", processed_connections, len(all_connections), f"Merged: {merged_count}")
        
        if connection_type in allowed_merge_phases:  # Only merge allowed evidence connections
            # Union the detections for high-confidence merging
            if uf.union(i, j):
                merged_count += 1
    
    # Get final components from Union-Find
    components = uf.get_components()
    print(f"Found {len(components)} components to merge")
    return components

def _create_merged_buildings(components, individual_detections):
    """Create final merged building objects from components
    
    Returns:
        list: List of merged building dictionaries
    """
    print("Step 4/4: Creating final merged buildings...")
    
    merged_buildings = []
    component_count = 0
    total_components = len(components)
    
    for comp_root, indices in components.items():
        component_count += 1
        
        # Progress tracking for final building creation
        if component_count % max(1, total_components // 10) == 0 or component_count == total_components:
            _print_merging_progress("Building creation", component_count, total_components)
        
        group_polygons = [individual_detections[idx]['polygon'] for idx in indices]
        group_confidences = [individual_detections[idx]['confidence'] for idx in indices]
        group_ids = [individual_detections[idx]['id'] for idx in indices]
        
        if group_polygons:
            combined_polygon_shape = unary_union(group_polygons)
            merged_envelope = combined_polygon_shape.envelope
            merged_buildings.append({
                'id': f"merged_{len(merged_buildings)}",
                'polygon': merged_envelope, 
                'coordinates': list(merged_envelope.exterior.coords),
                'confidence': max(group_confidences) if group_confidences else 0.0,
                'original_ids': sorted(group_ids),
                'original_count': len(indices)  # Add count of merged detections
            })
    
    return merged_buildings

def merge_overlapping_detections(individual_detections, 
                                 iou_thresh, 
                                 touch_enabled, 
                                 min_edge_distance_deg,
                                 allowed_merge_phases=[1, 2]):
    """
    Merges overlapping and proximal detections using a selective multi-phase approach
    with Union-Find transitive merging and evidence-based filtering.
    
    Phases:
    - Phase 1: IoU overlap (STRONG evidence) - Always recommended  
    - Phase 2: Boundary-related + alignment (GOOD evidence) - Usually safe
    - Phase 3: Proximity/touching only (WEAK evidence) - May cause false merging
    
    Args:
        individual_detections: List of detection dictionaries with 'polygon', 'confidence', etc.
        iou_thresh: IoU threshold for Phase 1 merging
        touch_enabled: Whether to enable touching detection in phases 2 & 3
        min_edge_distance_deg: Max edge distance for proximity merging  
        allowed_merge_phases: List of phases to enable [1,2,3]. Default [1,2] prevents false merging.
        
    Returns:
        List of merged building dictionaries with transitive connections resolved.
    """
    if not individual_detections:
        return []

    print(f"Starting merging process for {len(individual_detections)} individual detections...")
    
    # Step 1: Pre-compute polygon axes for alignment analysis
    long_axes = _precompute_polygon_axes(individual_detections)
    
    # Step 2: Find all potential connections between detections
    all_connections = _find_all_connections(
        individual_detections, long_axes, iou_thresh, 
        touch_enabled, min_edge_distance_deg
    )
    
    # Step 3: Process connections using Union-Find algorithm
    components = _process_union_find_merging(
        all_connections, individual_detections, allowed_merge_phases
    )
    
    # Step 4: Create final merged building objects
    merged_buildings = _create_merged_buildings(components, individual_detections)
    
    print(f"Merging completed: {len(individual_detections)} individual detections → {len(merged_buildings)} merged buildings")
    return merged_buildings

def process_tile_batch(tile_batch, model, conf, model_lock):
    """Process a batch of tiles and return their detection results"""
    batch_results = []
    
    for tile in tile_batch:
        try:
            # Get tile image (in memory)
            tile_image = get_tile_image(tile)
            
            # Create a temporary file for detection
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
                # Save the image to the temporary file
                tile_image.save(temp_path, format='PNG')
            
            try:
                # Detect buildings using the temporary file path
                # Ensure model access is serialized
                with model_lock:
                    results, _ = detect_buildings(model, temp_path, conf=conf)
                
                # Process detection results
                boxes, confidences, class_ids = process_tile_detections(results)
                
                # Add to results
                tile_bounds = get_tile_bounds(tile)
                tile_detections = {
                    'tile': f"{tile.z}/{tile.x}/{tile.y}",
                    'bounds': tile_bounds,
                    'detections': len(boxes),
                    'boxes': boxes.tolist() if boxes.size > 0 else [],
                    'confidences': confidences.tolist() if len(confidences) > 0 else [],
                    'class_ids': class_ids.tolist() if len(class_ids) > 0 else [],
                    'image': tile_image  # Store the image in memory
                }
                batch_results.append(tile_detections)
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            
        except Exception as e:
            print(f"Error processing tile {tile}: {e}")
    
    return batch_results

def create_batches(items, batch_size):
    """Split a list of items into batches of the specified size"""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

def _initialize_detection_session(output_dir):
    """Initialize detection session with timing and directory setup"""
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    return start_time

def _load_and_prepare_data(geojson_path, zoom):
    """Load GeoJSON data and prepare tiles for processing"""
    geojson_data = load_geojson(geojson_path)
    polygon = extract_polygon(geojson_data)
    tiles = get_tiles_for_polygon(polygon, zoom=zoom)
    print(f"Found {len(tiles)} tiles that intersect with the polygon")
    return tiles

def _handle_resume_logic(resume_from_saved, output_dir, tiles):
    """Handle resume from saved tiles logic and return tiles to process"""
    all_detections_raw_per_tile = []
    tiles_to_process = tiles
    
    if resume_from_saved:
        saved_tile_results = load_saved_tile_results(output_dir)
        if saved_tile_results:
            # Convert saved results back to the expected format (add image field as None)
            for saved_tile in saved_tile_results:
                saved_tile['image'] = None  # Images are not saved, set to None
            all_detections_raw_per_tile.extend(saved_tile_results)
            
            # Find which tiles still need to be processed
            processed_tile_ids = {tile_data['tile'] for tile_data in saved_tile_results}
            tiles_to_process = [tile for tile in tiles 
                              if f"{tile.z}/{tile.x}/{tile.y}" not in processed_tile_ids]
            
            print(f"Resuming from saved results: {len(saved_tile_results)} tiles already processed")
            print(f"Remaining tiles to process: {len(tiles_to_process)}")
    
    return all_detections_raw_per_tile, tiles_to_process

def _execute_tile_processing(tiles_to_process, batch_size, model, conf, output_dir, all_detections_raw_per_tile):
    """Execute parallel tile processing and return updated detections"""
    if not tiles_to_process:
        print("All tiles have been processed previously. Proceeding to merging...")
        return all_detections_raw_per_tile
    
    # Create batches of remaining tiles
    tile_batches = create_batches(tiles_to_process, batch_size)
    print(f"Created {len(tile_batches)} batches with batch size {batch_size}")
    
    # Fixed optimal worker count based on previous benchmarks
    num_workers = 2
    
    # Create a lock for model access
    model_lock = threading.Lock()
    
    # Create a partial function with fixed arguments, including the lock
    process_batch = partial(process_tile_batch, model=model, conf=conf, model_lock=model_lock)
    
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all batches and get a list of futures
        future_to_batch = {executor.submit(process_batch, batch): i for i, batch in enumerate(tile_batches)}
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_batch), 
                          total=len(tile_batches), 
                          desc="Processing tile batches"):
            batch_idx = future_to_batch[future]
            try:
                batch_detections = future.result()
                if batch_detections:
                    # Save each tile immediately after processing
                    for tile_detection in batch_detections:
                        save_tile_results(tile_detection, output_dir, tile_detection['tile'])
                    
                    all_detections_raw_per_tile.extend(batch_detections) # Store raw per-tile results
            except (RuntimeError, ValueError, IOError) as exc:
                print(f"Batch {batch_idx} generated an exception: {exc}")
                # Continue processing other batches even if one fails
    
    return all_detections_raw_per_tile

def _process_merging_phase(all_detections_raw_per_tile, merge_iou_threshold, merge_touch_enabled, merge_min_edge_distance_deg):
    """Process merging phase and return merged results"""
    total_raw_detections = sum(len(t['boxes']) for t in all_detections_raw_per_tile)
    print(f"\n{'='*60}")
    print(f"MERGING PHASE: Converting {total_raw_detections} raw detections")
    print(f"{'='*60}")
    
    print(f"Converting {total_raw_detections} raw detections to Shapely objects for merging...")
    individual_shapely_detections = convert_tile_detections_to_shapely_polygons(all_detections_raw_per_tile)
    
    print(f"\n{'='*60}")
    print(f"DETAILED MERGING: Processing {len(individual_shapely_detections)} individual detections")
    print("Merging parameters:")
    print(f"  - IoU threshold: {merge_iou_threshold}")
    print(f"  - Touch enabled: {merge_touch_enabled}")
    print(f"  - Min edge distance: {merge_min_edge_distance_deg} degrees")
    print("  - Allowed phases: [1, 2] (IoU overlap + Boundary-related)")
    print(f"{'='*60}")
    
    merged_buildings_list = merge_overlapping_detections(
        individual_shapely_detections,
        merge_iou_threshold,
        merge_touch_enabled,
        merge_min_edge_distance_deg,
        allowed_merge_phases=[1, 2]  # Only merge high-confidence phases
    )
    
    # Calculate merging statistics
    reduction_count = len(individual_shapely_detections) - len(merged_buildings_list)
    reduction_percentage = (reduction_count / len(individual_shapely_detections)) * 100 if individual_shapely_detections else 0
    
    print(f"\n{'='*60}")
    print("MERGING RESULTS:")
    print(f"  Input individual detections: {len(individual_shapely_detections)}")
    print(f"  Final merged buildings: {len(merged_buildings_list)}")
    print(f"  Reductions achieved: {reduction_count} (-{reduction_percentage:.1f}%)")
    print(f"{'='*60}\n")

    # Format merged results
    final_detections_for_json = []
    final_merged_shapely_objects = []
    
    for mb in merged_buildings_list:
        final_detections_for_json.append({
            'id': mb['id'],
            'coordinates': mb['coordinates'], # Already in geojson-friendly format
            'confidence': mb['confidence'],
            'original_count': mb['original_count']
        })
        final_merged_shapely_objects.append(mb) # Contains the 'polygon' Shapely object
    
    return final_detections_for_json, final_merged_shapely_objects, len(merged_buildings_list)

def _process_no_merging_phase(all_detections_raw_per_tile):
    """Process individual detections without merging"""
    print("Merging disabled. Processing individual detections for output.")
    individual_shapely_detections = convert_tile_detections_to_shapely_polygons(all_detections_raw_per_tile)
    
    final_detections_for_json = []
    final_merged_shapely_objects = []
    
    for det in individual_shapely_detections:
        final_detections_for_json.append({
            'id': det['id'],
            'coordinates': list(det['polygon'].exterior.coords),
            'confidence': det['confidence']
        })
        final_merged_shapely_objects.append(det) # Contains the 'polygon' Shapely object
    
    return final_detections_for_json, final_merged_shapely_objects, len(individual_shapely_detections)

def _create_results_payload(total_buildings_final, tiles, zoom, conf, enable_merging, merge_iou_threshold, merge_touch_enabled, merge_min_edge_distance_deg, final_detections_for_json, start_time):
    """Create final JSON results payload"""
    end_time = time.time()
    execution_time = end_time - start_time
    
    json_results_payload = {
        'total_buildings': total_buildings_final,
        'total_tiles': len(tiles), # This remains the number of processed tiles
        'zoom': zoom,
        'confidence_threshold': conf, # Original detection confidence
        'merging_enabled': enable_merging,
        'merge_iou_threshold': merge_iou_threshold if enable_merging else None,
        'merge_touch_enabled': merge_touch_enabled if enable_merging else None,
        'merge_min_edge_distance_deg': merge_min_edge_distance_deg if enable_merging else None,
        'detections': final_detections_for_json, # This is now a list of merged buildings
        'execution_time': execution_time
    }
    
    return json_results_payload

def _generate_visualization(geojson_path, output_dir, total_buildings_final, tiles, zoom, conf, final_merged_shapely_objects, all_detections_raw_per_tile):
    """Generate and save visualization"""
    visualization_path = os.path.join(output_dir, "polygon_visualization.png")
    
    visualization_results_data = {
        'total_buildings': total_buildings_final,
        'total_tiles': len(tiles),
        'zoom': zoom,
        'confidence_threshold': conf, # Original detection confidence for context
        'merged_detections_shapely': final_merged_shapely_objects, 
        'raw_tile_detections_for_background': all_detections_raw_per_tile 
    }
    
    print("Visualizing merged detections...")
    visualize_polygon_detections(
        geojson_path, 
        visualization_results_data, 
        visualization_path
    )

def _save_output_files(output_dir, json_results_payload, geojson_path):
    """Save all output files in different formats"""
    # Save main results
    results_path = os.path.join(output_dir, "detection_results.json")
    with open(results_path, 'w') as f:
        json.dump(json_results_payload, f, indent=2)
    
    # Save buildings to JSON
    buildings_json_path = os.path.join(output_dir, "buildings.json")
    save_buildings_to_json(json_results_payload, buildings_json_path)
    
    # Save buildings in simple format (only buildings inside polygon)
    buildings_simple_path = os.path.join(output_dir, "buildings_simple.json")
    save_buildings_simple_format(json_results_payload, geojson_path, buildings_simple_path)
    
    # Save incremental simple format results (combined from all tiles)
    buildings_incremental_simple_path = os.path.join(output_dir, "buildings_incremental_simple.json")
    save_incremental_simple_format(output_dir, buildings_incremental_simple_path)
    
    return results_path

def _finalize_session(output_dir, results_path, total_buildings_final, execution_time):
    """Finalize session with cleanup and status reporting"""
    cleanup_tile_files(output_dir)
    
    print(f"Processing completed in {execution_time:.2f} seconds")
    print(f"Detection results saved to {results_path}")
    print(f"Total buildings detected: {total_buildings_final}")

def detect_buildings_in_polygon(model, geojson_path, output_dir="polygon_detection_results", zoom=18, conf=0.25, batch_size=5,
                                enable_merging=True,
                                merge_iou_threshold=1.1, 
                                merge_touch_enabled=True, 
                                merge_min_edge_distance_deg=0.00001, # Diubah dari merge_centroid_proximity_deg
                                resume_from_saved=True  # New parameter to enable resuming from saved tiles
                                ):
    """
    Detect buildings within a polygon defined in a GeoJSON file using optimized batch processing.
    Optionally merges fragmented detections.
    
    Args:
        model: Loaded YOLOv8 model
        geojson_path: Path to the GeoJSON file
        output_dir: Directory to save detection results
        zoom: Zoom level for tiles
        conf: Confidence threshold for individual detections
        batch_size: Number of tiles per batch
        enable_merging: Whether to perform post-processing to merge fragmented detections.
        merge_iou_threshold: IoU threshold for considering detections part of the same group for merging.
        merge_touch_enabled: Whether touching polygons are considered for merging.
        merge_min_edge_distance_deg: Max edge distance (degrees) for merging non-touching, non-overlapping detections.
        resume_from_saved: Whether to resume from previously saved tile results.
        
    Returns:
        Dictionary with detection results and execution time
    """
    # Initialize session
    start_time = _initialize_detection_session(output_dir)
    
    # Load and prepare data
    tiles = _load_and_prepare_data(geojson_path, zoom)
    
    # Handle resume logic
    all_detections_raw_per_tile, tiles_to_process = _handle_resume_logic(resume_from_saved, output_dir, tiles)
    
    # Execute tile processing
    all_detections_raw_per_tile = _execute_tile_processing(tiles_to_process, batch_size, model, conf, output_dir, all_detections_raw_per_tile)
    
    # Process results based on merging setting
    if enable_merging:
        final_detections_for_json, final_merged_shapely_objects, total_buildings_final = _process_merging_phase(
            all_detections_raw_per_tile, merge_iou_threshold, merge_touch_enabled, merge_min_edge_distance_deg)
    else:
        final_detections_for_json, final_merged_shapely_objects, total_buildings_final = _process_no_merging_phase(
            all_detections_raw_per_tile)
    
    # Create results payload
    json_results_payload = _create_results_payload(
        total_buildings_final, tiles, zoom, conf, enable_merging, 
        merge_iou_threshold, merge_touch_enabled, merge_min_edge_distance_deg, 
        final_detections_for_json, start_time)
    
    # Generate visualization
    _generate_visualization(geojson_path, output_dir, total_buildings_final, tiles, zoom, conf, final_merged_shapely_objects, all_detections_raw_per_tile)
    
    # Save output files
    results_path = _save_output_files(output_dir, json_results_payload, geojson_path)
    
    # Finalize session
    _finalize_session(output_dir, results_path, total_buildings_final, json_results_payload['execution_time'])
    
    return json_results_payload

if __name__ == "__main__":
    # Check if shapely and mercantile are installed
    try:
        import shapely
        import mercantile
    except ImportError:
        print("This script requires shapely and mercantile packages.")
        print("Please install them with: pip install shapely mercantile")
        sys.exit(1)
    
    # Path to the model
    model_path = "../best.pt"
    
    # Create example GeoJSON if needed
    if len(sys.argv) > 1:
        geojson_path = sys.argv[1]
    else:
        print("No GeoJSON file provided, creating an example...")
        geojson_path = create_example_geojson()
    
    # Output directory
    output_dir = "polygon_detection_results"
    
    # Set batch size (can be specified as command line argument)
    batch_size = 5
    if len(sys.argv) > 2:
        try:
            batch_size = int(sys.argv[2])
        except (ValueError, TypeError):
            pass # Keep default if conversion fails
    
    # Load the YOLOv8 model
    model = load_model(model_path)
    
    # Detect buildings in the polygon with optimized batch processing
    # Enhanced merging parameters for multi-tile building detection:
    results = detect_buildings_in_polygon(
        model, geojson_path, output_dir, zoom=18, conf=0.25, batch_size=batch_size,
        enable_merging=True,  # Enable Union-Find transitive merging
        merge_iou_threshold=0.1,       # Lower threshold for overlapping parts
        merge_touch_enabled=True,      # Enable touching detection
        merge_min_edge_distance_deg=0.000001,  # ~55 meters max distance for proximity merging
        resume_from_saved=False  # Enable resuming from saved tile results
    )
    
    print("\nDetection Summary:")
    print(f"Total buildings detected: {results['total_buildings']}")
    print(f"Total tiles processed: {results['total_tiles']}")
    print(f"Execution time: {results['execution_time']:.2f} seconds")
    print(f"Results saved to {output_dir}/detection_results.json")
    print(f"Visualization saved to {output_dir}/polygon_visualization.png")
    print(f"Building data saved to {output_dir}/buildings.json")
    print(f"Simple building data saved to {output_dir}/buildings_simple.json")