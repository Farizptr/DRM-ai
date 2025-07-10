import json
from shapely.geometry import Point, Polygon, shape
from .geojson_utils import load_geojson, extract_polygon

def save_buildings_to_json(results_data, output_path="buildings.json"):
    """
    Save merged building data to a JSON file.
    Each building will have its ID, exterior coordinates, confidence, and original detection count.
    
    Args:
        results_data: Detection results payload, where results_data['detections'] 
                      is a list of merged building dictionaries.
        output_path: Path to save the JSON file
        
    Returns:
        Path to the saved JSON file
    """
    # The 'detections' key in results_data now holds the list of merged buildings
    # Each item already has 'id', 'coordinates', 'confidence', and 'original_count' (if merging enabled)
    buildings_to_save = results_data.get('detections', [])

    # If merging was disabled, the structure might be slightly different
    # We ensure a consistent output format here, prioritizing fields from merged structure
    # or adapting if it's from non-merged individual detections.

    formatted_buildings = []
    for i, bldg_data in enumerate(buildings_to_save):
        formatted_building = {
            "building_id": bldg_data.get('id', f"bldg_{i}"),
            "geometry_type": "Polygon",
            "coordinates": [bldg_data.get('coordinates', [])], # GeoJSON format for Polygon coordinates
            "confidence": bldg_data.get('confidence', 0.0),
        }
        if 'original_count' in bldg_data: # Specific to merged buildings
            formatted_building['original_detection_count'] = bldg_data['original_count']
        
        formatted_buildings.append(formatted_building)

    # Prepare the final JSON structure (could be a simple list or a GeoJSON-like structure)
    output_json = {
        "type": "FeatureCollection",
        "total_buildings": results_data.get('total_buildings', len(formatted_buildings)),
        "merging_enabled": results_data.get('merging_enabled', False),
        "features": [
            {
                "type": "Feature",
                "id": bldg["building_id"],
                "properties": {
                    "confidence": bldg["confidence"],
                    # Add other properties as needed, e.g., original_detection_count
                    **({ "original_detection_count": bldg["original_detection_count"] } if "original_detection_count" in bldg else {})
                },
                "geometry": {
                    "type": bldg["geometry_type"],
                    "coordinates": bldg["coordinates"]
                }
            } for bldg in formatted_buildings
        ]
    }
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(output_json, f, indent=2)
    
    print(f"Merged building data saved to {output_path}")
    print(f"Total buildings saved: {output_json['total_buildings']}")
    
    return output_path

def save_buildings_simple_format(results_data, geojson_path, output_path="buildings_simple.json"):
    """
    Save buildings in simple format with only buildings inside the GeoJSON polygon.
    Output format: [{id, longitude, latitude}, ...]
    
    Args:
        results_data: Detection results payload
        geojson_path: Path to the GeoJSON file containing the target polygon
        output_path: Path to save the simple JSON file
        
    Returns:
        Path to the saved JSON file
    """
    # Load the GeoJSON polygon
    geojson_data = load_geojson(geojson_path)
    target_polygon = extract_polygon(geojson_data)
    
    # Get building detections
    buildings_to_check = results_data.get('detections', [])
    
    buildings_inside = []
    sequential_id = 1  # Start with ID 1 for sequential numbering
    
    for bldg_data in buildings_to_check:
        # Get building coordinates and create Shapely polygon
        coordinates = bldg_data.get('coordinates', [])
        if not coordinates:
            continue
            
        try:
            # Create Shapely polygon from coordinates
            building_polygon = Polygon(coordinates)
            
            # Get centroid
            centroid = building_polygon.centroid
            centroid_point = Point(centroid.x, centroid.y)
            
            # Check if centroid is inside target polygon
            if target_polygon.contains(centroid_point):
                # Use sequential ID instead of original ID
                buildings_inside.append({
                    "id": str(sequential_id),  # Sequential ID: 1, 2, 3, ...
                    "longitude": round(centroid.x, 8),
                    "latitude": round(centroid.y, 8)
                })
                sequential_id += 1  # Increment for next building
                
        except Exception as e:
            print(f"Error processing building {bldg_data.get('id', 'unknown')}: {e}")
            continue
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(buildings_inside, f, indent=2)
    
    print(f"Buildings inside polygon saved to {output_path}")
    print(f"Total buildings inside polygon: {len(buildings_inside)}")
    
    return output_path 