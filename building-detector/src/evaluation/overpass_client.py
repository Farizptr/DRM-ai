"""
Overpass API Client for fetching OpenStreetMap building data.

This module provides a client for querying building data from OpenStreetMap
using the Overpass API as ground truth for building detection evaluation.
"""

import json
import requests
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from shapely.geometry import Polygon, Point, shape
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OverpassAPIError(Exception):
    """Custom exception for Overpass API related errors."""
    pass


class OverpassClient:
    """
    Client for fetching building data from OpenStreetMap via Overpass API.
    
    This client is designed to retrieve ground truth building data for 
    accuracy evaluation of building detection models.
    """
    
    def __init__(self, endpoint: str = "https://overpass-api.de/api/interpreter", timeout: int = 30):
        """
        Initialize the Overpass client.
        
        Args:
            endpoint: Overpass API endpoint URL
            timeout: Request timeout in seconds
        """
        self.endpoint = endpoint
        self.timeout = timeout
        
    def get_buildings_in_polygon(self, geojson_polygon: Dict) -> Dict:
        """
        Fetch all buildings within a given polygon from OpenStreetMap.
        Only includes buildings whose centroid is within the study area.
        
        Args:
            geojson_polygon: GeoJSON polygon feature or FeatureCollection
            
        Returns:
            GeoJSON FeatureCollection containing building data
            
        Raises:
            OverpassAPIError: If API request fails or data is invalid
        """
        try:
            # Extract polygon coordinates for Overpass query
            polygon_coords = self._extract_polygon_coordinates(geojson_polygon)
            
            # Create Shapely polygon for centroid filtering
            study_area_polygon = self._create_study_area_polygon(geojson_polygon)
            
            # Build Overpass query
            query = self._build_overpass_query(polygon_coords)
            
            # Execute query
            logger.info("Fetching building data from OpenStreetMap...")
            osm_data = self._execute_query(query)
            
            # Parse response to GeoJSON
            geojson_result = self._parse_osm_to_geojson(osm_data)
            
            # Filter buildings by centroid within study area
            original_count = len(geojson_result["features"])
            filtered_features = self._filter_buildings_by_centroid(
                geojson_result["features"], 
                study_area_polygon
            )
            filtered_count = len(filtered_features)
            
            # Update features with filtered data
            geojson_result["features"] = filtered_features
            
            # Add metadata
            geojson_result["metadata"] = {
                "source": "OpenStreetMap",
                "query_time": datetime.now().isoformat(),
                "original_buildings": original_count,
                "filtered_buildings": filtered_count,
                "buildings_removed": original_count - filtered_count,
                "filtering_method": "centroid_within_polygon",
                "api_endpoint": self.endpoint
            }
            
            logger.info(f"Original buildings found: {original_count}")
            logger.info(f"Buildings after centroid filtering: {filtered_count}")
            logger.info(f"Buildings removed (edge cases): {original_count - filtered_count}")
            
            return geojson_result
            
        except Exception as e:
            logger.error(f"Error fetching buildings from OSM: {str(e)}")
            raise OverpassAPIError(f"Failed to fetch building data: {str(e)}")
    
    def _extract_polygon_coordinates(self, geojson_polygon: Dict) -> List[Tuple[float, float]]:
        """
        Extract polygon coordinates from GeoJSON.
        
        Args:
            geojson_polygon: GeoJSON polygon data
            
        Returns:
            List of (lat, lon) coordinate tuples for Overpass query
        """
        try:
            # Handle FeatureCollection
            if geojson_polygon.get("type") == "FeatureCollection":
                if not geojson_polygon.get("features"):
                    raise ValueError("Empty FeatureCollection")
                geometry = geojson_polygon["features"][0]["geometry"]
            
            # Handle Feature
            elif geojson_polygon.get("type") == "Feature":
                geometry = geojson_polygon["geometry"]
            
            # Handle direct Geometry
            else:
                geometry = geojson_polygon
            
            if geometry.get("type") != "Polygon":
                raise ValueError("Geometry must be a Polygon")
            
            # Extract coordinates (first ring only, ignore holes)
            coords = geometry["coordinates"][0]
            
            # Convert to (lat, lon) format for Overpass
            # Note: GeoJSON uses [lon, lat] but Overpass uses [lat, lon]
            polygon_coords = [(lat, lon) for lon, lat in coords]
            
            logger.debug(f"Study area coordinates (lat, lon): {polygon_coords[:3]}...")
            
            return polygon_coords
            
        except Exception as e:
            raise ValueError(f"Invalid GeoJSON polygon format: {str(e)}")
    
    def _create_study_area_polygon(self, geojson_polygon: Dict) -> Polygon:
        """
        Create Shapely polygon from GeoJSON for centroid filtering.
        
        Args:
            geojson_polygon: GeoJSON polygon data
            
        Returns:
            Shapely Polygon object
        """
        try:
            # Handle FeatureCollection
            if geojson_polygon.get("type") == "FeatureCollection":
                if not geojson_polygon.get("features"):
                    raise ValueError("Empty FeatureCollection")
                geometry = geojson_polygon["features"][0]["geometry"]
            
            # Handle Feature
            elif geojson_polygon.get("type") == "Feature":
                geometry = geojson_polygon["geometry"]
            
            # Handle direct Geometry
            else:
                geometry = geojson_polygon
            
            # Create Shapely polygon directly from GeoJSON geometry
            # Note: Shapely uses (lon, lat) format, same as GeoJSON
            polygon = shape(geometry)
            
            logger.debug(f"Study area polygon bounds: {polygon.bounds}")
            
            return polygon
            
        except Exception as e:
            raise ValueError(f"Error creating study area polygon: {str(e)}")
    
    def _build_overpass_query(self, polygon_coords: List[Tuple[float, float]]) -> str:
        """
        Build Overpass QL query for buildings within polygon.
        
        Args:
            polygon_coords: List of (lat, lon) coordinate tuples
            
        Returns:
            Overpass QL query string
        """
        # Format coordinates for poly filter
        poly_coords = " ".join([f"{lat} {lon}" for lat, lon in polygon_coords])
        
        # Build query
        query = f"""
        [out:json][timeout:{self.timeout}];
        (
          way["building"](poly:"{poly_coords}");
          relation["building"](poly:"{poly_coords}");
        );
        out geom;
        """
        
        return query
    
    def _execute_query(self, query: str) -> Dict:
        """
        Execute Overpass query and return response.
        
        Args:
            query: Overpass QL query string
            
        Returns:
            Parsed JSON response from Overpass API
        """
        try:
            response = requests.post(
                self.endpoint,
                data={"data": query},
                timeout=self.timeout,
                headers={"User-Agent": "BuildingDetector/1.0 (Accuracy Evaluation)"}
            )
            
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.Timeout:
            raise OverpassAPIError(f"Request timeout after {self.timeout} seconds")
        except requests.exceptions.RequestException as e:
            raise OverpassAPIError(f"Request failed: {str(e)}")
        except json.JSONDecodeError:
            raise OverpassAPIError("Invalid JSON response from Overpass API")
    
    def _parse_osm_to_geojson(self, osm_data: Dict) -> Dict:
        """
        Parse OSM response to GeoJSON format.
        
        Args:
            osm_data: Raw OSM data from Overpass API
            
        Returns:
            GeoJSON FeatureCollection
        """
        features = []
        
        for element in osm_data.get("elements", []):
            try:
                if element["type"] == "way" and "geometry" in element:
                    # Convert OSM way to GeoJSON polygon
                    coords = [[node["lon"], node["lat"]] for node in element["geometry"]]
                    
                    # Ensure polygon is closed
                    if coords[0] != coords[-1]:
                        coords.append(coords[0])
                    
                    feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [coords]
                        },
                        "properties": {
                            "osm_id": element["id"],
                            "osm_type": element["type"],
                            "building": element.get("tags", {}).get("building", "yes"),
                            **element.get("tags", {})
                        }
                    }
                    features.append(feature)
                    
            except (KeyError, IndexError) as e:
                logger.warning(f"Skipping malformed OSM element {element.get('id', 'unknown')}: {str(e)}")
                continue
        
        return {
            "type": "FeatureCollection",
            "features": features
        }
    
    def _filter_buildings_by_centroid(self, buildings: List[Dict], study_area: Polygon) -> List[Dict]:
        """
        Filter buildings to only include those whose centroid is within the study area polygon.
        
        Args:
            buildings: List of GeoJSON building features
            study_area: Shapely Polygon representing the study area
            
        Returns:
            Filtered list of building features
        """
        filtered_buildings = []
        
        logger.debug(f"Study area bounds: {study_area.bounds}")
        logger.debug(f"Study area contains {len(buildings)} buildings to filter")
        
        for i, building in enumerate(buildings):
            try:
                # Create Shapely geometry from GeoJSON
                building_geom = shape(building["geometry"])
                
                # Calculate centroid
                centroid = building_geom.centroid
                
                # Log first few buildings for debugging
                if i < 3:
                    logger.debug(f"Building {i+1} centroid: ({centroid.x:.6f}, {centroid.y:.6f})")
                
                # Check if centroid is within study area
                if study_area.contains(centroid):
                    # Add centroid coordinates to properties for reference
                    building["properties"]["centroid_lon"] = centroid.x
                    building["properties"]["centroid_lat"] = centroid.y
                    building["properties"]["centroid_within_area"] = True
                    filtered_buildings.append(building)
                    logger.debug(f"Building {building['properties'].get('osm_id')} - centroid WITHIN study area")
                else:
                    # Log removed building for debugging
                    if i < 5:  # Log first few removed buildings
                        logger.debug(f"Building {building['properties'].get('osm_id')} - centroid OUTSIDE study area")
                    
            except Exception as e:
                logger.warning(f"Error processing building {building.get('properties', {}).get('osm_id', 'unknown')}: {str(e)}")
                continue
        
        return filtered_buildings


def test_overpass_client():
    """
    Test function for OverpassClient.
    """
    # Sample polygon (small area for testing)
    test_polygon = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [106.7403338391689, -6.217401241157376],
                [106.7403338391689, -6.2191121104196725],
                [106.74211696367672, -6.2191121104196725],
                [106.74211696367672, -6.217401241157376],
                [106.7403338391689, -6.217401241157376]
            ]]
        },
        "properties": {}
    }
    
    client = OverpassClient()
    
    try:
        result = client.get_buildings_in_polygon(test_polygon)
        print(f"✅ Success! Found {result['metadata']['total_buildings']} buildings")
        return result
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None


if __name__ == "__main__":
    # Run test
    test_overpass_client()
