import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon, shape, box
from shapely.ops import unary_union

from ..utils.geojson_utils import load_geojson, extract_polygon
from ..core.tile_utils import create_stitched_image

# Helper to convert Shapely Polygon to Matplotlib Patch
def shapely_polygon_to_mpl_patch(shapely_polygon, **kwargs):
    """Convert a Shapely Polygon to a Matplotlib Polygon patch."""
    # For Polygons with no interior rings (holes)
    if shapely_polygon.interiors:
        # More complex polygons with holes require Path and PathPatch
        # This is a simplified example assuming simple polygons (envelopes from merge)
        print("Warning: Polygon has interiors (holes), visualization might be simplified.")
        # Fallback to exterior for simplicity, or implement full Path conversion
        path_data = [(patches.Path.MOVETO, list(shapely_polygon.exterior.coords)[0])]
        path_data.extend([(patches.Path.LINETO, pt) for pt in list(shapely_polygon.exterior.coords)[1:]])
        path_data.append((patches.Path.CLOSEPOLY, list(shapely_polygon.exterior.coords)[0]))
        
        # Handle interiors (holes)
        for interior in shapely_polygon.interiors:
            path_data.append((patches.Path.MOVETO, list(interior.coords)[0]))
            path_data.extend([(patches.Path.LINETO, pt) for pt in list(interior.coords)[1:]])
            path_data.append((patches.Path.CLOSEPOLY, list(interior.coords)[0]))
            
        path = patches.Path(np.array([v[1] for v in path_data]), np.array([c[0] for c in path_data]))
        return patches.PathPatch(path, **kwargs)
    else:
        # Simple polygon without holes
        return patches.Polygon(np.array(list(shapely_polygon.exterior.coords)), closed=True, **kwargs)

def _setup_plot_area(polygon_area):
    """Set up the initial plot area and return figure, axis, and bounds"""
    _, ax = plt.subplots(1, figsize=(15, 15))
    minx, miny, maxx, maxy = polygon_area.bounds
    return ax, (minx, miny, maxx, maxy)

def _create_background_image(ax, raw_tile_data, bounds):
    """Create stitched image background if available, otherwise set default bounds"""
    minx, miny, maxx, maxy = bounds
    
    if raw_tile_data and any(d.get('image') for d in raw_tile_data):
        try:
            stitched_image, transform_params = create_stitched_image(raw_tile_data)
            ax.imshow(stitched_image, extent=[
                transform_params['min_west'], 
                transform_params['min_west'] + transform_params['width_deg'],
                transform_params['max_north'] - transform_params['height_deg'],
                transform_params['max_north']
            ])
            ax.set_xlim(transform_params['min_west'], transform_params['min_west'] + transform_params['width_deg'])
            ax.set_ylim(transform_params['max_north'] - transform_params['height_deg'], transform_params['max_north'])
        except Exception as e:
            print(f"Warning: Failed to create stitched image for background: {e}")
            print("Falling back to GeoJSON polygon bounds.")
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)
    else:
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

def _plot_single_polygon(ax, geom, color):
    """Plot a single polygon geometry"""
    if geom.geom_type == 'Polygon':
        x_coords, y_coords = geom.exterior.xy
        ax.plot(x_coords, y_coords, color=color, linewidth=2, alpha=0.7)
    elif geom.geom_type == 'MultiPolygon':
        for poly_part in geom.geoms:
            x_coords, y_coords = poly_part.exterior.xy
            ax.plot(x_coords, y_coords, color=color, linewidth=2, alpha=0.7)

def _add_feature_label(ax, feature, geom):
    """Add label text for a feature if it has a name property"""
    if 'properties' in feature and 'name' in feature['properties']:
        ax.text(geom.centroid.x, geom.centroid.y, feature['properties']['name'], 
                fontsize=10, ha='center', va='center', 
                bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0.1})

def _plot_geojson_polygons(ax, geojson_data_loaded, polygon_area):
    """Plot the GeoJSON polygon areas"""
    if geojson_data_loaded['type'] == 'FeatureCollection':
        for i, feature in enumerate(geojson_data_loaded['features']):
            geom = shape(feature['geometry'])
            if geom.geom_type in ['Polygon', 'MultiPolygon']:
                color = plt.cm.tab20(i % 20)
                _plot_single_polygon(ax, geom, color)
                _add_feature_label(ax, feature, geom)
    else:  # Single Polygon
        x_coords, y_coords = polygon_area.exterior.xy
        ax.plot(x_coords, y_coords, color='blue', linewidth=2, alpha=0.7)

def _plot_tile_boundaries(ax, raw_tile_data):
    """Plot tile boundaries if raw tile data is available"""
    if not raw_tile_data:
        return
    
    for tile_info in raw_tile_data:
        bounds = tile_info.get('bounds')
        if bounds:
            tile_shapely = box(bounds[0], bounds[1], bounds[2], bounds[3])
            x_coords, y_coords = tile_shapely.exterior.xy
            ax.plot(x_coords, y_coords, color='gray', linewidth=0.5, alpha=0.3)

def _extract_display_id(det_id):
    """Extract display ID by removing common prefixes"""
    display_id = det_id
    prefixes = ['merged_', 'det_', 'b_']
    for prefix in prefixes:
        if det_id.startswith(prefix):
            display_id = det_id.replace(prefix, '')
            break
    return display_id

def _process_single_building_detection(det_data, sequential_id, ax):
    """Process a single building detection and return patch data"""
    shapely_poly = det_data.get('polygon')
    confidence = det_data.get('confidence', 0.5)
    det_id = det_data.get('id', f"b_{sequential_id}")

    if not shapely_poly or not isinstance(shapely_poly, Polygon) or shapely_poly.is_empty:
        return None, None, None, None

    # Create matplotlib patch
    mpl_patch = shapely_polygon_to_mpl_patch(shapely_poly, linewidth=1, facecolor='none', edgecolor='none')
    
    # Plot centroid marker and ID
    centroid = shapely_poly.centroid
    ax.plot(centroid.x, centroid.y, marker='o', color='darkblue', markersize=1, alpha=0.6)
    
    # Use sequential ID instead of extracting from det_id
    ax.text(centroid.x, centroid.y, str(sequential_id), color='black', fontsize=4,
            ha='center', va='bottom')

    return mpl_patch, shapely_poly, confidence, det_id

def _process_building_detections(ax, merged_detections, polygon_area):
    """Process all building detections, filter by polygon area, sort by position, and return patch collections data"""
    building_patches_mpl = []
    shapely_polygons_for_eval = []
    confidence_values_for_color = []
    building_ids_for_text = []

    # Filter buildings that are inside the polygon area
    buildings_inside = []
    for det_data in merged_detections:
        shapely_poly = det_data.get('polygon')
        if shapely_poly and isinstance(shapely_poly, Polygon) and not shapely_poly.is_empty:
            centroid = shapely_poly.centroid
            # Check if building centroid is inside the study area polygon
            if polygon_area.contains(centroid):
                buildings_inside.append(det_data)
    
    # Sort buildings from top-left: latitude descending (north first), then longitude ascending (west first)
    buildings_inside.sort(key=lambda det: (-det.get('polygon').centroid.y, det.get('polygon').centroid.x))
    
    # Process sorted buildings with sequential IDs
    for sequential_id, det_data in enumerate(buildings_inside, start=1):
        mpl_patch, shapely_poly, confidence, det_id = _process_single_building_detection(det_data, sequential_id, ax)
        
        if mpl_patch is not None:
            building_patches_mpl.append(mpl_patch)
            shapely_polygons_for_eval.append(shapely_poly)
            confidence_values_for_color.append(confidence)
            building_ids_for_text.append(det_id)

    return building_patches_mpl, shapely_polygons_for_eval, confidence_values_for_color, building_ids_for_text

def _add_building_patches(ax, building_patches_mpl):
    """Add building patches to the plot"""
    if building_patches_mpl:
        normal_collection = PatchCollection(
            building_patches_mpl, facecolor='none', alpha=1.0, edgecolor='black', linewidth=0.3
        )
        ax.add_collection(normal_collection)

def _configure_plot_appearance(ax, results_data, raw_tile_data, building_patches_mpl):
    """Configure plot title, labels, and legend"""
    # Set title and labels
    ax.set_title(f"Merged Building Detections ({results_data.get('total_buildings', 0)} buildings)", fontsize=16)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='blue', lw=2, label='GeoJSON Area'),
    ]
    if raw_tile_data:
        legend_elements.append(plt.Line2D([0], [0], color='gray', lw=0.5, label='Tile Boundaries'))
    if building_patches_mpl:
        legend_elements.append(patches.Patch(facecolor='none', edgecolor='black', label='Building Detection'))
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize='small')
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.tight_layout(pad=1.5)

def _save_or_show_plot(output_path):
    """Save plot to file or show it"""
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Visualization of merged detections saved to {output_path}")
        plt.close()
    else:
        plt.show()

def visualize_polygon_detections(geojson_path, results_data, output_path=None):
    """
    Visualize merged building detections across the entire GeoJSON area on a single map.
    
    Args:
        geojson_path: Path to the GeoJSON file.
        results_data: Dictionary containing detection results, structured for merged detections.
                      Expected keys: 'merged_detections_shapely' (list of dicts with 'polygon', 'confidence'),
                                     'raw_tile_detections_for_background' (optional, for stitched background),
                                     'total_buildings', 'total_tiles', etc.
        output_path: Path to save the visualization (optional).
    """
    # Load and extract polygon data
    geojson_data_loaded = load_geojson(geojson_path)
    polygon_area = extract_polygon(geojson_data_loaded)
    
    # Setup plot area
    ax, bounds = _setup_plot_area(polygon_area)
    
    # Create background image
    raw_tile_data = results_data.get('raw_tile_detections_for_background')
    _create_background_image(ax, raw_tile_data, bounds)
    
    # Plot GeoJSON polygons
    _plot_geojson_polygons(ax, geojson_data_loaded, polygon_area)
    
    # Plot tile boundaries
    _plot_tile_boundaries(ax, raw_tile_data)
    
    # Process building detections
    merged_detections = results_data.get('merged_detections_shapely', [])
    building_patches_mpl, _, _, _ = _process_building_detections(ax, merged_detections, polygon_area)
    
    # Add building patches to plot
    _add_building_patches(ax, building_patches_mpl)
    
    # Configure plot appearance
    _configure_plot_appearance(ax, results_data, raw_tile_data, building_patches_mpl)
    
    # Save or show plot
    _save_or_show_plot(output_path) 