import os
import json
import math
import cv2
import numpy as np
from shapely.geometry import mapping, MultiLineString, LineString, Point
from shapely.affinity import translate

def load_linestring_from_geojson_for_pretrain(file_path):
    """
    Load trajectories from a GeoJSON file containing MultiLineString geometries.

    Args:
        file_path (str): Path to the GeoJSON file.

    Returns:
        List[List[List[str]]]: Multi-trajectory samples. Each sample contains multiple trajectories.
    """
    with open(file_path, 'r') as f:
        geojson_data = json.load(f)
    
    trajectories = []
    for feature in geojson_data['features']:
        linestring = feature['geometry']['coordinates']
        # Convert coordinates to "x,y" format
        trajectory = [[int(x),int(y)] for x, y in linestring]
        trajectories.append(trajectory)   
    return trajectories

def load_linestring_from_geojson_for_finetune(file_path):
    """
    Load trajectories from a GeoJSON file containing MultiLineString geometries.

    Args:
        file_path (str): Path to the GeoJSON file.

    Returns:
        List[List[List[str]]]: Multi-trajectory samples. Each sample contains multiple trajectories.
    """
    with open(file_path, 'r') as f:
        geojson_data = json.load(f)
    
    trajectories = []
    shifts = []
    patch_xys = []
    for feature in geojson_data['features']:
        multilinestring = feature['geometry']
        sample = []
        for linestring in multilinestring:
            # Convert coordinates to "x,y" format
            trajectory = [[int(x),int(y)] for x, y in linestring]
            sample.append(trajectory)
        trajectories.append(sample) 
        shifts.append(
            (int(feature['properties'].get('tr_x', 0)), int(feature['properties'].get('tr_y', 0)))
            if feature.get('properties') else (0, 0)
        )

        patch_xys.append(
            (int(feature['properties'].get('patch_x', 0)), int(feature['properties'].get('patch_y', 0)))
            if feature.get('properties') else (0, 0)
)
    return trajectories, shifts, patch_xys

def normalize_linestring_orientation(line):
    """
    Normalize the orientation of a LineString such that the node with the smaller
    x-coordinate comes first. If the x-coordinates are the same, use the y-coordinate
    to decide the order.

    Parameters:
    - line: A Shapely LineString object.

    Returns:
    - A LineString with normalized orientation.
    """
        
    if not isinstance(line, LineString) and not isinstance(line, list):
        raise ValueError("Input must be a LineString object or a list.")
    if isinstance(line, LineString):
        line = list(line.coords)

    if len(line) < 2:
        raise ValueError("LineString must have at least two points.")

    start = line[0]
    end = line[-1]

    # Reverse the line if the x of the first point is greater than the x of the last point
    # Or if x is the same, use y to determine order
    if start[0] > end[0] or (start[0] == end[0] and start[1] > end[1]):
        return LineString(line[::-1])

    return LineString(line)

def draw_multilinestring_on_image(multi_lines, image_size, output_image_path, base_image=None, \
                                  line_color=(255, 255, 255), line_thickness=4, switch_xy=False, random_color=True):
    """
    Draw a MultiLineString on an image, shifting it to fit within [0,500] if needed.

    Parameters:
    - multi_line: A Shapely MultiLineString object.
    - image_size: Tuple (height, width) for the output image.
    - output_image_path: Path to save the output image.
    - line_color: Tuple (B, G, R) for the line color (default: green).
    - line_thickness: Thickness of the lines (default: 2).
    """
#     print(shifted_multi_line)
    # Create a blank image (black background)
    height, width = image_size
    if base_image is None:
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
    else: 
        image = base_image
    
    if multi_lines == []:
        cv2.imwrite(output_image_path, image)
        return

    # Iterate over each LineString in the MultiLineString
    for line in multi_lines:
        # Generate random color for the line
#         r, g, b = np.random.randint(0, 256, size=3)
        if random_color:
            r = np.random.randint(150, 256)  # High red (200-255)
            g = np.random.randint(0, 101)    # Low green (0-100)
            b = np.random.randint(0, 101)    # Low blue (0-100)
        else:
            r, g, b = line_color
        # Convert line coordinates to OpenCV format
        if isinstance(line, LineString):
            coords = np.array(list(line.coords), dtype=np.int32)
        elif isinstance(line, list):
             coords = np.array(line, dtype=np.int32)
        else:
            raise("line must be LineString or List")
        # If switch_xy is True, swap X and Y coordinates
        if switch_xy:
            coords = coords[:, [1, 0]]  # Swap columns X ↔ Y
        coords = coords.reshape((-1, 1, 2))  # Reshape for OpenCV
        # Draw the line on the image
        cv2.polylines(image, [coords], isClosed=False, color=(int(b),int(g),int(r)), \
                      thickness=line_thickness)

    # Save the image
    cv2.imwrite(output_image_path, image)
    return image

def convert_prediction_to_multilinestring(coord_list, max_id=500):
    """
    Convert a list of coordinates into a MultiLineString, handling specific rules:
    - Remove coordinates (1501, 1501), (1502, 1502), and (1503, 1503).
    - Start a new LineString whenever encountering (1504, 1504).

    Parameters:
    - coord_list: List of coordinate tuples.

    Returns:
    - MultiLineString object.
    """
    lines = []
    segment = []

    for coord in coord_list:
        # Skip specific coordinates
        if coord in [(max_id+1, max_id+1), (max_id+2, max_id+2), (max_id+3, max_id+3)]:
            continue
        
        # Start a new LineString when encountering (1504, 1504)
        if coord == (max_id+4, max_id+4):
            if len(segment) > 1:  # Add the current segment if it has enough points
                lines.append(segment)
            segment = []  # Start a new segment
            continue

        # Add the current coordinate to the segment
        segment.append(coord)

    # Add the last segment to the lines
    if len(segment) > 1:
        lines.append(segment)

    return lines

def save_line_groups_to_geojson(line_patch_groups, output_path):
    """
    Saves a list of list of LineStrings to a GeoJSON file.
    Each inner list becomes a MultiLineString feature.
    
    Args:
        line_groups: List of lists of LineString objects.
        output_path: File path to save GeoJSON.
    """
    features = []
    for key, value in line_patch_groups.items():
        line_list  = [list(line.coords) for line in value]
        feature = {
            "type": "Feature",
            "geometry": line_list,
            "properties": {
                "patch_x": key[0],
                "patch_y": key[1]
            } 
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    with open(output_path, "w") as f:
        json.dump(geojson, f, indent=2)
    print(f"Saved {len(features)} MultiLineString features to {output_path}")