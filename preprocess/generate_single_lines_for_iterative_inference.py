import json
import os
import cv2
import numpy as np
import random
import math
import networkx as nx
from shapely.affinity import translate
from shapely.geometry import LineString, shape, mapping, MultiLineString, box, Point, GeometryCollection
from collections import defaultdict
from utils import normalize_linestring_orientation
from shapely.ops import snap, split, unary_union, linemerge
from shapely.strtree import STRtree
from collections import deque
from preprocess.generate_single_lines_for_inference_iter0 import filter_linestrings_within_buffer

PATCH_SIZE=500

def load_linestrings_from_geojson(geojson_file):
    """
    Load LineString objects from a GeoJSON file.

    Parameters:
    - geojson_file: Path to the GeoJSON file.

    Returns:
    - List of LineStrings.
    """
    with open(geojson_file, 'r') as f:
        geojson_data = json.load(f)
    
    lines = []
    for feature in geojson_data['features']:
        lines.append(feature['geometry'])
    return lines


def interpolate_linestring(line, threshold=15):
    """
    Interpolate evenly spaced nodes along a LineString.

    Parameters:
    - line: A Shapely LineString object.
    - num_points: The desired number of nodes along the line.

    Returns:
    - A new LineString with evenly spaced nodes.
    """
    if not isinstance(line, LineString):
        raise ValueError("Input must be a LineString.")

    # Calculate the total length of the line
    line_length = line.length
    # Calculate the number of points needed
    num_points = math.ceil(line_length / threshold) + 1  # Include both endpoints
    # Generate evenly spaced distances along the line
    distances = np.linspace(0, line_length, num_points)

    # Interpolate points at these distances
    interpolated_points = [line.interpolate(distance) for distance in distances]

    # Create a new LineString with the interpolated points
    return LineString(interpolated_points)

def translate_swamp_xy_line(line, x_translation, y_translation, switch_xy):
    """
    Process a single LineString by applying translation, coordinate switching, and interpolation.

    Parameters:
    - line: A LineString object.
    - x_translation: Translation in the x direction.
    - y_translation: Translation in the y direction.

    Returns:
    - Processed LineString.
    """
    if switch_xy:
        line = LineString([(y, x) for x, y in line.coords])
        line = translate(line, xoff=y_translation, yoff=x_translation)
    else:
        line = translate(line, xoff=x_translation, yoff=y_translation)
    return line


def save_paths_to_geojson(paths_list, patch_xys, output_file):
    """
    Save paths as a MultiLineString into a GeoJSON file.

    Parameters:
    - paths: List of LineString objects.
    - output_file: Path to the output GeoJSON file.

    Returns:
    - None
    """
    geojson_data = {
        "type": "FeatureCollection",
        "features": []
    }
    existing_features = []

    for i, paths in enumerate(paths_list):
        feat = {
                "type": "Feature",
                "geometry": paths,
                "properties": {
                    "patch_x": patch_xys[i][0],
                    "patch_y": patch_xys[i][1]
                }
            }
        existing_features.append(feat)
   
    # Update GeoJSON structure
    geojson_data["features"] = existing_features

    # Save to a GeoJSON file
    with open(output_file, "w") as f:
        json.dump(geojson_data, f, indent=2)

    print(f"Paths saved as MultiLineString to {output_file}")
    
def angle_of_line(line):
    """Compute the orientation angle (in degrees) of a LineString."""
#     start, end = line.coords[0], line.coords[-1]
    p0 = line.coords[0]
    p1 = line.coords[-1]
    start, end = sorted([p0, p1])  # canonical order
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle = math.degrees(math.atan2(dy, dx)) % 180  # 0-180 symmetry
    return angle

def angle_difference(a1, a2):
    """Compute the minimal angle difference between two angles (0–180)."""
    return min(abs(a1 - a2), 180 - abs(a1 - a2))


def filter_linestrings_within_bbox_input(geojson_file, bbox, shift_threshold=1500, switch_xy=False):
    """
    Filter LineString geometries from a GeoJSON file within a bounding box.

    Parameters:
    - geojson_file: Path to the GeoJSON file.
    - bbox: A tuple (minx, miny, maxx, maxy) representing the bounding box.

    Returns:
    - List of LineStrings within the bounding box.
    """
    with open(geojson_file, 'r') as f:
        geojson_data = json.load(f)

    # Create a bounding box polygon
    bbox_polygon = box(*bbox)
    # Randomize translation if bbox is larger than the threshold
    apply_translation = max(bbox) > shift_threshold
    # Apply random translation if bbox exceeds threshold
    x_translation,y_translation = 0, 0
    if apply_translation:
        minx, miny, maxx, maxy = bbox

        # Randomize x and y translations within bounds
        x_translation = -minx
        y_translation = -miny
    
    filtered_lines = []
    for feature in geojson_data['features']:
        geometry = shape(feature['geometry'])

        # Handle LineString and MultiLineString
        if isinstance(geometry, LineString):
            geometries = [geometry]
        elif isinstance(geometry, MultiLineString):
            geometries = list(geometry.geoms)
        else:
            print(f"Unsupported geometry type: {feature['geometry']['type']}")
            continue

        for line in geometries:
            # Check if the LineString intersects the bbox
            if line.intersects(bbox_polygon):
                # Intersect LineString with bbox
                geom_within_bbox = line.intersection(bbox_polygon)
                line_within_bbox = []
                if isinstance(geom_within_bbox, GeometryCollection):
                    for geom in geom_within_bbox.geoms:
                        if isinstance(geom, LineString):
                            line_within_bbox.append(geom)
                        elif isinstance(geom, MultiLineString):
                            line_within_bbox.extend(list(geom.geoms))
                    line_within_bbox = MultiLineString(line_within_bbox)
                elif isinstance(geom_within_bbox, MultiLineString) or isinstance(geom_within_bbox, LineString):
                    line_within_bbox = geom_within_bbox
                
                # Handle MultiLineString resulting from intersection
                if isinstance(line_within_bbox, MultiLineString):
                    for subline in line_within_bbox.geoms:
                        if len(list(subline.coords)) < 2:
                            continue  
                        subline = translate_swamp_xy_line(subline, x_translation, y_translation, switch_xy)
                        interp_subline = interpolate_linestring(subline)
                        filtered_lines.append(interp_subline)
                elif isinstance(line_within_bbox, LineString):
                    if len(list(line_within_bbox.coords)) < 2:
                        continue                   
                    line_within_bbox = translate_swamp_xy_line(line_within_bbox, x_translation, y_translation, switch_xy)
                    interp_line_within_bbox = interpolate_linestring(line_within_bbox)
                    filtered_lines.append(interp_line_within_bbox)
    return filtered_lines, (x_translation, y_translation)

def average_orientation(line):
    """
    Calculate the line orientation by averaging
    all orientations of segments
    """
    """
    Calculate the average orientation of a LineString based on its segments.

    Returns:
        angle in degrees (range: [0, 180))
    """
    coords = list(line.coords)
    if len(coords) < 2:
        return None  # Not enough points

    # Compute direction vectors of segments
    vectors = []
    for (x1, y1), (x2, y2) in zip(coords[:-1], coords[1:]):
        dx = x2 - x1
        dy = y2 - y1
        norm = math.hypot(dx, dy)
        if norm != 0:
            vectors.append((dx / norm, dy / norm))

    # Average vector components
    if not vectors:
        return None
    avg_dx, avg_dy = np.mean(vectors, axis=0)

    # Convert to angle in degrees
    angle_rad = math.atan2(avg_dy, avg_dx)
    angle_deg = math.degrees(angle_rad) % 180  # Orientation in [0,180)
    return angle_deg

def extend_linestring_by_average_orientation(line, bounds=(0, 0, 500, 500), extension_length=1000):
    """
    Extend a LineString in both directions along its average orientation,
    and clip it to the bounding box.

    Args:
        line (LineString): Input line.
        bounds (tuple): (minx, miny, maxx, maxy)
        extension_length (float): Distance to extend on both ends.

    Returns:
        LineString or None: Clipped LineString or None if invalid.
    """
    coords = np.array(line.coords)

    if len(coords) < 2:
        return None

    # Compute average segment vector
    vectors = coords[1:] - coords[:-1]
    avg_vector = np.mean(vectors, axis=0)
    norm = np.linalg.norm(avg_vector)
    if norm == 0:
        return None

    direction = avg_vector / norm

    # Extend both ends
    start_ext = coords[0] - direction * extension_length
    end_ext = coords[-1] + direction * extension_length

    extended = LineString([tuple(start_ext)] + [tuple(pt) for pt in coords] + [tuple(end_ext)])

    # Clip to bounding box
    boundary = box(*bounds)
    clipped = extended.intersection(boundary)

    # Return longest part if multi-segment
    if clipped.is_empty:
        return None
    elif clipped.geom_type == "LineString":
        return clipped
    elif clipped.geom_type == "MultiLineString":
        return max(clipped.geoms, key=lambda g: g.length)
    else:
        return None

def get_templates(lines):
    line_template_dict = {}
    for ln in lines:
        ext_ln = extend_linestring_by_average_orientation(ln)
        if ext_ln is not None:
            line_template_dict[ln] = ext_ln
    return line_template_dict

def point_to_line_distance(point, line):
    """
    Computes the perpendicular distance from a point to the infinite line
    defined by a LineString.
    """
    # Get two endpoints of the line
    x1, y1 = line.coords[0]
    x2, y2 = line.coords[-1]
    # Calculate line parameters (ax + by + c = 0)
    a = y2 - y1
    b = x1 - x2
    c = x2*y1 - x1*y2
    # Distance formula: |ax0 + by0 + c| / sqrt(a^2 + b^2)
    x0, y0 = point.x, point.y
    return abs(a*x0 + b*y0 + c) / np.hypot(a, b)

def are_lines_collinear(line1, line2, angle_tol=5, dist_tol=5):
    """
    Determines whether two LineStrings are along the same infinite line.
    
    Parameters:
      - line1, line2: Input LineString objects.
      - angle_tol: Maximum orientation difference (in degrees) to consider them collinear.
      - dist_tol: Maximum perpendicular distance (in the same units as your geometries) 
                  allowed between one line and the infinite extension of the other.
                  
    Returns:
      - True if the lines are considered collinear (i.e. along the same infinite line),
        False if they are merely parallel or non-collinear.
    """
    if line2.within(line1.buffer(20)) or \
                line2.intersection(line1.buffer(20)).length /line2.length >= 0.5:
        return False
    # Calculate orientations in a canonical way.
    orient1 = angle_of_line(line1)
    orient2 = angle_of_line(line2)
    angle_diff = angle_difference(orient1, orient2)
    
    if angle_diff > angle_tol:
        return False  # Their orientations differ too much, not collinear.
    if line1.distance(line2) > dist_tol:
        return True
    # For collinearity, check the perpendicular distance from one line’s midpoint
    # to the infinite line of the other.
    mid1 = Point(np.mean([pt[0] for pt in line1.coords]), np.mean([pt[1] for pt in line1.coords]))
    mid2 = Point(np.mean([pt[0] for pt in line2.coords]), np.mean([pt[1] for pt in line2.coords]))
    
    # Compute distance from mid1 to infinite line of line2, and vice versa.
    d1 = point_to_line_distance(mid1, line2)
    d2 = point_to_line_distance(mid2, line1)
#     print(d1, d2, (d1 < dist_tol) or (d2 < dist_tol))
    # If either perpendicular distance is below the tolerance, we consider them collinear.
    return (d1 >= dist_tol) and (d2 >= dist_tol)


def filter_noise_lines(lines, main_line, angle_threshold=30):
    """
    Removes noise lines that have a significantly different orientation from the main line.

    :param lines: List of Shapely LineString objects
    :param main_line: Reference main LineString
    :param angle_threshold: Maximum allowed deviation (default: 30 degrees)
    :return: List of filtered LineString objects
    """
    main_angle = average_orientation(main_line)  # Get reference angle

    def is_valid(line):
        line_angle = average_orientation(line)
        angle_diff = abs(line_angle - main_angle)
        angle_diff = min(angle_diff, 180-angle_diff)
#         print(main_angle, line_angle, angle_diff)
        return angle_diff <= angle_threshold  # Keep if within threshold
    
    def is_within(target_line, query_line):
#         print(query_line.within(target_line.buffer(20)),
#                 query_line.intersection(target_line.buffer(20)).length /query_line.length, query_line.length)
        return query_line.within(target_line.buffer(20)) or \
                query_line.intersection(target_line.buffer(20)).length /query_line.length >= 0.5

    # Apply filtering
    filtered_lines = [line for line in lines if is_valid(line) or is_within(main_line, line)]
    final_filter = [line for line in filtered_lines if not are_lines_collinear(main_line, line)]
    # Calculate total length of the filtered lines
    total_length = sum(line.length for line in final_filter)
#     print(main_angle, total_length, main_line.length, total_length/main_line.length, len(filtered_lines))
    return final_filter, total_length

def deduplicate_linestring_groups(groups):
    """
    Deduplicate a list of LineString groups (list of lists), treating order as irrelevant.
    Two groups with the same LineStrings (regardless of order) are considered duplicates.

    Args:
        groups (List[List[LineString]]): List of LineString groups.

    Returns:
        List[List[LineString]]: Deduplicated groups.
    """
    seen = set()
    unique = []

    for group in groups:
        # Use sorted WKT strings to define a unique key
        key = tuple(sorted(geom.wkt for geom in group))
        if key not in seen:
            seen.add(key)
            unique.append(group)

    return unique
# Main Workflow
def process_geojson_to_paths(input_geojson_file, map_file, save_input_geojson_path,\
                             stride=500, patch_size=500, dim_threshold=3000):
    """
    Process a GeoJSON file to construct paths from LineStrings.

    Parameters:
    - geojson_file: Path to the GeoJSON file.

    Returns:
    - List of LineStrings representing the paths.
    """
    
    map_image = cv2.imread(map_file)
    h, w, _ = map_image.shape
    
    input_lines = load_linestrings_from_geojson(input_geojson_file)
    interp_input_lines = []
    patch_xys = []
    cnt = 0
    for x in range(0, w, stride):
        for y in range(0, h, stride):
#             if not (x==1500 and y==3500):
#                 continue
            input_roi = [x, y, x+patch_size, y+patch_size]
            input_line_list, (tr_x, tr_y) = filter_linestrings_within_bbox_input(input_geojson_file, input_roi,\
                                                              shift_threshold=dim_threshold, switch_xy=False)
            _ln_linetr = [LineString(ln) for ln in input_line_list]
            ln_linetr = sorted(_ln_linetr, key=lambda ln: ln.length, reverse=True)

            grouped_lines = []
            visited = {}
            for ln in ln_linetr:
                visited[ln] = False
            
            templates = get_templates(ln_linetr)
            i = 0
            for ln, ext_ln in templates.items():
                if ln.length < 60:
                    continue
                _matched_linestrings, _matched_length = \
                    filter_linestrings_within_buffer(ext_ln, ln_linetr, buffer_distance=10)
                
                matched_linestrings, matched_length = filter_noise_lines(_matched_linestrings, ext_ln, angle_threshold=5)
#                 matched_linestrings, matched_length = further_filter_noise_lines(matched_linestrings, angle_threshold=30, \
#                                                                                  dist_threshold=20)
                
                if matched_linestrings == [] or matched_length<min(ext_ln.length, PATCH_SIZE)//5 or matched_length<50: #
                    matched_linestrings = [ln]
                grouped_lines.append(matched_linestrings)
#                 print(f"===== {cnt} =====")
                cnt += 1
            dedup_grouped_lines = deduplicate_linestring_groups(grouped_lines)

            for group in dedup_grouped_lines:
                interp_ln_list = [interpolate_linestring(ln) for ln in group]

                _interp_input_lines = [list(normalize_linestring_orientation(interp_ln).coords) \
                                           for interp_ln in interp_ln_list]
                interp_input_lines.append(_interp_input_lines)
                patch_xys.append([x, y])
                
    print(len(interp_input_lines), len(patch_xys))
    save_paths_to_geojson(interp_input_lines, patch_xys, save_input_geojson_path)

if __name__ == "__main__": 
    input_dir = "/data/weiweidu/criticalmaas_data/eval_15month_line_results/geojson"
    map_dir = "/data/weiweidu/criticalmaas_data/eval_15month_line_results/maps"
    output_dir = './inference_input_data'
    os.makedirs(output_dir, exist_ok=True)

#     input_dir = '/data/weiweidu/criticalmaas_data/line_results_eval'
#     map_dir = "/data/weiweidu/criticalmaas_data/eval_data_perfomer"
#     output_dir = './inference_input_data'
    
    for subdir in os.listdir(map_dir):
        map_name = subdir.split('.')[0]
#         if map_name != "CO_DenverW":
        if map_name != "28c001b9156917e92339642964f935a933c95349134963495649964552ecc1e1":
            continue
        print(f'===> Processing {map_name} <===')
        map_image_path = f"{map_dir}/{map_name}.tif"
        
        input_geojson_path = f"{input_dir}/{map_name}_fault_line.geojson"
#         input_geojson_path = f"{input_dir}/{map_name}/{map_name}_fault_line.geojson"
        input_output_path = f"{output_dir}/{map_name}.geojson"
        
        if not os.path.exists(map_image_path) or not os.path.exists(input_geojson_path):
            print(map_image_path)
            print(input_geojson_path)
            print("Files don't exist!")
            continue
        
        process_geojson_to_paths(input_geojson_path, map_image_path, input_output_path,\
                                 stride=500, dim_threshold=500)
        print(f'===> Processed {map_name} <===')