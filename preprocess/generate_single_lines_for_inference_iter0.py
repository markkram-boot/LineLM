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
from process_data.multiline_to_trajectory import MultiLineGraph
from utils import normalize_linestring_orientation
from process_data.transform_and_buffer_line import transform_and_extend_line
from shapely.ops import nearest_points

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
        if feature['geometry']['type'] == 'LineString':
            lines.append(shape(feature['geometry']))
    
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

def split_linestring_into_segments(line):
    """
    Split a LineString into individual segments while keeping only the start and end nodes.

    :param line: A Shapely LineString object.
    :return: A list of LineString segments.
    """
    coords = list(line.coords)  # Extract coordinates
    segments = [LineString([coords[i], coords[i + 1]]) for i in range(len(coords) - 1)]
    
    return segments

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
                        filtered_lines.extend(split_linestring_into_segments(interp_subline))
                elif isinstance(line_within_bbox, LineString):
                    if len(list(line_within_bbox.coords)) < 2:
                        continue                   
                    line_within_bbox = translate_swamp_xy_line(line_within_bbox, x_translation, y_translation, switch_xy)
                    interp_line_within_bbox = interpolate_linestring(line_within_bbox)
                    filtered_lines.extend(split_linestring_into_segments(interp_line_within_bbox))

    return filtered_lines, (x_translation, y_translation)

def save_paths_to_geojson(paths_list, tr_xs, tr_ys, patch_xs, patch_ys, reference_lines, output_file):
    """
    Save paths as a MultiLineString into a GeoJSON file.

    Parameters:
    - paths: List of LineString objects.
    - output_file: Path to the output GeoJSON file.

    Returns:
    - None
    """
    # Load or initialize GeoJSON data
    geojson_data = {
        "type": "FeatureCollection",
        "features": []
    }
    existing_features = []
        
    for i, paths in enumerate(paths_list):
        feat = {
                "type": "Feature",
                "geometry": [list(l.coords) for l in paths],
                "properties": {
                    "tr_x": tr_xs[i],
                    "tr_y": tr_ys[i],
                    "patch_x": patch_xs[i],
                    "patch_y": patch_ys[i],
                    "reference_line": list(reference_lines[i].coords)
                }
            }
        existing_features.append(feat)
   
    # Update GeoJSON structure
    geojson_data["features"] = existing_features

    # Save to a GeoJSON file
    with open(output_file, "w") as f:
        json.dump(geojson_data, f, indent=2)

    print(f"Paths saved as MultiLineString to {output_file}")
    
def filter_linestrings_within_buffer(target_line, lines_list, buffer_distance, intersect_distance_threshold=0):
    """
    Given a LineString A and a list of LineStrings B, buffer A and return all LineStrings in B within the buffered A.

    Parameters:
    - target_line (LineString): The LineString to buffer.
    - lines_list (list of LineString): A list of LineStrings to check against the buffer.
    - buffer_distance (float): The distance for buffering the target_line.

    Returns:
    - list of LineString: LineStrings from B that fall within the buffered A.
    """
    if not isinstance(target_line, LineString):
        raise ValueError("target_line must be a LineString")
    
    if not all(isinstance(line, LineString) for line in lines_list):
        raise ValueError("All elements in lines_list must be LineStrings")

    # Buffer the target LineString
    buffered_area = target_line.buffer(buffer_distance)

    filtered_lines = []
    
    for line in lines_list:
        if line.intersects(buffered_area):  # Check if the line intersects with the buffer
            intersection = line.intersection(buffered_area)  # Get the intersecting portion           
            # Compute intersection length (only valid if intersection is a LineString or MultiLineString)
            if intersection.length / line.length > intersect_distance_threshold:  # include small branches:
                filtered_lines.append(line)
    length = sum([l.length for l in filtered_lines])

    return filtered_lines, length

def calculate_line_orientation(line):
    """
    Computes the orientation (angle) of a LineString in degrees.

    :param line: A Shapely LineString
    :return: Angle in degrees (0-180)
    """
#     # Extract first and last points of the LineString
#     x1, y1 = line.coords[0]
#     x2, y2 = line.coords[-1]

#     # Compute the angle using arctan2
#     angle_rad = np.arctan2(y2 - y1, x2 - x1)
#     angle_deg = np.degrees(angle_rad) % 180  # Normalize to [0, 180] range

#     return angle_deg
    p0 = line.coords[0]
    p1 = line.coords[-1]
    start, end = sorted([p0, p1])  # canonical order
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad) % 180
    return angle_deg

def shortest_line_between(line1, line2):
    """
    Given two LineString objects, calculates the shortest connecting LineString between them.
    
    :param line1: First LineString.
    :param line2: Second LineString.
    :return: A tuple containing:
             - The connecting LineString representing the shortest distance between the two.
             - The length of this connecting LineString.
    """
    # Find the nearest points between line1 and line2.
    p1, p2 = nearest_points(line1, line2)
    
    # Create a LineString connecting the two nearest points.
    connecting_line = LineString([p1, p2])
    
    return connecting_line


def filter_noise_lines(lines, main_line, angle_threshold=30):
    """
    Removes noise lines that have a significantly different orientation from the main line.

    :param lines: List of Shapely LineString objects
    :param main_line: Reference main LineString
    :param angle_threshold: Maximum allowed deviation (default: 30 degrees)
    :return: List of filtered LineString objects
    """
    main_angle = calculate_line_orientation(main_line)  # Get reference angle

    def is_valid(line):
        line_angle = calculate_line_orientation(line)
        angle_diff = abs(line_angle - main_angle)
        angle_diff = min(angle_diff, 180-angle_diff)
#         print(main_angle, line_angle)
        return angle_diff <= angle_threshold  # Keep if within threshold

    # Apply filtering
    filtered_lines = [line for line in lines if is_valid(line)]
    # Calculate total length of the filtered lines
    total_length = sum(line.length for line in filtered_lines)
#     print(main_angle, total_length, main_line.length, total_length/main_line.length, len(filtered_lines))
    return filtered_lines, total_length


def extend_line_to_bounds(line, bounds):
    """
    Extends a given LineString to the intersection with the bounding box defined by bounds.
    
    Parameters:
      - line: A Shapely LineString.
      - bounds: Tuple of ((min_x, min_y), (max_x, max_y)).
    
    Returns:
      - A LineString representing the infinite line clipped to the bounding box,
        or None if no intersection occurs.
    """
    (min_x, min_y), (max_x, max_y) = bounds
    bbox = box(min_x, min_y, max_x, max_y)
    
    # Compute the direction of the line.
    p0 = line.coords[0]
    p1 = line.coords[-1]
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    norm = (dx**2 + dy**2) ** 0.5
    if norm == 0:
        return line  # Degenerate line.
    dx /= norm
    dy /= norm

    # Create an "infinite" line by extending far in both directions.
    factor = 10000  # A sufficiently large factor.
    infinite_line = LineString([
        (p0[0] - factor * dx, p0[1] - factor * dy),
        (p0[0] + factor * dx, p0[1] + factor * dy)
    ])
    
    # Intersect with the bounding box.
    extended = infinite_line.intersection(bbox)
    if extended.is_empty:
        return None
    if extended.geom_type == 'MultiLineString':
        # Choose the longest segment if multiple parts result.
        extended = max(extended, key=lambda l: l.length)
    return extended

def extended_lines_relationship(line1, line2, bounds):
    """
    Given two LineStrings, extends them to the provided bounding box and then:
      - If the extended lines intersect, returns the intersection geometry.
      - Otherwise, returns the shortest connecting LineString between them.
    
    Parameters:
      - line1, line2: Input Shapely LineString objects.
      - bounds: Bounding box as ((min_x, min_y), (max_x, max_y)).
      
    Returns:
      - ext1: Extended version of line1.
      - ext2: Extended version of line2.
      - result: Either the intersection geometry (if they intersect) or a connecting LineString.
      - method: A string indicating whether 'intersection' or 'nearest' was used.
    """
    ext1 = extend_line_to_bounds(line1, bounds)
    ext2 = extend_line_to_bounds(line2, bounds)
    if ext1 is None or ext2 is None:
        return None, None, None, "invalid"
    
    if ext1.intersects(ext2):
        inter = ext1.intersection(ext2)
        method = "intersection"
        return inter, method
    else:
        p1, p2 = nearest_points(ext1, ext2)
        connecting = LineString([p1, p2])
        method = "nearest"
        return connecting, method

def circular_difference(angle1, angle2):
    """
    Computes the minimal circular difference between two angles (in degrees) within [0, 180).

    :param angle1: First angle in degrees.
    :param angle2: Second angle in degrees.
    :return: The minimal difference in degrees.
    """
    diff = abs(angle1 - angle2)
    return min(diff, 180 - diff)

# def further_filter_noise_lines(lines, angle_threshold=30):
#     """
#     Filters out noise LineStrings from a list based on two conditions:
#       1. Each line’s orientation (compared to the main_line) must be within angle_threshold.
#       2. For every pair among the initially filtered lines, the shortest connecting LineString’s
#          orientation (compared to the main_line) must also be within angle_threshold.
#          If not, both lines in that pair are removed.

#     :param lines: List of Shapely LineString objects.
#     :param angle_threshold: Maximum allowed deviation in degrees (default: 30).
#     :return: Tuple (final_filtered_lines, total_length) where:
#              - final_filtered_lines: List of filtered LineString objects.
#              - total_length: Sum of lengths of the final filtered lines.
#     """
#     n = len(lines)

#     remove_indices = set()
#     for i in range(n):
#         for j in range(i + 1, n):
#             if i in remove_indices or j in remove_indices:
#                 continue
#             connecting_line = shortest_line_between(lines[i], lines[j])
#             connect_angle = calculate_line_orientation(connecting_line)
#             # Compare connecting line's orientation with each line's orientation.
#             diff_connect_i = circular_difference(connect_angle, calculate_line_orientation(lines[i]))
#             diff_connect_j = circular_difference(connect_angle, calculate_line_orientation(lines[j]))

#             if connecting_line.length <= 5:
#                 continue
#             if diff_connect_i > angle_threshold or diff_connect_j > angle_threshold:
#                 remove_indices.add(i)
#                 remove_indices.add(j)
    
#     final_filtered = [lines[i] for i in range(n) if i not in remove_indices]
#     total_length = sum(line.length for line in final_filtered)
    
#     return final_filtered, total_length

# def further_filter_noise_lines(lines, angle_threshold=30, bounds=((0,0),(500,500))):
#     """
#     Filters out noise LineStrings from a list based on two conditions:
#       1. Each line’s orientation (compared to the main_line) must be within angle_threshold.
#       2. For every pair among the initially filtered lines, the shortest connecting LineString’s
#          orientation (compared to the main_line) must also be within angle_threshold.
#          If not, both lines in that pair are removed.

#     :param lines: List of Shapely LineString objects.
#     :param angle_threshold: Maximum allowed deviation in degrees (default: 30).
#     :return: Tuple (final_filtered_lines, total_length) where:
#              - final_filtered_lines: List of filtered LineString objects.
#              - total_length: Sum of lengths of the final filtered lines.
#     """
#     n = len(lines)

#     remove_indices = set()
#     for i in range(n):
#         for j in range(i + 1, n):
#             if i in remove_indices or j in remove_indices:
#                 continue
#             connecting_line = shortest_line_between(lines[i], lines[j])
#             connect_angle = calculate_line_orientation(connecting_line)
#             # Compare connecting line's orientation with each line's orientation.
#             diff_connect_i = circular_difference(connect_angle, calculate_line_orientation(lines[i]))
#             diff_connect_j = circular_difference(connect_angle, calculate_line_orientation(lines[j]))

#             if connecting_line.length <= 5:
#                 continue
#             if diff_connect_i > angle_threshold or diff_connect_j > angle_threshold:
#                 remove_indices.add(i)
#                 remove_indices.add(j)
            
#             ext_connecting_line, relation = extended_lines_relationship(lines[i], lines[j], bounds)
#             if relation == "nearest" and ext_connecting_line.length > 0:
#                 remove_indices.add(i)
#                 remove_indices.add(j)
                
    
#     final_filtered = [lines[i] for i in range(n) if i not in remove_indices]
#     total_length = sum(line.length for line in final_filtered)
    
#     return final_filtered, total_length

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

def differentiate_line_relationship(lineA, lineB):
    """
    Differentiates the position relationship between two LineStrings.
    Returns a string describing their relationship.
    
    - If the perpendicular distance from an endpoint of one line to the
      infinite extension of the other is less than tol, then they are collinear,
      meaning they lie along the same infinite line (they may be separated by a gap).
    - Otherwise, they are parallel but offset (or not parallel at all).
    """
    # Compute orientations
    angleA = calculate_line_orientation(lineA)
    angleB = calculate_line_orientation(lineB)
    
    # Use one endpoint of lineB and compute its perpendicular distance to lineA's infinite line.
    ptB1 = Point(lineB.coords[0])
    dist1 = point_to_line_distance(ptB1, lineA)
    
    ptB2 = Point(lineB.coords[-1])
    dist2 = point_to_line_distance(ptB2, lineA)
    
    return min(dist1, dist2)

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
    # Calculate orientations in a canonical way.
    orient1 = calculate_line_orientation(line1)
    orient2 = calculate_line_orientation(line2)
    angle_diff = circular_difference(orient1, orient2)
    
    if angle_diff > angle_tol:
        return False  # Their orientations differ too much.
    
    # For collinearity, check the perpendicular distance from one line’s midpoint
    # to the infinite line of the other.
    mid1 = Point(np.mean([pt[0] for pt in line1.coords]), np.mean([pt[1] for pt in line1.coords]))
    mid2 = Point(np.mean([pt[0] for pt in line2.coords]), np.mean([pt[1] for pt in line2.coords]))
    
    # Compute distance from mid1 to infinite line of line2, and vice versa.
    d1 = point_to_line_distance(mid1, line2)
    d2 = point_to_line_distance(mid2, line1)
#     print(d1, d2, (d1 < dist_tol) or (d2 < dist_tol))
    # If either perpendicular distance is below the tolerance, we consider them collinear.
    return (d1 < dist_tol) or (d2 < dist_tol)

def further_filter_noise_lines(lines, angle_threshold=30, dist_threshold=10):
    """
    Filters out noise LineStrings from a list based on two conditions:
      1. Each line’s orientation (compared to the main_line) must be within angle_threshold.
      2. For every pair among the initially filtered lines, the shortest connecting LineString’s
         orientation (compared to the main_line) must also be within angle_threshold.
         If not, both lines in that pair are removed.

    :param lines: List of Shapely LineString objects.
    :param angle_threshold: Maximum allowed deviation in degrees (default: 30).
    :return: Tuple (final_filtered_lines, total_length) where:
             - final_filtered_lines: List of filtered LineString objects.
             - total_length: Sum of lengths of the final filtered lines.
    """
    n = len(lines)

    remove_indices = set()
    for i in range(n):
        for j in range(i + 1, n):
            if i in remove_indices or j in remove_indices:
                continue
#             dist = differentiate_line_relationship(lines[i], lines[j])
#             if dist > dist_threshold:
            if not are_lines_collinear(lines[i], lines[j], angle_threshold, dist_threshold):
                remove_indices.add(i)
                remove_indices.add(j)              
    
    final_filtered = [lines[i] for i in range(n) if i not in remove_indices]
    total_length = sum(line.length for line in final_filtered)
    
    return final_filtered, total_length


def get_buffer_zones(start, end, angle_interval=10, shift_interval=10, bounds=((0,0), (500,500))):
    buffer_zones = []

    # Loop through angles in the given interval (0° to 180°)
    for angle in range(0, 180, angle_interval):
#     for angle in range(50, 60, angle_interval):
        for shift in range(-500, 500, shift_interval):
            # Apply dx or dy based on angle rules
            if 0 <= angle <= 45 or 135 < angle < 180:  # dx shift
                dx, dy = shift, 0
            else:  # dy shift (0,90] and (90,135]
                dx, dy = 0, shift

            # Apply transformation
            buffered_line = transform_and_extend_line(start, end, angle, dx, dy, bounds=bounds)
            if buffered_line is not None:
                buffer_zones.append(buffered_line)

    return buffer_zones

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

    input_traj_list = []
    tr_x_list, tr_y_list = [], []
    patch_x_list, patch_y_list = [], []
    ref_lines = []
    referece_lines = get_buffer_zones((250,0), (250,500), angle_interval=5, shift_interval=10)
    
#     plain_image = np.ones((500,500,3)) * 255
#     for line in referece_lines:
#         if isinstance(line, LineString):
#             # Convert Shapely LineString to NumPy array (integer coordinates)
#             pts = np.array(line.coords, dtype=np.int32)

#             # Draw the line on the image
#             cv2.polylines(plain_image, [pts], isClosed=False, color=(0,0,255), thickness=1)
#         else:
#             raise TypeError("Each element in the list must be a Shapely LineString.")
#     cv2.imwrite('template.png', plain_image)
    cnt = 0
    for x in range(0, w, stride):
        for y in range(0, h, stride):
            input_roi = [x, y, x+patch_size, y+patch_size]
            input_line_list, (tr_x, tr_y) = filter_linestrings_within_bbox_input(input_geojson_file, input_roi,\
                                                              shift_threshold=dim_threshold, switch_xy=False)
            if len(input_line_list) == 0 :
                continue
            
            for ref_ln in referece_lines:
                _matched_linestrings, _matched_length = \
                    filter_linestrings_within_buffer(ref_ln, input_line_list, buffer_distance=10)
                if _matched_linestrings == [] or _matched_length<min(ref_ln.length, PATCH_SIZE)//5:
                    continue
                matched_linestrings, matched_length = filter_noise_lines(_matched_linestrings, ref_ln, angle_threshold=30)
                if matched_linestrings == [] or matched_length<min(ref_ln.length, PATCH_SIZE)//5:# or matched_length<50: #
                    continue
                # matched_linestrings = _matched_linestrings
                matched_object = MultiLineGraph(matched_linestrings)
                matched_object.build_graph()
                matched_paths, _ = matched_object.construct_paths()
                matched_trajs = matched_object.paths_to_linestrings(matched_paths) 
                # final_filtered_lines = matched_trajs
                final_filtered_lines, final_lines_length = further_filter_noise_lines(matched_trajs , \
                                                                                      angle_threshold=30, dist_threshold=10)
                # print(f'final filtering length: {final_lines_length}')
                
#                 if final_filtered_lines == [] or final_lines_length<min(ref_ln.length, PATCH_SIZE)//5:
#                     continue
                input_traj_list.append([normalize_linestring_orientation(l) for l in final_filtered_lines])
                tr_x_list.append(tr_x)
                tr_y_list.append(tr_y)
                patch_x_list.append(x)
                patch_y_list.append(y)
                ref_lines.append(ref_ln)
#                 print(cnt)
                cnt += 1
    save_paths_to_geojson(input_traj_list, tr_x_list, tr_y_list, patch_x_list, patch_y_list, ref_lines, save_input_geojson_path)

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
