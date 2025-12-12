from shapely.geometry import LineString, MultiLineString, Point, GeometryCollection
from shapely.ops import unary_union, linemerge, split, snap, substring
import networkx as nx
import os
import json
import numpy as np
import cv2
from process_data.multiline_to_line_postprocess import MultiLineGraph
from utils import draw_multilinestring_on_image
from collections import Counter, defaultdict

def load_multilinestring_from_geojson(geojson_file):
    """
    Load MultiLineString geometries from a GeoJSON file.

    Parameters:
    - geojson_file: Path to the GeoJSON file.

    Returns:
    - A Shapely MultiLineString object if geometries are found; otherwise, None.
    """
    with open(geojson_file, 'r') as f:
        geojson_data = json.load(f)

    line_dict = {}
    # Process each feature in the GeoJSON
    for feature in geojson_data['features']:
        geometry = feature['geometry']
        row, col = feature['properties']['patch_x'], feature['properties']['patch_y'] 
        line_list = []
        for ln in geometry:
            line_list.append(LineString(ln))
        line_dict[(row,col)] = line_list
    return line_dict

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
        for ln in line_list:
            feature = {
                "type": "Feature",
                "geometry": ln,
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
    
def save_line_groups_for_patch_to_geojson(line_patch_groups, output_path):
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

def group_and_select_longest(_lines, buffer_distance, overlap_threshold=0.9):
    """
    Groups LineStrings into clusters such that for any pair of lines in a group, 
    the ratio of the intersected length (using buffering) to their total length is above the threshold.
    Then, selects and returns the longest LineString from each cluster.

    Parameters:
      - lines (list of LineString): List of LineString objects.
      - buffer_distance (float): Buffer distance applied to each line when computing overlaps.
      - overlap_threshold (float, default=0.9): The minimum overlap ratio required for two lines 
        to be considered in the same group.

    Returns:
      - list of LineString: The longest LineString from each cluster.
    """
    lines = []
    for line in _lines:
        if isinstance(line, LineString):
            lines.append(line)
        elif isinstance(line, list) and all(isinstance(coord, (list, tuple)) and len(coord) == 2 for coord in line):
            lines.append(LineString(line))
        else:
            raise ValueError("Invalid input: elements must be LineString objects or lists of (x, y) coordinate pairs.")

    
    n = len(lines)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    # Build graph based on overlap ratio.
    for i in range(n):
        for j in range(i + 1, n):
            # Create buffered versions of the lines with a flat cap style.
            buffered_i = lines[i].buffer(buffer_distance, cap_style=2)
            buffered_j = lines[j].buffer(buffer_distance, cap_style=2)
            
            # Calculate overlap ratios in both directions.
            ratio_i = lines[j].intersection(buffered_i).length / lines[j].length if lines[j].length > 0 else 0
            ratio_j = lines[i].intersection(buffered_j).length / lines[i].length if lines[i].length > 0 else 0
            
            # If either overlap ratio meets or exceeds the threshold, add an edge.
            if ratio_i >= overlap_threshold or ratio_j >= overlap_threshold:
                G.add_edge(i, j)
    
    # Find maximal cliques; each clique represents a group where every pair meets the threshold.
    cliques = list(nx.find_cliques(G))
    groups = [[lines[i] for i in clique] for clique in cliques]
    
    # For each group, select the longest LineString.
    longest_lines = [max(group, key=lambda line: line.length) for group in groups]
    return longest_lines

def build_graph(line_strings):
    """Builds an undirected graph from LineStrings where nodes are shared endpoints."""
    G = nx.Graph()
    for i, line in enumerate(line_strings):
        G.add_node(i, line=line)
        for j in range(i + 1, len(line_strings)):
            line2 = line_strings[j]
            # Check if they share any endpoint
            endpoints1 = [line.coords[0], line.coords[-1]]
            endpoints2 = [line2.coords[0], line2.coords[-1]]
            if set(endpoints1) & set(endpoints2):
                G.add_edge(i, j)
    return G

def is_loop_component(component_lines):
    """Determines if a group of LineStrings form a loop based on endpoint degrees."""
    endpoints = []
    for line in component_lines:
        endpoints.append(line.coords[0])
        endpoints.append(line.coords[-1])
    
    counts = Counter(endpoints)
    
    # In a perfect loop, every endpoint appears exactly twice
    return all(count % 2 == 0 for count in counts.values())

def remove_loops(line_strings):
    """Removes line strings that collectively form closed loops."""
    G = build_graph(line_strings)
    components = list(nx.connected_components(G))
    non_loop_lines = []

    for component in components:
        lines = [line_strings[idx] for idx in component]
        if not is_loop_component(lines):
            non_loop_lines.extend(lines)

    return non_loop_lines

def smooth_line(line, smoothing_factor=1.0):
    """
    Reduces zigzags in a LineString by applying Douglas–Peucker simplification
    and then removing duplicate or near-duplicate nodes. This avoids using spline
    smoothing while still producing a smoother line.
    
    Args:
        line (LineString): The input LineString.
        tolerance (float): Tolerance for the simplify() operation.
        
    Returns:
        LineString: A simplified and cleaned version of the input line.
    """
    # Simplify the line using Douglas-Peucker algorithm
    simplified = line.simplify(smoothing_factor, preserve_topology=True)
    return simplified

def split_line(line, n_parts):
    """
    Splits a LineString into n equal-length parts.

    Parameters:
    - line: A Shapely LineString.
    - n_parts: Number of segments to split into.

    Returns:
    - List of LineStrings.
    """
    if n_parts < 1:
        raise ValueError("Number of parts must be at least 1.")

    total_length = line.length
    segment_length = total_length / n_parts

    segments = []

    for i in range(n_parts):
        start = i * segment_length
        end = (i + 1) * segment_length
        segment = substring(line, start, end)
        segments.append(segment)

    return segments

def split_simplify_and_connect(line, simplify_tolerance=2.0):
    """
    Splits a long LineString into sub-LineStrings of about segment_length,
    simplifies each segment using the Douglas–Peucker algorithm (simplify),
    removes duplicate nodes, and then connects them into one continuous LineString.
    
    Args:
        line (LineString): Input long LineString.
        segment_length (float): Approximate length for each sub-segment.
        simplify_tolerance (float): Tolerance used in simplify() to remove minor deviations.
        precision (int): Decimal precision for removing duplicate nodes.
    
    Returns:
        LineString: The final, continuous, smoothed LineString.
    """
    total_length = line.length
    segments = split_line(line, n_parts=max(int(total_length//30), 1))
    
    # Concatenate segments while connecting endpoints.
    merged_coords = []
    for seg in segments:
        # Remove duplicate nodes in each segment.
        seg = smooth_line(seg, smoothing_factor=simplify_tolerance)
        seg_coords = list(seg.coords)
        if merged_coords:
            # If the last coordinate doesn't equal the first coordinate of seg,
            # add the first coordinate (to connect them).
            if merged_coords[-1] != seg_coords[0]:
                merged_coords.append(seg_coords[0])
            merged_coords.extend(seg_coords[1:])
        else:
            merged_coords.extend(seg_coords)
    return LineString(merged_coords)

def merge_overlapping_lines(lines, snap_tolerance=1.0):
    """
    Merges a list of LineStrings that are overlapped or nearly overlapped.
    Nearby endpoints (within snap_tolerance) are snapped together to close small gaps,
    and the lines are merged using unary_union and linemerge.
    
    Args:
        lines (List[LineString]): Input list of LineString geometries.
        snap_tolerance (float): Maximum distance to snap endpoints.

    Returns:
        List[LineString]: A list of merged, continuous LineString objects.
    """
    if not lines:
        return []
    if len(lines) == 1:
        return lines

    # Combine all lines into a single geometry (used as a snapping base)
    unioned = unary_union(lines)
    
    # Snap each line to the unioned geometry to correct small misalignments
    snapped_lines = [snap(line, unioned, snap_tolerance) for line in lines]
    
    # Union all snapped lines to dissolve overlaps/gaps
    merged_geometry = unary_union(snapped_lines)
    
    # Merge contiguous line segments into continuous LineStrings
    merged_result = linemerge(merged_geometry)
    
    # Return a list of LineStrings
    if merged_result.is_empty:
        return []
    elif merged_result.geom_type == "LineString":
        return [merged_result]
    elif merged_result.geom_type == "MultiLineString":
        return list(merged_result.geoms)
    else:
        return []
    
def can_merge(line1, line2):
    """ Check if two lines can be merged by comparing their endpoints. """
    return line1.coords[-1] == line2.coords[0] or line1.coords[0] == line2.coords[-1]

def merge_lines(lines):
    merged = True
    while merged:
        merged = False
        i = 0
        while i < len(lines):
            j = i + 1
            while j < len(lines):
                if can_merge(lines[i], lines[j]):
                    # Merge lines[i] and lines[j]
                    new_line = LineString([*lines[i].coords, *lines[j].coords])
                    lines[i] = new_line  # Replace the line at index i with the merged line
                    del lines[j]  # Remove the old line at index j
                    merged = True  # Set flag to continue the loop
                    break  # Exit the inner loop to restart checking
                j += 1
            if merged:
                break  # Restart from the first line if any merge occurred
            i += 1
    return lines

def keep_half_linestring(line):
    """
    If the input LineString is a loop (starts and ends at the same point), keep the first half.
    Otherwise, return the original line.
    """
    coords = list(line.coords)
    
    # Check if it's a loop (start == end)
    is_loop = coords[0] == coords[-1]

    if is_loop:
        # Remove closing point to avoid duplication
        coords = coords[:-1]
        # Keep the first half (+1 for continuity)
        half = len(coords) // 2 + 1
        new_line = LineString(coords[:half])
        return new_line
    else:
        return line  # not a loop, return unchanged (or handle differently if needed)

def remove_out_of_range_points(line, min_val=0, max_val=500):
    """
    Rounds out-of-bound x or y values to min_val or max_val for each point in the LineString.
    Returns a new LineString or None if fewer than 2 points remain.
    """
    clipped_coords = []
    for x, y in line.coords:
        x_clipped = min(max(x, min_val), max_val)
        y_clipped = min(max(y, min_val), max_val)
        clipped_coords.append((x_clipped, y_clipped))

    # Remove duplicate consecutive points (optional but often helpful)
    unique_coords = [clipped_coords[0]]
    for pt in clipped_coords[1:]:
        if pt != unique_coords[-1]:
            unique_coords.append(pt)

    if len(unique_coords) < 2:
        return None  # Not enough distinct points to form a valid LineString
    return LineString(unique_coords)

def share_endpoint(line1, line2):
    """Returns True if two LineStrings share an endpoint."""
    endpts1 = [line1.coords[0], line1.coords[-1]]
    endpts2 = [line2.coords[0], line2.coords[-1]]
    return any(p1 == p2 for p1 in endpts1 for p2 in endpts2)

def filter_short_single_components(line_strings, length_threshold):
    """
    Remove single-LineString components whose length is below the threshold.

    Args:
        line_strings (List[LineString]): List of LineStrings.
        length_threshold (float): Minimum length to keep if the line is isolated.

    Returns:
        List[LineString]: Filtered list of LineStrings.
    """
    # Step 1: Build graph
    G = nx.Graph()
    for i, line in enumerate(line_strings):
        G.add_node(i)

    for i in range(len(line_strings)):
        for j in range(i + 1, len(line_strings)):
            if share_endpoint(line_strings[i], line_strings[j]):
                G.add_edge(i, j)

    # Step 2: Analyze components
    keep_indices = set()
    for component in nx.connected_components(G):
        if len(component) > 1:
            keep_indices.update(component)
        else:
            idx = list(component)[0]
            if line_strings[idx].length >= length_threshold:
                keep_indices.add(idx)

    # Step 3: Return filtered lines
    return [line_strings[i] for i in sorted(keep_indices)]

def merge_and_smooth_lines(lines, snap_tolerance=1.0, smoothing_factor=1.0, small_line_thres=20):
    """
    Merges a list of overlapping or nearly overlapping LineStrings, and then smooths each merged LineString.

    Args:
        lines (List[LineString]): Input lines.
        snap_tolerance (float): Tolerance for snapping endpoints.
        smoothing_factor (float): Smoothing factor for spline interpolation.

    Returns:
        List[LineString]: A list of merged and smoothed LineStrings.
    """
#     merged_lines = merge_overlapping_lines(lines, snap_tolerance)
#     merged_lines = merge_lines(merged_lines)

    merged_lines = merge_overlapping_lines(lines, snap_tolerance)
    merged_lines = [keep_half_linestring(i) for i in merged_lines]
    smoothed_lines = [split_simplify_and_connect(ml, simplify_tolerance=smoothing_factor) \
                      for ml in merged_lines]
    # Step 4: clip out-of-range points (x,y <0 or >=499)
    final_lines = []
    for ln in smoothed_lines:
        filtered_ln = remove_out_of_range_points(ln, 0, 499)
        if filtered_ln and filtered_ln.length > 0:
            final_lines.append(filtered_ln)

    final_lines_no_loops = remove_loops(final_lines)
    final_lines_no_loops_short = filter_short_single_components(final_lines_no_loops, small_line_thres)
    return final_lines_no_loops_short
