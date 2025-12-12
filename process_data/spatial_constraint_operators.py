from shapely.geometry import MultiLineString, LineString, GeometryCollection
from shapely.ops import unary_union
from shapely.affinity import scale
from scipy.spatial.distance import directed_hausdorff
import numpy as np
import math

def convert_list_to_mls(line_list):
    return MultiLineString([LineString(i) for i in line_list])

def jaccard_index(_mls1, _mls2, buffer_dist=15):
    """Compute the Jaccard index based on buffered intersection."""
    mls1 = convert_list_to_mls(_mls1)
    mls2 = convert_list_to_mls(_mls2)
    
    buffered1 = unary_union(mls1).buffer(buffer_dist)
    buffered2 = unary_union(mls2).buffer(buffer_dist)
    intersection = buffered1.intersection(buffered2).area
    union = buffered1.union(buffered2).area
    return intersection / union if union > 0 else 1.0


def largest_orientation_change(_mls):
    """Calculate the largest orientation change in a MultiLineString."""
    mls = convert_list_to_mls(_mls)
    
    max_change = 0
    for line in mls.geoms:
        coords = list(line.coords)
        angles = []
        for i in range(len(coords) - 1):
            dx = abs(coords[i + 1][0] - coords[i][0])
            dy = abs(coords[i + 1][1] - coords[i][1])
            angle = math.atan2(dy, dx)
            angles.append(angle)
        
        for j in range(len(angles) - 1):
            change = abs(angles[j + 1] - angles[j])
#             change = min(change, 2 * math.pi - change)  # Account for circular nature of angles
            max_change = max(max_change, change)
    
    return math.degrees(max_change)

def calculate_angle(v1, v2):
    """Calculate the angle in degrees between two vectors."""
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    magnitude_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
    magnitude_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0  # Avoid division by zero if a vector has zero length
    cosine_angle = dot_product / (magnitude_v1 * magnitude_v2)
    clamped_cosine_angle = max(-1.0, min(1.0, cosine_angle))

    angle_radians = math.acos(clamped_cosine_angle)
    return math.degrees(angle_radians)

def has_sharp_turns(multilines, threshold_angle=30):
    """Check if a LineString has any sharp turns based on a threshold angle."""
    mls = convert_list_to_mls(multilines)
    
    for line in mls.geoms:
        coords = list(line.coords)
        for i in range(1, len(coords) - 1):
#             v1 = (coords[i][0] - coords[i-1][0], coords[i][1] - coords[i-1][1])
#             v2 = (coords[i+1][0] - coords[i][0], coords[i+1][1] - coords[i][1])
            v1 = (coords[i-1][0] - coords[i][0], coords[i-1][1] - coords[i][1])
            v2 = (coords[i+1][0] - coords[i][0], coords[i+1][1] - coords[i][1])
            angle = calculate_angle(v1, v2)
            if angle < threshold_angle:
                return True, angle  # Return True and the angle if sharp turn is found
    return False, None  # Return False if no sharp turn is found


# def calculate_curvature(p1, p2, p3):
#     """
#     Calculate the curvature at point p2 given three consecutive points p1, p2, p3.
#     Curvature is measured using the angle between the two segments.
#     """
#     v1 = np.array(p2) - np.array(p1)  # Vector from p1 to p2
#     v2 = np.array(p3) - np.array(p2)  # Vector from p2 to p3

#     norm_v1 = np.linalg.norm(v1)
#     norm_v2 = np.linalg.norm(v2)

#     if norm_v1 == 0 or norm_v2 == 0:  # Avoid division by zero
#         return 0

#     dot_product = np.dot(v1, v2)
#     cross_product = np.cross(v1, v2)

#     angle = np.arctan2(np.linalg.norm(cross_product), dot_product)  # Angle in radians
#     angle_degrees = np.degrees(angle)  # Convert to degrees

#     return angle_degrees  # Curvature as absolute angle in degrees
def angle_between_lines(p1, p2, p3):
    """
    Compute angle between p1→p2 and p2→p3 at vertex p2, in degrees.
    """
    # Vector from p2 to p1
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    # Vector from p2 to p3
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    
    # Dot product
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = math.hypot(*v1)
    mag2 = math.hypot(*v2)

    if mag1 == 0 or mag2 == 0:
        return 0.0

    # Clamp to avoid domain error
    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    angle_rad = math.acos(cos_angle)
    return math.degrees(angle_rad)

def distance_ratio(p1, p2, p3):
    """
    Compute the ratio of distance(p1, p3) / (distance(p1, p2) + distance(p2, p3)).
    
    Parameters:
    - p1, p2, p3: Tuples or lists representing (x, y) coordinates.

    Returns:
    - Ratio as a float.
    """
    # Convert to numpy arrays
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)

    # Compute distances
    dist_p1_p2 = np.linalg.norm(p2 - p1)
    dist_p2_p3 = np.linalg.norm(p3 - p2)
    dist_p1_p3 = np.linalg.norm(p3 - p1)

    # Compute ratio
#     ratio = dist_p1_p3 / (dist_p1_p2 + dist_p2_p3) if dist_p1_p2+dist_p2_p3!=0 else 0
    ratio = angle_between_lines(p1, p2, p3)

    return ratio, max(dist_p1_p2, dist_p2_p3)

def max_curvature_segment_length(lines):
    """
    Computes the maximum curvature from a list of lines.
    """
    if not all(isinstance(ls, list) for ls in lines):
        raise ValueError("Input must be a list of LineString objects.")

    max_curv = float('inf')
    max_seg_len = 0
    for line in lines:
        coords = line

        # Compute curvature for each point with its neighbors
        for i in range(1, len(coords) - 1):
#             curvature = calculate_curvature(coords[i - 1], coords[i], coords[i + 1])
            curvature, seg_len = distance_ratio(coords[i - 1], coords[i], coords[i + 1])
            max_curv = min(max_curv, curvature)
            max_seg_len = max(max_seg_len, seg_len)
    return max_curv, max_seg_len  # Maximum curvature in radians

    
def cal_length_similarity_input_output(A_coords, B_coords, buffer_distance):
    """
    Computes the ratio of the length of A that is within the buffered region of B.

    Parameters:
    - A_coords (List[List[List[float]]]): List of lines representing A (each line as list of [x, y] coordinates).
    - B_coords (List[List[List[float]]]): List of lines representing B (each line as list of [x, y] coordinates).
    - buffer_distance (float): The buffer radius around B.

    Returns:
    - Ratio (float): The proportion of A's length within the buffered area.
    """

    # Convert coordinate lists to MultiLineString objects
    A = MultiLineString([LineString(line) for line in A_coords])
    B = MultiLineString([LineString(line) for line in B_coords])

    # Create buffer around B
    buffer_B = B.buffer(buffer_distance)

    # Compute the intersection of A with the buffered region
    A_within_buffer = A.intersection(buffer_B)

    # Compute lengths
    total_length_A = A.length
    total_length_B = B.length
    intersected_length_A = A_within_buffer.length if A_within_buffer else 0

    # Calculate ratios, avoiding division by zero
    ratio_A = intersected_length_A / total_length_A if total_length_A > 0 else 0
    ratio_B = intersected_length_A / total_length_B if total_length_B > 0 else 0

    return ratio_A, ratio_B

def calculate_line_orientation(line):
    """
    Computes the orientation (angle) of a LineString in degrees.
    The angle is computed from the first to the last coordinate and normalized to [0, 180].
    
    :param line: A Shapely LineString.
    :return: Orientation angle in degrees.
    """
#     x1, y1 = line.coords[0]
#     x2, y2 = line.coords[-1]
#     angle_rad = np.arctan2(y2 - y1, x2 - x1)
#     angle_deg = np.degrees(angle_rad) % 180
#     return angle_deg
    # Get the first and last coordinates
    p0 = line.coords[0]
    p1 = line.coords[-1]
    
    # Sort the endpoints lexicographically to ensure a canonical order.
    start, end = sorted([p0, p1])
    
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle_rad = np.arctan2(dy, dx)
    # Normalize the angle to [0, 180)
    angle_deg = np.degrees(angle_rad) % 180
    return angle_deg

def calculate_average_orientation(lines):
    """
    Computes the weighted average orientation of a list of LineString objects,
    weighting each segment by its length.
    
    :param lines: List of Shapely LineString objects.
    :return: Weighted average orientation (degrees).
    """
    total_length = 0.0
    weighted_sum = 0.0
    angle_sum = 0.0 
    angles = []
    for line in lines:
#         length = line.length
        angle = calculate_line_orientation(line)
        angle_sum += angle
        angles.append(angle)
#         total_length += length
#         weighted_sum += angle * length
#     print(angles)
#     return weighted_sum / total_length if total_length != 0 else 0
    return angle_sum / len(lines) if lines!=[] else 0

def extract_lines(geom):
    """
    Extracts LineString objects from a geometry, which might be a LineString, MultiLineString,
    or GeometryCollection.
    
    :param geom: A Shapely geometry.
    :return: List of LineString objects.
    """
    if geom.is_empty:
        return []
    elif isinstance(geom, LineString):
        return [geom]
    elif isinstance(geom, MultiLineString):
        return list(geom.geoms)
    elif isinstance(geom, GeometryCollection):
        return [g for g in geom if isinstance(g, LineString)]
    else:
        return []

def break_line_by_intersection(main_line, candidate_lines, buffer_distance):
    """
    Splits the main_line (after buffering) into two parts:
      - intersected_lines: parts that intersect the union of candidate_lines.
      - non_intersected_lines: the remainder.
    
    The main_line is first buffered by buffer_distance (e.g., 5 units) to capture near intersections.
    
    :param main_line: The main LineString.
    :param candidate_lines: List of candidate LineString objects.
    :param buffer_distance: Buffer distance to apply to the main line before intersection.
    :return: Tuple (list of intersected LineStrings, list of non-intersected LineStrings)
    """
    # Create the union of all candidate lines for efficient intersection testing.
    union_candidates = unary_union(candidate_lines)
    
    # Extract the list of coordinates from the main line.
    coords = list(main_line.coords)
    
    intersected_segments = []
    non_intersected_segments = []
    
    # Process each segment (pair of consecutive coordinates).
    for i in range(len(coords) - 1):
        segment = LineString([coords[i], coords[i+1]])
        # Buffer the segment by the given distance.
        buffered_segment = segment.buffer(buffer_distance, cap_style=2)
        
        # Test if the buffered segment intersects the candidate union.
        if buffered_segment.intersects(union_candidates):
            intersected_segments.append(segment)
        else:
            non_intersected_segments.append(segment)
    
    return intersected_segments, non_intersected_segments


def cal_orientation_similarity_input_output(A_coords, B_coords, buffer_distance, diff_orientation_threshold=30):
    # Convert coordinate lists to MultiLineString objects
    main_line = LineString(A_coords[0])
    candidate_lines = [LineString(line) for line in B_coords]
    
    # Break the main line into intersected and non-intersected segments.
    intersected_lines, non_intersected_lines = break_line_by_intersection(main_line, candidate_lines, buffer_distance)

    if non_intersected_lines == []:
        return True
    elif intersected_lines == []:
        return False
    # Compute average orientations (weighted by segment lengths) for each set.
    avg_orientation_intersected = calculate_average_orientation(intersected_lines)
    avg_orientation_non_intersected = calculate_average_orientation(non_intersected_lines)
   
    diff = abs(avg_orientation_intersected - avg_orientation_non_intersected)
    orientation_difference = min(diff, 180-diff)
#     print(avg_orientation_intersected, avg_orientation_non_intersected, orientation_difference)
    return False if orientation_difference > diff_orientation_threshold else True
    
