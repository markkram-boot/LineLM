import math
from shapely.geometry import LineString, box

def rotate_point(x, y, cx, cy, angle_radians):
    """
    Rotate a point (x, y) around center (cx, cy) by a given angle in radians.
    """
    x_rot = cx + (x - cx) * math.cos(angle_radians) - (y - cy) * math.sin(angle_radians)
    y_rot = cy + (x - cx) * math.sin(angle_radians) + (y - cy) * math.cos(angle_radians)
    return x_rot, y_rot

def extend_line_to_bounds(line, bounds):
    """
    Ensure a line fully reaches the edges of a bounding box.
    
    :param line: A Shapely LineString to be extended.
    :param bounds: ((x_min, y_min), (x_max, y_max)) Bounding box.
    :return: Fully extended LineString.
    """
    x_min, y_min, x_max, y_max = bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]
    
    # Get start and end points of the line
    x1, y1 = line.coords[0]
    x2, y2 = line.coords[1]
    
    bbox = box(x_min, y_min, x_max, y_max)  # Create bounding box
    if not line.intersects(bbox):
        return None, None  # If line is completely outside, return None

    # Compute line equation y = mx + c
    if x1 == x2:  # Vertical line case
#         extended_line = LineString([(x1, y_min), (x1, y_max)])
        return (x1, y_min), (x1, y_max)
    elif y1 == y2:  # Horizontal line case
#         extended_line = LineString([(x_min, y1), (x_max, y1)])
        return (x_min, y1), (x_max, y1)
    else:
        m = (y2 - y1) / (x2 - x1)
        c = y1 - m * x1
        
        # Compute intersections with bounding box
        x_left, y_left = x_min, m * x_min + c
        x_right, y_right = x_max, m * x_max + c
        y_top, x_top = y_max, (y_max - c) / m
        y_bottom, x_bottom = y_min, (y_min - c) / m

        # Collect valid points inside the bounding box
        points = []
        if y_min <= y_left <= y_max:
            points.append((x_left, y_left))
        if y_min <= y_right <= y_max:
            points.append((x_right, y_right))
        if x_min <= x_top <= x_max:
            points.append((x_top, y_top))
        if x_min <= x_bottom <= x_max:
            points.append((x_bottom, y_bottom))

        # Ensure exactly two points for the extended line
        if len(points) >= 2:
            extended_line = LineString([points[0], points[1]])
        else:
            return None, None  # Should not happen, but for safety

#     return extended_line
    return points[0], points[1]

def transform_and_extend_line(start, end, angle_degrees, dx, dy, bounds=((0,0), (500,500))):
    """
    Rotates a line around (250,250), translates it, and ensures it extends fully to the bounding box.

    :param start: Tuple (x1, y1) start node.
    :param end: Tuple (x2, y2) end node.
    :param angle_degrees: Rotation angle in degrees.
    :param dx: Translation in x-direction.
    :param dy: Translation in y-direction.
    :param bounds: Bounding box ((x_min, y_min), (x_max, y_max)).
    :return: Fully extended transformed LineString.
    """
#     cx, cy = 250, 250  # Center of rotation
#     angle_radians = math.radians(angle_degrees)

#     # Step 1: Rotate around center (250, 250)
#     start_rotated = rotate_point(start[0], start[1], cx, cy, angle_radians)
#     end_rotated = rotate_point(end[0], end[1], cx, cy, angle_radians)
#     print(f"start_rotated = {start_rotated}")
#     print(f"end_rotated = {end_rotated}")
#     # Step 2: Translate by (dx, dy)
#     start_translated = (start_rotated[0] + dx, start_rotated[1] + dy)
#     end_translated = (end_rotated[0] + dx, end_rotated[1] + dy)

#     # Step 3: Ensure the line extends to the bounding box
#     transformed_line = LineString([start_translated, end_translated])
#     print("transformed_line: ", transformed_line)
#     extended_line = extend_line_to_bounds(transformed_line, bounds)

    cx, cy = 250, 250  # Center of rotation
    angle_radians = math.radians(angle_degrees)

    # Step 1: Rotate around center (250, 250)
    _start_rotated = rotate_point(start[0], start[1], cx, cy, angle_radians)
    _end_rotated = rotate_point(end[0], end[1], cx, cy, angle_radians)
    start_rotated, end_rotated = extend_line_to_bounds(LineString([_start_rotated, _end_rotated]), bounds)
    
#     print(f"start_rotated = {start_rotated}")
#     print(f"end_rotated = {end_rotated}")
    
    # Step 2: Translate by (dx, dy)
    start_translated = (start_rotated[0] + dx, start_rotated[1] + dy)
    end_translated = (end_rotated[0] + dx, end_rotated[1] + dy)

    # Step 3: Ensure the line extends to the bounding box
    transformed_line = LineString([start_translated, end_translated])
#     print("transformed_line: ", transformed_line)
    extended_start, extended_end = extend_line_to_bounds(transformed_line, bounds)
    
    return LineString([extended_start, extended_end]) if extended_start is not None else None

def get_buffered_transformed_line(start, end, angle_degrees, dx, dy, buffer_distance, bounds=((0,0), (500,500))):
    """
    Returns a buffered LineString after transformation and extension.

    :param start: Tuple (x1, y1) start node.
    :param end: Tuple (x2, y2) end node.
    :param angle_degrees: Rotation angle in degrees.
    :param dx: Translation in x-direction.
    :param dy: Translation in y-direction.
    :param buffer_distance: Buffer distance around the transformed line.
    :param bounds: Bounding box ((x_min, y_min), (x_max, y_max)).
    :return: Buffered Polygon of the transformed and extended line.
    """
    extended_line = transform_and_extend_line(start, end, angle_degrees, dx, dy, bounds)
    
    if extended_line:
        return extended_line.buffer(buffer_distance)  # Create a buffered version
    return None  # Return None if the line is out of bounds

if __name__ == "__main__":
    # Example Usage
    start_node = (250, 0)
    end_node = (250, 500)
    angle = 45  # Rotation in degrees
    dx, dy = 490, 0   # Translation
    buffer_distance = 10  # Buffer size

    transformed_line = transform_and_extend_line(start_node, end_node, angle, dx, dy)
#     buffered_line = get_buffered_transformed_line(start_node, end_node, angle, dx, dy, buffer_distance)

    print("Transformed and Fully Extended Line:", transformed_line)
#     print("Buffered Line:", buffered_line)