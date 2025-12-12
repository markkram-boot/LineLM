from shapely.geometry import LineString, MultiLineString, Point
from shapely.affinity import translate
from shapely.ops import unary_union, linemerge, snap
import shapely
import geopandas as gpd

def offset_geometry(geom, dx=0, dy=0):
    """
    Offset a LineString or MultiLineString by (dx, dy).

    Args:
        geom: LineString or MultiLineString
        dx: shift in x direction
        dy: shift in y direction

    Returns:
        Translated geometry
    """
    if isinstance(geom, (LineString, MultiLineString)):
        return translate(geom, xoff=dx, yoff=dy)
    else:
        raise ValueError("Geometry must be a LineString or MultiLineString")
        
def save_linestrings_to_geojson(lines, output_path, crs=None):
    """
    Save a list of LineStrings to a GeoJSON file.

    Args:
        lines (List[LineString]): List of shapely LineString objects.
        output_path (str): File path to save the GeoJSON.
        crs (str): Coordinate reference system (default: WGS 84).
    """
    gdf = gpd.GeoDataFrame(geometry=lines, crs=crs)
    gdf.to_file(output_path, driver="GeoJSON")
    print(f"Saved {len(lines)} LineStrings to {output_path}")
    

def tolerant_merge_lines(lines, snap_tolerance=0.5):
    """
    Snap each LineString to the nearest other line and merge into continuous LineStrings.

    Args:
        lines (List[LineString]): Input geometries.
        snap_tolerance (float): Distance threshold to close gaps.

    Returns:
        List[LineString]: Merged and cleaned geometries.
    """
    if not lines:
        return []

    # Step 1: For snapping, exclude self from snap base
    snapped_lines = []
    for i, line in enumerate(lines):
        # Build base union from all other lines
        other_lines = lines[:i] + lines[i+1:]

        if not other_lines:
            snapped_lines.append(line)
            continue

        # Find the closest line to snap to
        closest_line = []
        for ln in other_lines:
            dist = line.distance(ln)
            if 0 < dist <= snap_tolerance:
                closest_line.append(ln)

        if closest_line != []:
            # Snap current line to the closest one
            snapped = line
            for ln in closest_line:
                snapped = snap(line, ln, snap_tolerance)
            snapped_lines.append(snapped)
        else:
            snapped_lines.append(line)

    # Step 2: Merge snapped lines into as continuous lines as possible
    merged = linemerge(unary_union(snapped_lines))
    # Step 3: Normalize output as list of LineStrings
    if isinstance(merged, LineString):
        return [merged]
    elif isinstance(merged, MultiLineString):
        return list(merged.geoms)
    else:
        return []
# def tolerant_merge_lines(lines, snap_tolerance=0.2):
#     """
#     Merge overlapping or nearly-connected LineStrings into one or more continuous LineStrings.

#     Args:
#         lines (List[LineString]): Input geometries.
#         snap_tolerance (float): Max distance to snap endpoints.

#     Returns:
#         List[LineString]: Merged line geometries.
#     """
#     if not lines:
#         return []

#     # Step 1: Combine all lines into a single geometry for snapping reference
#     base_union = unary_union(lines)

#     # Step 2: Snap each line to the union to close gaps
#     snapped = [snap(line, base_union, snap_tolerance) for line in lines]

#     # Step 3: Merge all snapped lines
#     merged = linemerge(unary_union(snapped))

#     # Step 4: Normalize to list of LineStrings
#     if isinstance(merged, LineString):
#         return [merged]
#     elif isinstance(merged, MultiLineString):
#         return list(merged.geoms)
#     else:
#         return []
    
# def endpoints_overlap(line1, line2, tolerance=1e-6):
#     start1, end1 = Point(line1.coords[0]), Point(line1.coords[-1])
#     start2, end2 = Point(line2.coords[0]), Point(line2.coords[-1])

#     return (
#         (start1.distance(start2) <= tolerance or
#         start1.distance(end2) <= tolerance) and
#         (end1.distance(start2) <= tolerance or
#         end1.distance(end2) <= tolerance)
#     )

def endpoints_overlap(line1, line2, tolerance=1e-6):
    start2, end2 = Point(line2.coords[0]), Point(line2.coords[-1])

    return start2.distance(line1)<=tolerance and end2.distance(line1)<=tolerance
    
# def remove_redundant_lines(lines, buffer_tolerance=0.5):
#     """
#     Remove LineStrings that are fully within the buffer of another line.
    
#     Args:
#         lines (List[LineString]): List of LineStrings (after merging).
#         buffer_tolerance (float): Buffer radius used to detect redundancy.
    
#     Returns:
#         List[LineString]: Cleaned list with redundant lines removed.
#     """
#     keep_indices = set(range(len(lines)))
#     for i, line_i in enumerate(lines):
#         for j, line_j in enumerate(lines):
#             if i >= j or j not in keep_indices:
#                     continue

#             buffer_j = line_j.buffer(buffer_tolerance)

#             # Check if all points in i are inside j's buffer
#             if line_i.within(line_j.buffer(buffer_tolerance)) and endpoints_overlap(line_i, line_j, buffer_tolerance/2)\
#                 and i in keep_indices:
#                 keep_indices.remove(i)
# #                 # Else if j is shorter and within i's buffer, drop j
# #                 elif all(line_i.buffer(buffer_tolerance).contains(Point(pt)) for pt in line_j.coords):
# #                     if j in keep_indices and line_j.length <= line_i.length:
# #                         keep_indices.remove(j)
        
        
# #         is_redundant = False
# #         for j, other in enumerate(lines):
# #             if i != j:
# #                 if line.within(other.buffer(buffer_tolerance)) and endpoints_overlap(line, other, buffer_tolerance/2):
# #                     is_redundant = True
# #                     break
# #         if not is_redundant:
# #             keep.append(line)
#     return [lines[i] for i in keep_indices]

def remove_redundant_lines(lines, buffer_tolerance=0.5):
    """
    Efficiently remove redundant LineStrings that are fully within the buffer of another line.
    Only one line is kept if two are overlapping.
    
    Args:
        lines (List[LineString]): List of LineStrings.
        buffer_tolerance (float): Buffer radius used to detect redundancy.
    
    Returns:
        List[LineString]: Cleaned list with redundant lines removed.
    """
    keep_indices = set(range(len(lines)))
    buffers = [line.buffer(buffer_tolerance) for line in lines]

    for i in range(len(lines)):
        if i not in keep_indices:
            continue

        line_i = lines[i]
        for j in range(i + 1, len(lines)):
            if j not in keep_indices:
                continue

            line_j = lines[j]

            # Fast filter: check endpoint overlap first
            if not endpoints_overlap(line_i, line_j, buffer_tolerance / 2):
                continue

            # Then check containment
            if line_i.within(buffers[j]):
                keep_indices.discard(i)
                break
            elif line_j.within(buffers[i]):
                keep_indices.discard(j)

    return [lines[i] for i in keep_indices]