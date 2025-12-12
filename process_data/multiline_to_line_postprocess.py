from shapely.geometry import MultiLineString, LineString, Point
import networkx as nx
from collections import defaultdict
import math
from shapely.ops import split, unary_union, snap, linemerge
import itertools
import numpy as np


class MultiLineGraph:
    def __init__(self, multiline: MultiLineString):
        """
        Initialize the graph from a MultiLineString.
        :param multiline: A Shapely MultiLineString object.
        """
        self.linestrings = multiline
        self.graph = nx.Graph()
        self.build_graph()
        self.node_degrees = defaultdict(int)
    
    def build_graph(self, buffer_tolerance=1.0):
        self.graph = nx.Graph()
        seen_nodes = {}   # Mapping from coordinate -> Point
        seen_edges = set()

        for line in self.linestrings:
            coords = list(line.coords)
            for i in range(len(coords) - 1):
                raw_start = Point(coords[i])
                raw_end = Point(coords[i + 1])

                # Check if start is near any seen node
                start = self._get_or_merge_node(raw_start, seen_nodes, buffer_tolerance)
                end = self._get_or_merge_node(raw_end, seen_nodes, buffer_tolerance)

                # Add nodes (if not already in the graph)
                self.graph.add_node(start)
                self.graph.add_node(end)

                # Add edge if not already seen
                edge = tuple(sorted([start, end]))
                if edge not in seen_edges:
                    self.graph.add_edge(start, end, weight=1)
                    seen_edges.add(edge)

        self.node_degrees = dict(self.graph.degree)

#     def build_graph(self, buffer_tolerance=2.0):
#         """
#         Build a graph from linestrings, avoiding edge additions that fall within an existing buffer zone.

#         - If the edge falls fully in buffer → skip
#         - If it partially overlaps → connect only the portion outside to closest node
#         - If no overlap → add the edge directly
#         """
#         self.graph.clear()
#         seen_edges = set()

#         for line in self.linestrings:
#             coords = list(line.coords)
#             for i in range(len(coords) - 1):
#                 p1 = coords[i]
#                 p2 = coords[i + 1]
#                 candidate_edge = LineString([p1, p2])

#                 # Step 1: Construct buffer from current graph edges
#                 current_edges = [LineString([u, v]) for u, v in self.graph.edges]
#                 buffer_zone = unary_union(current_edges).buffer(buffer_tolerance) if current_edges else None

#                 if buffer_zone and candidate_edge.within(buffer_zone):
#                     continue  # Case 1: Fully within buffer → skip

#                 elif buffer_zone and candidate_edge.intersects(buffer_zone):
#                     # Case 2: Partial overlap → try to preserve part outside the buffer
#                     p1, p2 = Point(candidate_edge.coords[0]), Point(candidate_edge.coords[1])
#                      # Step 3: Check which point is inside buffer
#                     if p1.within(buffer_zone) and not p2.within(buffer_zone):
#                         inside, outside = p1, p2
#                     elif p2.within(buffer_zone) and not p1.within(buffer_zone):
#                         inside, outside = p2, p1
#                     # Step 4: Find closest graph node to the inside point
#                     closest_node = min(self.graph.nodes, key=lambda n: Point(n).distance(inside))
#                     final_edge = tuple(sorted([closest_node, (outside.x, outside.y)]))
#                     if final_edge not in seen_edges:
#                         self.graph.add_node(final_edge[0])
#                         self.graph.add_node(final_edge[1])
#                         self.graph.add_edge(*final_edge)
#                         seen_edges.add(final_edge)
# #                     outside = candidate_edge.difference(buffer_zone)

# #                     # If the result is still a LineString (partial), connect it
# #                     if isinstance(outside, LineString):
# #                         ext_coords = list(outside.coords)
# #                         if len(ext_coords) >= 2:
# #                             new_start, new_end = ext_coords[0], ext_coords[-1]

# #                             # Find the closest existing graph node to connect
# #                             graph_nodes = [Point(n) for n in self.graph.nodes]
# #                             if not graph_nodes:
# #                                 continue  # nothing to connect to

# #                             edge_to = new_start if Point(new_start).distance(Point(new_end)) > 0 else new_end
# #                             nearest = min(self.graph.nodes, key=lambda n: Point(n).distance(Point(edge_to)))

# #                             final_edge = tuple(sorted([nearest, new_end])) if Point(nearest).distance(Point(new_end)) > Point(nearest).distance(Point(new_start)) else tuple(sorted([nearest, new_start]))

# #                             if final_edge not in seen_edges:
# #                                 self.graph.add_node(final_edge[0])
# #                                 self.graph.add_node(final_edge[1])
# #                                 self.graph.add_edge(*final_edge)
# #                                 seen_edges.add(final_edge)

#                     # If `outside` is MultiLineString, handle similarly (can add that if needed)
#                     continue

#                 else:
#                     # Case 3: Completely outside buffer → add as is
#                     edge = tuple(sorted([p1, p2]))
#                     if edge not in seen_edges:
#                         self.graph.add_node(p1)
#                         self.graph.add_node(p2)
#                         self.graph.add_edge(p1, p2)
#                         seen_edges.add(edge)

#         self.node_degrees = dict(self.graph.degree)

    def _get_or_merge_node(self, point, seen_nodes, buffer_tolerance):
        """
        If `point` is within buffer_tolerance of any seen node, return that existing node.
        Otherwise, add it as a new node.
        """
        for existing in seen_nodes:
            if point.distance(seen_nodes[existing]) <= buffer_tolerance:
                return existing  # Return merged (existing) node key

        # Not close to any seen node — add new one
        key = tuple(point.coords[0])
        seen_nodes[key] = point
        return key
    
    def can_use_node(self, node, node_usage):
        """
        Check if a node can be used based on its degree and current usage.
        :param node: The node to check.
        :param node_usage: Dictionary tracking the usage of each node.
        :return: True if the node can still be used, False otherwise.
        """
        return node_usage[node] < 1
#         if self.node_degrees[node] > 1:
#             return node_usage[node] < 1 #(self.node_degrees[node] - 1)
#         else:
#             return node_usage[node] < 1
    
    def angle_between(self, p1, p2, p3):
        """Computes the angle (in degrees) formed at point `p2` between segments p1→p2 and p2→p3."""
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        magnitude_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
        magnitude_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
        if magnitude_v1 == 0 or magnitude_v2 == 0:
            return 0  # Avoid division by zero if a vector has zero length
        cosine_angle = dot_product / (magnitude_v1 * magnitude_v2)
        clamped_cosine_angle = max(-1.0, min(1.0, cosine_angle))

        angle_radians = math.acos(clamped_cosine_angle)
        angle_degree = math.degrees(angle_radians)

        smoothness = abs(90 - abs(90 - angle_degree))  # range: 0 (smooth), to 90 (sharpest)
        return smoothness

    
    def construct_paths(self, angle_tolerance=90):
        """
        Extracts long, smooth paths from the graph.
        Each edge is used at most once.

        Args:
            angle_tolerance: Max angle (degrees) allowed between steps for "smooth" turning.

        Returns:
            List of LineString objects, one per smooth path.
        """
        used_edges = set()
        paths = []

        def get_next_node(path):
            current = path[-1]
            prev = path[-2] if len(path) > 1 else None
            candidates = []
            for neighbor in self.graph.neighbors(current):
                edge = tuple(sorted((current, neighbor)))
                if edge in used_edges:
                    continue
                if prev:
                    turn_angle = self.angle_between(prev, current, neighbor)
                    if turn_angle <= angle_tolerance:
                        candidates.append((turn_angle, neighbor))
                else:
                    candidates.append((0, neighbor))  # no previous → allow any direction
            # Sort by smoothest (smallest turn angle)
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1] if candidates else None

        def walk_path(start, neighbor):
            path = [start, neighbor]
            used_edges.add(tuple(sorted((start, neighbor))))
            while True:
                next_node = get_next_node(path)
                if next_node is None:
                    break
                edge = tuple(sorted((path[-1], next_node)))
                used_edges.add(edge)
                path.append(next_node)
            return path

        # Step 1: Start from all endpoint nodes (degree == 1)
        endpoint_nodes = [node for node, deg in self.graph.degree if deg == 1]
        for start in endpoint_nodes:
            for neighbor in self.graph.neighbors(start):
                edge = tuple(sorted((start, neighbor)))
                if edge in used_edges:
                    continue
                path = walk_path(start, neighbor)
#                 if len(path) <= 5:
#                     continue
                paths.append(LineString(path))

        # Step 2: Visit any remaining unused edges (internal loops or disconnected)
        for u, v in self.graph.edges:
            edge = tuple(sorted((u, v)))
            if edge in used_edges:
                continue
            path = walk_path(u, v)
#             if len(path) <= 5:
#                 continue
            paths.append(LineString(path))
        
        return paths
    
    def paths_to_linestrings(self, paths):
        """
        Convert paths (list of nodes) to LineStrings.

        Parameters:
        - paths: List of paths (each path is a list of nodes).

        Returns:
        - List of LineString objects.
        """
        return [LineString(path) for path in paths]
    
    def graph_edges_to_linestrings(self):
        """
        Converts all edges in the graph to LineString objects.

        Returns:
            List[LineString] - one LineString per edge
        """
        lines = []
        for u, v in self.graph.edges:
            line = LineString([u, v])
            lines.append(line)
        return lines
    
def break_lines_at_intersections(lines, projection_tolerance=1.0):
    """
    Breaks each LineString in the list at the intersection points (or near-intersections) with others.
    
    Args:
        lines: List of LineString objects
        snap_tolerance: float - distance for snapping endpoints and ensuring intersections

    Returns:
        List of LineString segments after splitting
    """
    segments = []

    for i, base in enumerate(lines):
        # Collect projected points from intersections with other lines
        projected_points = []
        for j, other in enumerate(lines):
            if i == j:
                continue

            # Use all points from the other line
            for pt in other.coords:
                p = Point(pt)
                dist = base.distance(p)
                if dist <= projection_tolerance:
                    proj = base.interpolate(base.project(p))
                    if not any(proj.equals(existing) for existing in projected_points):
                        projected_points.append(proj)

        # Only split if we have valid projected points
        if projected_points:
            splitter = unary_union(projected_points)
            result = split(base, splitter)
            segments.extend(result.geoms)
        else:
            segments.append(base)

    return segments

def extract_segments(lines):
    """
    Break each LineString into (start, end) segments.
    Returns a list of segments, each as a tuple of 2-tuples: ((x1, y1), (x2, y2))
    """
    segments = []
    for line in lines:
        coords = list(line.coords)
        for i in range(len(coords) - 1):
            segment = LineString([coords[i], coords[i+1]])
            segments.append(segment)
    return segments

def snap_and_merge_lines(lines, snap_tolerance=1.0):
    """
    Snap only endpoints of LineStrings that are close together, then merge overlapping lines.
    Returns a list of merged LineStrings without introducing zigzags.

    Args:
        lines (List[LineString])
        snap_tolerance (float): snapping distance for endpoints

    Returns:
        List[LineString]
    """
    if not lines:
        return []

    endpoints = []
    for line in lines:
        if line.is_empty:
            continue
        coords = list(line.coords)
        endpoints.append(Point(coords[0]))
        endpoints.append(Point(coords[-1]))

    # Step 1: Create a mapping of points that should be snapped together
    snapped_points = {}
    for i, p1 in enumerate(endpoints):
        for j, p2 in enumerate(endpoints):
            if i >= j:
                continue
            if p1.distance(p2) <= snap_tolerance:
                midpoint = Point(np.mean([p1.x, p2.x]), np.mean([p1.y, p2.y]))
                snapped_points[p1.wkt] = midpoint
                snapped_points[p2.wkt] = midpoint

    # Step 2: Replace endpoints with snapped versions
    snapped_lines = []
    for line in lines:
        coords = list(line.coords)
        start = Point(coords[0])
        end = Point(coords[-1])

        if start.wkt in snapped_points:
            coords[0] = (snapped_points[start.wkt].x, snapped_points[start.wkt].y)
        if end.wkt in snapped_points:
            coords[-1] = (snapped_points[end.wkt].x, snapped_points[end.wkt].y)

        snapped_lines.append(LineString(coords))

    # Step 3: Merge the snapped lines
    merged = linemerge(unary_union(snapped_lines))

    if isinstance(merged, LineString):
        return [merged]
    elif hasattr(merged, "geoms"):
        return list(merged.geoms)
    else:
        return []