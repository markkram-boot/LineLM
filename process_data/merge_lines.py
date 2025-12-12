from shapely.geometry import LineString, Point, MultiLineString
from shapely.ops import linemerge, substring, snap, unary_union
from shapely.geometry import CAP_STYLE
import networkx as nx
import math

class LineConflationGraph:
    def __init__(self, buffer_distance=1e-6, max_orientation_diff_deg=10):
        """
        Initialize the graph and set buffer distance for soft overlap detection.
        
        :param buffer_distance: Distance used to buffer existing lines for fuzzy overlap detection.
        """
        self.graph = nx.Graph()
        self.line_id_counter = 0
        self.buffer_distance = buffer_distance
        self.max_orientation_diff_rad = max_orientation_diff_deg

    def _add_line(self, line):
        """
        Add a LineString to the graph with a unique ID.
        """
        if line.length > 0:
            self.graph.add_node(self.line_id_counter, geometry=line)
            self.line_id_counter += 1

    def _get_overlap_length(self, line1, line2_buffered):
        """
        Compute the length of overlap between line1 and the buffered geometry of another line.
        """
        intersection = line1.intersection(line2_buffered)
        if intersection.is_empty:
            return 0
        if intersection.geom_type == 'LineString':
            return intersection.length
        elif intersection.geom_type == 'MultiLineString':
            return sum(part.length for part in intersection.geoms)
        return 0
    
    def _connect_to_overlapped_nodes(self, remaining_line, overlapped_line):
        if remaining_line.is_empty or overlapped_line.is_empty:
            return remaining_line

        endpoints = [Point(remaining_line.coords[0]), Point(remaining_line.coords[-1])]

        if isinstance(overlapped_line, MultiLineString):
            coords = []
            for part in overlapped_line.geoms:
                coords.extend(part.coords)
        else:
            coords = overlapped_line.coords

        nodes = [Point(c) for c in coords]

        closest = None
        min_dist = float("inf")
        for ep in endpoints:
            for node in nodes:
                dist = ep.distance(node)
                if dist < min_dist:
                    min_dist = dist
                    closest = (ep, node)

        if closest:
            bridge = LineString([closest[0], closest[1]])
            return linemerge([remaining_line, bridge])
        else:
            return remaining_line

    def _find_responsible_line(self, fragment, candidates):
        if not candidates:
            return LineString()
        frag_center = fragment.centroid
        return min(candidates, key=lambda l: l.distance(frag_center))
    
    def _is_orientation_similar(self, line1, line2):
        """
        Returns True if the orientation of `line1` and `line2` is within the threshold.
        Orientation is measured as the angle of the line from start to end.
        """
        def get_orientation(line):
            coords = list(line.coords)
            dx = coords[-1][0] - coords[0][0]
            dy = coords[-1][1] - coords[0][1]
            return math.atan2(dy, dx)

        angle1 = get_orientation(line1)
        angle2 = get_orientation(line2)

        diff = abs(angle1 - angle2)
        diff = min(diff, 2 * math.pi - diff)  # ensure shortest angular distance

        return diff <= self.max_orientation_diff_rad
    
    def _endpoints_match(self, line1, line2, tolerance=0):
        a1, a2 = Point(line1.coords[0]), Point(line1.coords[-1])
        b1, b2 = Point(line2.coords[0]), Point(line2.coords[-1])
        return (
            a1.distance(b1) <= tolerance or a2.distance(b2) <= tolerance or
            a1.distance(b2) <= tolerance or a2.distance(b1) <= tolerance
        )
    
    def snap_endpoints(self, line, target, tolerance):
        """
        Snap only the endpoints of a line to the nearest point on the target geometry if within tolerance.
        """
        coords = list(line.coords)

        # Snap start
        start = Point(coords[0])
        nearest_start = target.interpolate(target.project(start))
        if start.distance(nearest_start) <= tolerance and not start.equals(nearest_start):
            coords.insert(0, (nearest_start.x, nearest_start.y))

        # Snap end
        end = Point(coords[-1])
        nearest_end = target.interpolate(target.project(end))
        if end.distance(nearest_end) <= tolerance and not end.equals(nearest_end):
            coords.append((nearest_end.x, nearest_end.y))

        return LineString(coords)

    def connect_lines(self, new_line):
        """
        connect the new line to existing one,
        if they share the same endpoint
        """
        existing_lines = [(n, data['geometry']) for n, data in self.graph.nodes(data=True)]

        # Rule 1: Skip if fully contained in any existing line
        for _, existing_line in existing_lines:
            if existing_line.contains(new_line):
                return
    
        added_node_ids = []
        merged_flag = False
    
        # Rule 2: Merge with existing line if they share endpoints
        for node_id, existing_line in existing_lines:
            if self._endpoints_match(new_line, existing_line, tolerance=self.buffer_distance):
                merged = linemerge([existing_line, new_line])
                if isinstance(merged, LineString):
                    self.graph.nodes[node_id]["geometry"] = merged
                    merged_flag = True
                    break  # Only merge with one line
    
        # Rule 3: If no merge happened, add the new line as a new node
        if not merged_flag:
            self._add_line(new_line)
            added_node_ids.append(self.line_id_counter - 1)
    
        # Rule 4: Add edges between new lines and those sharing endpoints
        for new_node_id in added_node_ids:
            new_geom = self.graph.nodes[new_node_id]["geometry"]
            for existing_node_id, existing_geom in existing_lines:
                if self._endpoints_match(new_geom, existing_geom, tolerance=self.buffer_distance):
                    self.graph.add_edge(new_node_id, existing_node_id)


    def update_with_line(self, new_line):
        existing_lines = [(n, data['geometry']) for n, data in self.graph.nodes(data=True)]

        # Rule 1: Skip if fully contained
        for _, existing_line in existing_lines:
            buffered = existing_line.buffer(self.buffer_distance, cap_style=CAP_STYLE.flat)
            if buffered.contains(new_line):
                return

        # Rule 2: Track overlaps
        overlap_map = {}  # node_id -> { "line": LineString, "parts": [LineString, ...] }
        for node_id, existing_line in existing_lines:
            buffered = existing_line.buffer(self.buffer_distance, cap_style=CAP_STYLE.flat)
            intersection = new_line.intersection(buffered)
            parts = []
            if not intersection.is_empty:
                if intersection.geom_type == 'LineString':
                    parts.append(intersection)
                elif intersection.geom_type == 'MultiLineString':
                    parts.extend(intersection.geoms)
            if parts:
                overlap_map[node_id] = {"line": existing_line, "parts": parts}

        # Subtract overlapping parts
        remaining = new_line
        for data in overlap_map.values():
            for part in data["parts"]:
                remaining = remaining.difference(part)
                if remaining.is_empty:
                    return

        # Rule 3: Add remaining + snap to overlapping lines only
        added_node_ids = []
        if not remaining.is_empty:
            parts = [remaining] if isinstance(remaining, LineString) else list(remaining.geoms)

            for part in parts:
                snapped = part

                if overlap_map:
                    snap_base = unary_union([v["line"] for v in overlap_map.values()])
#                     snapped = snap(part, snap_base, self.buffer_distance*0.5)
                    snapped = self.snap_endpoints(part, snap_base, self.buffer_distance)
                # Check if snapped line can be merged into any overlapping line
                merged_flag = False
                merged_line = LineString()
                for node_id, overlap_data in overlap_map.items():
                    current_line = self.graph.nodes[node_id]["geometry"]
                    if self._endpoints_match(snapped, current_line, 0):
                        merged = linemerge([overlap_data["line"], snapped])
#                         unioned = unary_union([current_line, snapped])
#                         merged = linemerge(unioned)
                        if isinstance(merged, LineString) and merged.covers(current_line) and merged.covers(snapped):
                            self.graph.nodes[node_id]["geometry"] = merged
                            merged_flag = True
                            merged_line = merged
                            break  # only merge into one:
                if not merged_flag:
                    self._add_line(snapped)
                    added_node_ids.append(self.line_id_counter - 1)
                # Compute the unused portion of the snapped line (difference from merged line)
#                 leftover = snapped.difference(merged_line)
#                 if not leftover.is_empty:
#                     if isinstance(leftover, LineString):
#                         if self.graph.number_of_nodes() != self.line_id_counter:
#                             print("node counter: ", self.graph.number_of_nodes(), self.line_id_counter)
#                         added_node_ids.append(self.line_id_counter)
#                         self._add_line(leftover)
#                     elif isinstance(leftover, MultiLineString):
#                         for piece in leftover.geoms:
#                             if self.graph.number_of_nodes()-1 != self.line_id_counter:
#                                 print("node counter: ", self.graph.number_of_nodes(), self.line_id_counter)
#                             added_node_ids.append(self.line_id_counter)
#                             self._add_line(piece)
  
        # Step 4: Add edges
        for new_node_id in added_node_ids:
            new_geom = self.graph.nodes[new_node_id]["geometry"]
            for existing_node_id in overlap_map.keys():  # only connect to overlapping lines
                self.graph.add_edge(new_node_id, existing_node_id)
    
    def remove_small_isolated_lines(self, min_length):
        """
        Remove lines from the graph that are shorter than `min_length` and not connected to any others.
        """
        to_remove = []
        for node_id, data in self.graph.nodes(data=True):
            geom = data["geometry"]
            if geom.length < min_length and self.graph.degree[node_id] == 0:
                to_remove.append(node_id)
            elif geom.length < min_length//3:
                to_remove.append(node_id)

        self.graph.remove_nodes_from(to_remove)
    
    def get_all_lines(self):
        """
        Return all LineStrings currently in the graph.
        """
        return [data['geometry'] for _, data in self.graph.nodes(data=True)]