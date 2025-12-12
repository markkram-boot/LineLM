from shapely.geometry import MultiLineString, LineString
import networkx as nx
from collections import defaultdict
import math

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
    
    def build_graph(self):
        seen_nodes = set()  # Track nodes that have already been added
        seen_edges = set()  # Track added edges to avoid duplicates
        
        for line in self.linestrings:
            coords = list(line.coords)
            for i in range(len(coords) - 1):
                start = tuple(coords[i])
                end = tuple(coords[i + 1])
#                 self.graph.add_edge(start, end, weight=1)  # Add weight if needed
                # Add nodes only if they haven't been added before
                if start not in seen_nodes:
                    self.graph.add_node(start)
                    seen_nodes.add(start)

                if end not in seen_nodes:
                    self.graph.add_node(end)
                    seen_nodes.add(end)

                # Ensure edges are added only once (undirected edges should be checked as (min, max))
                edge = tuple(sorted([start, end]))  # Sort to prevent duplication of (A, B) vs (B, A)

                if edge not in seen_edges:
                    self.graph.add_edge(start, end, weight=1)  # Add edge with weight if needed
                    seen_edges.add(edge)
        
        self.node_degrees = dict(self.graph.degree)
    
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
    
    def calculate_orientation(self, point1, point2):
        """
        Calculate the orientation angle between two points.
        :param point1: The starting point as a tuple (x, y).
        :param point2: The ending point as a tuple (x, y).
        :return: Orientation angle in degrees (0 to 360).
        """
        if point2[0] - point1[0] > 0: 
            dx, dy = point2[0] - point1[0], point2[1] - point1[1]
        else:
            dx, dy = point1[0] - point2[0], point1[1] - point2[1]
        angle = math.degrees(math.atan2(dy, dx))
        return angle % 360
    
    def orientation_difference(self, angle1, angle2):
        """
        Calculate the absolute difference between two angles in degrees.

        Parameters:
        - angle1, angle2: Angles in degrees.

        Returns:
        - Absolute difference in degrees, considering circularity.
        """
        diff = abs(angle1 - angle2)
        return min(diff, 360 - diff)  # Account for circular difference

    
    def construct_paths(self, orientation_threshold=180):
        """
        Construct paths from the graph, limiting the usage of nodes
        based on their degree (node degree - 1).
        :return: List of constructed paths as lists of coordinates and the maximum path length.
        """
        node_usage = defaultdict(int)
        visited_edges = set()
        paths = []
        max_length = 0

        # Sort nodes by coordinates (x, y)
        nodes = sorted(self.graph.nodes(), key=lambda p: (p[0], p[1]))

        while nodes:
            # Start with the first unused node
            start_node = nodes.pop(0)
            if not self.can_use_node(start_node, node_usage):
                continue
            path = []
            current_node = start_node
            previous_node = None

            # Traverse the graph to construct a path
            while current_node is not None and self.can_use_node(current_node, node_usage):
                path.append(current_node)
                node_usage[current_node] += 1

                # Find neighbors and calculate orientations relative to the current node
                neighbors = list(self.graph.neighbors(current_node))
                if previous_node is not None:
                    current_orientation = self.calculate_orientation(previous_node, current_node)
                else:
                    current_orientation = None

                # Order neighbors by orientation closeness to the current direction
                if current_orientation is not None:
                    neighbors = sorted(
                        neighbors,
                        key=lambda neighbor: abs(
                            self.calculate_orientation(current_node, neighbor) - current_orientation
                        )
                    )
#                     neighbors = [
#                         neighbor
#                         for neighbor in neighbors
#                         if self.orientation_difference(
#                             self.calculate_orientation(current_node, neighbor), current_orientation
#                         ) <= orientation_threshold
#                     ]
#                     neighbors.sort(
#                         key=lambda neighbor: self.orientation_difference(
#                             self.calculate_orientation(current_node, neighbor), current_orientation
#                         )
#                     )

                # Find the next usable node
                next_node = None
                for neighbor in neighbors:
                    if self.can_use_node(neighbor, node_usage):
                        next_node = neighbor
                        break

                # Move to the next node
                previous_node = current_node
                current_node = next_node

            # Add the constructed path to the list of paths
            if len(path) > 1:
                max_length = max(max_length, len(path))
                paths.append(path)

        return paths, max_length
    
    def paths_to_linestrings(self, paths):
        """
        Convert paths (list of nodes) to LineStrings.

        Parameters:
        - paths: List of paths (each path is a list of nodes).

        Returns:
        - List of LineString objects.
        """
        return [LineString(path) for path in paths]
    
    