import json
from shapely.geometry import shape, mapping, LineString, Point
from shapely.strtree import STRtree
from collections import defaultdict

def load_linestrings_from_geojson(filepath):
    with open(filepath, 'r') as f:
        geojson_data = json.load(f)

    lines = []
    for feature in geojson_data.get("features", []):
        geom = shape(feature["geometry"])
        if geom.geom_type == "LineString":
            lines.append(geom)
        elif geom.geom_type == "MultiLineString":
            lines.extend(geom.geoms)  # flatten to individual LineStrings

    return lines

def find_dangling_short_lines(lines, length_threshold=10.0, tolerance=1e-6):
    # Ensure valid LineStrings
    lines = [line for line in lines if isinstance(line, LineString)]

    short_lines = [line for line in lines if line.length <= length_threshold]
    tree = STRtree(lines)

    def is_connected(pt: Point, current_line: LineString):
        buffered_pt = pt.buffer(tolerance)
        results = tree.query(buffered_pt)
        for index in results:
            other = lines[index]
            if other != current_line and buffered_pt.intersects(other):
                return True
        return False

    dangling = []
    for line in short_lines:
        start_pt = Point(line.coords[0])
        end_pt = Point(line.coords[-1])

        start_connected = is_connected(start_pt, line)
        end_connected = is_connected(end_pt, line)

        if (start_connected and not end_connected) or (end_connected and not start_connected):
            dangling.append(line)

    return dangling

def save_lines_to_geojson(lines, output_path):
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": mapping(line),
                "properties": {}  # Add attributes if needed
            }
            for line in lines
        ]
    }

    with open(output_path, "w") as f:
        json.dump(geojson, f, indent=2)

def remove_dangling_lines_from_geojson(map_name, iteration, output_geojson_dir="inference_output_data", length_threshold=100):
    geojson_dir = f"./{output_geojson_dir}/{map_name}_iter{iteration}"
    geojson_path = f"{geojson_dir}/{map_name}_stitch.geojson"

    output_geojson_path = f"{geojson_dir}/{map_name}_post.geojson"
    line_list = load_linestrings_from_geojson(geojson_path)
    dangling_lines = find_dangling_short_lines(line_list, length_threshold=length_threshold)
    # Remove dangling lines by comparing geometry equality
    cleaned_lines = [line for line in line_list if line not in dangling_lines]
    save_lines_to_geojson(cleaned_lines, output_geojson_path)

def remove_dangling_lines_from_list(line_list, length_threshold=100):
    dangling_lines = find_dangling_short_lines(line_list, length_threshold=length_threshold)
    # Remove dangling lines by comparing geometry equality
    cleaned_lines = [line for line in line_list if line not in dangling_lines]
    return cleaned_lines