import os
import geojson
from process_data.conflate_lines_on_patch import load_multilinestring_from_geojson, merge_overlapping_lines, smooth_line
from process_data.stitch_lines_helpers import *
from shapely.geometry import LineString, MultiLineString
from process_data.merge_lines import LineConflationGraph

# def stitch_results_on_patches(iteration, map_name, stitch_buffer=10):
#     geojson_dir = f"./inference_output_data/{map_name}_iter{iteration}"
#     input_geojson_path = f"{geojson_dir}/{map_name}.geojson"
#     output_geojson_path = f"{geojson_dir}/{map_name}_raw.geojson"

#     lines_on_patches = load_multilinestring_from_geojson(input_geojson_path)

#     rescaled_lines = []

#     for (x, y), lines in lines_on_patches.items():
# #         if not (0<x<3001 and 0<y<3500):
# #             continue
#         multilines = MultiLineString([LineString(ln) for ln in lines])
#         rescaled_multilines = offset_geometry(multilines, x, y)
#         rescaled_lines.extend(list(rescaled_multilines.geoms))
# #     rescaled_lines = sorted(rescaled_lines, key=lambda ln: ln.length, reverse=True)
# #     merged_lines = tolerant_merge_lines(rescaled_lines, snap_tolerance=10.0)
# #     merged_lines = sorted(merged_lines, key=lambda ln: ln.length, reverse=True)
# #     dedup_merged_lines = remove_redundant_lines(merged_lines, buffer_tolerance=10)
# #     stitched_lines = merge_overlapping_lines(dedup_merged_lines, stitch_buffer)
# #     grouped_lines = group_and_select_longest(rescaled_lines, 10, overlap_threshold=0.9)
# #     merged_smooth_lines = [smooth_line(line, smoothing_factor=10) for line in merged_lines]

#     save_linestrings_to_geojson(rescaled_lines, output_geojson_path)

def stitch_results_on_patches(iteration, map_name, output_geojson_dir="inference_output_data", stitch_buffer=10):
    geojson_dir = f"./{output_geojson_dir}/{map_name}_iter{iteration}"
    input_geojson_path = f"{geojson_dir}/{map_name}.geojson"
    output_geojson_path = f"{geojson_dir}/{map_name}_stitch.geojson"

    lines_on_patches = load_multilinestring_from_geojson(input_geojson_path)

    rescaled_lines = []

    for (x, y), lines in lines_on_patches.items():
#         if not (0<x<3001 and 0<y<3500):
#             continue
        multilines = MultiLineString([LineString(ln) for ln in lines])
        rescaled_multilines = offset_geometry(multilines, x, y)
        rescaled_lines.extend(list(rescaled_multilines.geoms))
    
    rescaled_lines = sorted(rescaled_lines, key=lambda ln: ln.length, reverse=True) 
    interm_geojson_path = f"{geojson_dir}/{map_name}_raw.geojson"
    save_linestrings_to_geojson(rescaled_lines, interm_geojson_path)
    
    graph = LineConflationGraph(buffer_distance=10.0)

    for line in rescaled_lines:
        graph.update_with_line(line)
    print(f"#nodes before removing: {graph.graph.number_of_nodes()}")
#     graph.remove_small_isolated_lines(80)
#     print(f"#nodes after removing: {graph.graph.number_of_nodes()}")
    merged_lines = []
    for i, line in enumerate(graph.get_all_lines()):
        merged_lines.append(line)
    merged_lines = sorted(merged_lines, key=lambda ln: ln.length, reverse=True) 
#     snapped_merged_lines = tolerant_merge_lines(merged_lines, stitch_buffer*1.1)
#     print(f'Before and after snap, #lines = : {len(merged_lines)}, {len(snapped_merged_lines)}')
    
    graph = LineConflationGraph(buffer_distance=10.0)
    for line in merged_lines:
        graph.update_with_line(line)
    graph.remove_small_isolated_lines(30)
    merged_lines = []
    for i, line in enumerate(graph.get_all_lines()):
        merged_lines.append(line)
    save_linestrings_to_geojson(merged_lines, output_geojson_path)
    