import os
import geojson
import multiprocessing as mp
from functools import partial
from process_data.conflate_lines_on_patch import load_multilinestring_from_geojson, merge_overlapping_lines, smooth_line
from process_data.stitch_lines_helpers import *
from shapely.geometry import LineString, MultiLineString
from process_data.merge_lines import LineConflationGraph
from shapely.validation import make_valid
from shapely import buffer

def validate_and_fix_geometry(geom):
    """Validate and fix geometry if needed"""
    try:
        if not geom.is_valid:
            # Try to fix the geometry
            fixed_geom = make_valid(geom)
            if fixed_geom.is_valid and not fixed_geom.is_empty:
                return fixed_geom
            else:
                return None
        return geom
    except Exception as e:
        print(f"Warning: Could not validate geometry: {e}")
        return None

def safe_update_with_line(graph, line):
    """Safely update graph with line, handling geometry errors"""
    try:
        # Validate the line geometry first
        validated_line = validate_and_fix_geometry(line)
        if validated_line is None or validated_line.is_empty:
            return False
        
        # Try to update with the validated line
        graph.update_with_line(validated_line)
        return True
    except Exception as e:
        print(f"Warning: Failed to add line to graph: {e}")
        return False

def process_line_batch(line_batch, buffer_distance=10.0):
    """Process a batch of lines and return a LineConflationGraph"""
    graph = LineConflationGraph(buffer_distance=buffer_distance)
    
    successful_additions = 0
    for idx, line in enumerate(line_batch):
        if safe_update_with_line(graph, line):
            successful_additions += 1
    
    print(f"Batch processed: {successful_additions}/{len(line_batch)} lines successfully added")
    return graph

def merge_graphs(main_graph, secondary_graph):
    """Merge a secondary graph into the main graph"""
    successful_merges = 0
    for line in secondary_graph.get_all_lines():
        if safe_update_with_line(main_graph, line):
            successful_merges += 1
    print(f"Graph merge: {successful_merges} lines successfully merged")
    return main_graph

def chunk_list(lst, chunk_size):
    """Split list into chunks of specified size"""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def filter_valid_lines(lines):
    """Filter out invalid lines and fix what can be fixed"""
    valid_lines = []
    for line in lines:
        try:
            # Check if line is valid
            if line.is_valid and not line.is_empty and line.length > 0:
                valid_lines.append(line)
            else:
                # Try to fix the line
                fixed_line = validate_and_fix_geometry(line)
                if fixed_line is not None and not fixed_line.is_empty and fixed_line.length > 0:
                    valid_lines.append(fixed_line)
        except Exception as e:
            print(f"Warning: Skipping invalid line: {e}")
            continue
    
    print(f"Filtered lines: {len(valid_lines)}/{len(lines)} valid lines")
    return valid_lines

def stitch_results_on_patches_multiprocess(iteration, map_name, output_geojson_dir="inference_output_data", 
                                         stitch_buffer=10, num_processes=8, batch_size=50):
    geojson_dir = f"./{output_geojson_dir}/{map_name}_iter{iteration}"
    input_geojson_path = f"{geojson_dir}/{map_name}.geojson"
    output_geojson_path = f"{geojson_dir}/{map_name}_stitch.geojson"

    try:
        lines_on_patches = load_multilinestring_from_geojson(input_geojson_path)
    except Exception as e:
        print(f"Error loading geojson: {e}")
        return

    rescaled_lines = []

    for (x, y), lines in lines_on_patches.items():
        try:
            multilines = MultiLineString([LineString(ln) for ln in lines])
            rescaled_multilines = offset_geometry(multilines, x, y)
            rescaled_lines.extend(list(rescaled_multilines.geoms))
        except Exception as e:
            print(f"Warning: Error processing patch ({x}, {y}): {e}")
            continue
    
    print(f"Initial lines loaded: {len(rescaled_lines)}")
    
    # Filter and validate lines
    rescaled_lines = filter_valid_lines(rescaled_lines)
    rescaled_lines = sorted(rescaled_lines, key=lambda ln: ln.length, reverse=True) 
    
    interm_geojson_path = f"{geojson_dir}/{map_name}_raw.geojson"
    save_linestrings_to_geojson(rescaled_lines, interm_geojson_path)
    
    print(f"Processing {len(rescaled_lines)} valid lines with multiprocessing")
    
    if len(rescaled_lines) == 0:
        print("No valid lines to process")
        return
    
    # Split lines into batches for parallel processing
    line_batches = list(chunk_list(rescaled_lines, batch_size))
    print(f"Created {len(line_batches)} batches of size {batch_size}")
    
    # Process batches in parallel
    print("Processing line batches in parallel...")
    try:
        with mp.Pool(processes=num_processes) as pool:
            process_func = partial(process_line_batch, buffer_distance=stitch_buffer)
            batch_graphs = pool.map(process_func, line_batches)
    except Exception as e:
        print(f"Error in parallel processing: {e}")
        return
    
    print(f"Processed {len(batch_graphs)} batch graphs, now merging...")
    
    # Merge all batch graphs into a single graph
    main_graph = LineConflationGraph(buffer_distance=stitch_buffer)
    
    for i, batch_graph in enumerate(batch_graphs):
        if i % 10 == 0:  # Reduced frequency for less spam
            print(f"Merging batch graph {i+1}/{len(batch_graphs)}")
        
        try:
            batch_lines = batch_graph.get_all_lines()
            for line in batch_lines:
                safe_update_with_line(main_graph, line)
        except Exception as e:
            print(f"Warning: Error merging batch {i}: {e}")
            continue
    
    print(f"#nodes after initial merge: {main_graph.graph.number_of_nodes()}")
    
    # Get merged lines from the combined graph
    merged_lines = []
    try:
        for i, line in enumerate(main_graph.get_all_lines()):
            merged_lines.append(line)
        merged_lines = sorted(merged_lines, key=lambda ln: ln.length, reverse=True)
    except Exception as e:
        print(f"Error getting merged lines: {e}")
        return
    
    print(f"Got {len(merged_lines)} merged lines")
    
    # Second pass for cleanup with error handling
    print("Second pass for cleanup...")
    final_graph = LineConflationGraph(buffer_distance=stitch_buffer)
    successful_final = 0
    
    for i, line in enumerate(merged_lines):
        if i % 100 == 0:
            print(f"Second pass: Processing line {i + 1}/{len(merged_lines)} (Success: {successful_final})")
        
        if safe_update_with_line(final_graph, line):
            successful_final += 1
    
    print(f"Second pass complete: {successful_final}/{len(merged_lines)} lines successfully processed")
    print(f"#nodes before removing small lines: {final_graph.graph.number_of_nodes()}")
    
    try:
        final_graph.remove_small_isolated_lines(30)
        print(f"#nodes after removing small lines: {final_graph.graph.number_of_nodes()}")
    except Exception as e:
        print(f"Warning: Error removing small lines: {e}")
    
    final_lines = []
    try:
        for i, line in enumerate(final_graph.get_all_lines()):
            final_lines.append(line)
        
        save_linestrings_to_geojson(final_lines, output_geojson_path)
        print(f"Saved {len(final_lines)} final lines to {output_geojson_path}")
    except Exception as e:
        print(f"Error saving final results: {e}")

# Enhanced hierarchical version with error handling
def stitch_results_hierarchical(iteration, map_name, output_geojson_dir="inference_output_data", 
                               stitch_buffer=10, num_processes=4, batch_size=50):
    """Hierarchical merging approach with error handling"""
    geojson_dir = f"./{output_geojson_dir}/{map_name}_iter{iteration}"
    input_geojson_path = f"{geojson_dir}/{map_name}.geojson"
    output_geojson_path = f"{geojson_dir}/{map_name}_stitch.geojson"

    try:
        lines_on_patches = load_multilinestring_from_geojson(input_geojson_path)
    except Exception as e:
        print(f"Error loading geojson: {e}")
        return

    rescaled_lines = []
    for (x, y), lines in lines_on_patches.items():
        try:
            multilines = MultiLineString([LineString(ln) for ln in lines])
            rescaled_multilines = offset_geometry(multilines, x, y)
            rescaled_lines.extend(list(rescaled_multilines.geoms))
        except Exception as e:
            print(f"Warning: Error processing patch ({x}, {y}): {e}")
            continue
    
    # Filter and validate lines
    rescaled_lines = filter_valid_lines(rescaled_lines)
    rescaled_lines = sorted(rescaled_lines, key=lambda ln: ln.length, reverse=True)
    
    # Process in multiple levels for better scalability
    current_lines = rescaled_lines
    level = 0
    
    while len(current_lines) > batch_size * 2:  # Continue until manageable size
        level += 1
        print(f"Level {level}: Processing {len(current_lines)} lines")
        
        line_batches = list(chunk_list(current_lines, batch_size))
        
        # Process batches in parallel
        try:
            with mp.Pool(processes=num_processes) as pool:
                process_func = partial(process_line_batch, buffer_distance=stitch_buffer)
                batch_graphs = pool.map(process_func, line_batches)
        except Exception as e:
            print(f"Error in level {level} processing: {e}")
            break
        
        # Extract lines from each batch graph
        current_lines = []
        for batch_graph in batch_graphs:
            try:
                current_lines.extend(batch_graph.get_all_lines())
            except Exception as e:
                print(f"Warning: Error extracting lines from batch: {e}")
                continue
        
        current_lines = filter_valid_lines(current_lines)
        current_lines = sorted(current_lines, key=lambda ln: ln.length, reverse=True)
        print(f"Level {level} complete: {len(current_lines)} lines remaining")
    
    # Final processing of remaining lines
    print("Final processing...")
    final_graph = LineConflationGraph(buffer_distance=stitch_buffer)
    successful_final = 0
    
    for i, line in enumerate(current_lines):
        if i % 50 == 0:
            print(f"Final: Processing line {i + 1}/{len(current_lines)} (Success: {successful_final})")
        
        if safe_update_with_line(final_graph, line):
            successful_final += 1
    
    try:
        final_graph.remove_small_isolated_lines(30)
    except Exception as e:
        print(f"Warning: Error removing small lines: {e}")
    
    try:
        final_lines = list(final_graph.get_all_lines())
        save_linestrings_to_geojson(final_lines, output_geojson_path)
        print(f"Saved {len(final_lines)} final lines to {output_geojson_path}")
    except Exception as e:
        print(f"Error saving final results: {e}")

# Keep the original function for backward compatibility (with error handling)
def stitch_results_on_patches(iteration, map_name, output_geojson_dir="inference_output_data", stitch_buffer=10):
    """Original single-threaded version with error handling"""
    geojson_dir = f"./{output_geojson_dir}/{map_name}_iter{iteration}"
    input_geojson_path = f"{geojson_dir}/{map_name}.geojson"
    output_geojson_path = f"{geojson_dir}/{map_name}_stitch.geojson"

    try:
        lines_on_patches = load_multilinestring_from_geojson(input_geojson_path)
    except Exception as e:
        print(f"Error loading geojson: {e}")
        return

    rescaled_lines = []

    for (x, y), lines in lines_on_patches.items():
        try:
            multilines = MultiLineString([LineString(ln) for ln in lines])
            rescaled_multilines = offset_geometry(multilines, x, y)
            rescaled_lines.extend(list(rescaled_multilines.geoms))
        except Exception as e:
            print(f"Warning: Error processing patch ({x}, {y}): {e}")
            continue
    
    # Filter and validate lines
    rescaled_lines = filter_valid_lines(rescaled_lines)
    rescaled_lines = sorted(rescaled_lines, key=lambda ln: ln.length, reverse=True) 
    
    interm_geojson_path = f"{geojson_dir}/{map_name}_raw.geojson"
    save_linestrings_to_geojson(rescaled_lines, interm_geojson_path)
    
    graph = LineConflationGraph(buffer_distance=stitch_buffer)
    successful_additions = 0

    for idx, line in enumerate(rescaled_lines):
        if idx % 100 == 0:
            print(f"Processing line {idx + 1}/{len(rescaled_lines)} (Success: {successful_additions})")
        
        if safe_update_with_line(graph, line):
            successful_additions += 1
    
    print(f"Successfully added {successful_additions}/{len(rescaled_lines)} lines")
    print(f"#nodes before removing: {graph.graph.number_of_nodes()}")
    
    try:
        merged_lines = list(graph.get_all_lines())
        merged_lines = sorted(merged_lines, key=lambda ln: ln.length, reverse=True) 
        
        # Second pass
        final_graph = LineConflationGraph(buffer_distance=stitch_buffer)
        for line in merged_lines:
            safe_update_with_line(final_graph, line)
        
        final_graph.remove_small_isolated_lines(30)
        final_lines = list(final_graph.get_all_lines())
        save_linestrings_to_geojson(final_lines, output_geojson_path)
        print(f"Saved {len(final_lines)} final lines")
    except Exception as e:
        print(f"Error in final processing: {e}")
