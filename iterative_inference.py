import os
import argparse
from inference_on_patch_large_model_multiprocess import inference_multiprocess
from stitch_lines_on_map import stitch_results_on_patches, stitch_results_on_patches_multiprocess
from remove_dangling_lines_on_map import remove_dangling_lines_from_geojson as remove_dangling_lines

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Generate input geojson for line summarization")
    parser.add_argument("--iteration", type=int, help="The index of iteration")
    parser.add_argument("--map_dir", type=str, help="Directory for map images")
    parser.add_argument("--extract_geojson_dir", type=str, default=None, help="Directory for the extract geojson")
    parser.add_argument("--in_geojson_dir", type=str, default=None, help="Directory for the input geojson")
    parser.add_argument("--out_geojson_dir", type=str, help="Directory for the output geojson")
    parser.add_argument("--in_geojson_name", type=str, default=None, help="The input geojson file name")
    parser.add_argument("--map_name", type=str, help="The base map file name")
    parser.add_argument("--stride", type=int, default=250)
    parser.add_argument("--patch_size", type=int, default=500)
    parser.add_argument("--model_version", type=int, default=100)
    parser.add_argument("--cuda", type=int, default=3)
    
    args = parser.parse_args()
    iteration = args.iteration
    
    map_dir = args.map_dir
    map_name = args.map_name
    if iteration == 0:
        input_geojson_dir = args.in_geojson_dir
    else:
        input_geojson_dir = f"./inference_output_data/{map_name}_iter{iteration-1}"
    
    output_geojson_dir = args.out_geojson_dir
    
    if iteration == 0:
        in_geojson_name = args.in_geojson_name
    else:
        in_geojson_name = f"{map_name}_post" 

    extract_geojson_dir = args.extract_geojson_dir
    extract_geojson_name = args.in_geojson_name 

    ######################################
    #### Generate Input Data
    ######################################
    os.makedirs(output_geojson_dir, exist_ok=True)

    map_image_path = f"{map_dir}/{map_name}.tif"  
    if not os.path.exists(map_image_path):
        map_image_path = f"{map_dir}/{map_name}.png" 
    print(f"=== Processing Map Image: {map_image_path} ===")
    input_geojson_path = f"{input_geojson_dir}/{in_geojson_name}.geojson"
    input_output_path = f"{output_geojson_dir}/{map_name}_iter{iteration}.geojson"
    
    if iteration == 0:
        from preprocess.generate_single_lines_for_inference_iter0 import process_geojson_to_paths   
        
        process_geojson_to_paths(input_geojson_path, map_image_path, input_output_path,\
                                 stride=args.stride, dim_threshold=args.patch_size)
    else:
        from preprocess.generate_single_lines_for_iterative_inference import process_geojson_to_paths
        extract_geojson_path = f"{extract_geojson_dir}/{extract_geojson_name}.geojson"
        process_geojson_to_paths(input_geojson_path, extract_geojson_path, map_image_path, input_output_path,\
                                 stride=args.stride, dim_threshold=args.patch_size)
    print(f"=== Processed {in_geojson_name} ===")
    
    ######################################
    #### Inference
    ######################################
    model_version = args.model_version
    inference_multiprocess(iteration, map_dir, map_name, model_epoch=model_version, num_processes=8, cuda_devices=[args.cuda], chunk_size=100)
    
    ######################################
    #### Stitch the Patch-level results to 
    #### Map-level
    ######################################
    stitch_results_on_patches_multiprocess(iteration, map_name, stitch_buffer=10)

    ######################################
    #### Remove dangling lines
    ######################################
    remove_dangling_lines(map_name, iteration)
