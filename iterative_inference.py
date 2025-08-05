import os
import argparse
from inference_on_patch_large_model import inference
from stitch_lines_on_map import stitch_results_on_patches
from remove_dangling_lines_on_map import remove_dangling_lines_from_geojson as remove_dangling_lines

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Generate input geojson for line summarization")
    parser.add_argument("--iteration", type=int, help="The index of iteration")
    parser.add_argument("--map_dir", type=str, help="Directory for map images")
    parser.add_argument("--in_geojson_dir", type=str, default=None, help="Directory for the input geojson")
    parser.add_argument("--out_geojson_dir", type=str, help="Directory for the output geojson")
    parser.add_argument("--in_geojson_name", type=str, default=None, help="The input geojson file name")
#     parser.add_argument("--out_geojson_name", type=str, help="The output geojson file name")
    parser.add_argument("--map_name", type=str, help="The base map file name")
    parser.add_argument("--stride", type=int, default=250)
    parser.add_argument("--patch_size", type=int, default=500)
    parser.add_argument("--model_version", type=int, default=100)
    parser.add_argument("--cuda", type=int, default=1)
    
    args = parser.parse_args()
    iteration = args.iteration
    
    map_dir = args.map_dir
    map_name = args.map_name
    if args.in_geojson_dir:
        input_geojson_dir = args.in_geojson_dir
    else:
        input_geojson_dir = f"./inference_output_data/{map_name}_iter{iteration-1}"
    output_geojson_dir = args.out_geojson_dir
    
    if args.in_geojson_name:
        in_geojson_name = args.in_geojson_name
    else:
        in_geojson_name = f"{map_name}_post" 
    
    ######################################
    #### Generate Input Data
    ######################################
    os.makedirs(output_geojson_dir, exist_ok=True)

    map_image_path = f"{map_dir}/{map_name}.tif"  
    if not os.path.exists(map_image_path):
        map_image_path = f"{map_dir}/{map_name}.png" 
    input_geojson_path = f"{input_geojson_dir}/{in_geojson_name}.geojson"
    input_output_path = f"{output_geojson_dir}/{map_name}_iter{iteration}.geojson"
    
    if iteration == 0:
        from preprocess.generate_single_lines_for_inference_iter0 import process_geojson_to_paths   
        
        process_geojson_to_paths(input_geojson_path, map_image_path, input_output_path,\
                                 stride=args.stride, dim_threshold=args.patch_size)
    else:
        from preprocess.generate_single_lines_for_iterative_inference import process_geojson_to_paths
        process_geojson_to_paths(input_geojson_path, map_image_path, input_output_path,\
                                 stride=args.stride, dim_threshold=args.patch_size)
    print(f"=== Processed {in_geojson_name} ===")
    
    ######################################
    #### Inference
    ######################################
    model_version = args.model_version
    inference(iteration, map_dir, map_name, model_epoch=model_version, cuda=args.cuda)
    
    ######################################
    #### Postprocess
    ######################################
##     postprocess(iteration, map_dir, map_name)
    
    ######################################
    #### Stitch the Patch-level results to 
    #### Map-level
    ######################################
    stitch_results_on_patches(iteration, map_name, stitch_buffer=10)

    ######################################
    #### Remove dangling lines
    ######################################
    remove_dangling_lines(map_name, iteration)