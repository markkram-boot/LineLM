import os
import cv2
import torch
import numpy as np
import multiprocessing as mp
import json
import pickle
from functools import partial
from model.bert import TwoDimensionalBERTTransformer
from model.dataloader4inference import TwoDimensionalDatasetWithSEQ
from utils import load_linestring_from_geojson_for_finetune, convert_prediction_to_multilinestring, draw_multilinestring_on_image, save_line_groups_to_geojson, convert_tensor_input_to_multilinestring
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import copy
from process_data.spatial_constraint_operators import max_curvature_segment_length
from process_data.conflate_lines_on_patch import group_and_select_longest
from collections import defaultdict
from process_data.multiline_to_line_postprocess import MultiLineGraph, break_lines_at_intersections, extract_segments, snap_and_merge_lines

max_len = 130
max_id = 500
vis_img_size = (500,500)

def process_batch_indices(start_idx, end_idx, input_file, model_weight_path, device_id, tokenizer_config, max_id, max_len, chunk_id, temp_dir):
    """Process a range of indices from the dataset - no tensor serialization"""
    
    # Set CUDA device for this process
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    
    # Load model once per chunk
    vocab_size_x = max_id + 5
    vocab_size_y = max_id + 5
    model = TwoDimensionalBERTTransformer(
        vocab_size_x=vocab_size_x,
        vocab_size_y=vocab_size_y,
        hidden_size=1024,
        num_hidden_layers=8,
        num_attention_heads=16, 
        intermediate_size=6*1024, 
        max_position_embeddings=max_len, 
        decoder_layers=8, 
        nhead=16
    )
    
    model.load_state_dict(torch.load(model_weight_path, map_location=device), strict=False)
    model.to(device)
    model.eval()
    
    # Load dataset for this chunk only
    input_sequences, shift_list, patch_xy_list, reference_lines = load_linestring_from_geojson_for_finetune(input_file)
    
    # Slice the data for this chunk
    chunk_input_sequences = input_sequences[start_idx:end_idx]
    chunk_shift_list = shift_list[start_idx:end_idx] 
    chunk_patch_xy_list = patch_xy_list[start_idx:end_idx]
    chunk_reference_lines = reference_lines[start_idx:end_idx]
    
    dataset = TwoDimensionalDatasetWithSEQ(chunk_input_sequences, chunk_shift_list, 
                                          chunk_patch_xy_list, chunk_reference_lines, 
                                          max_len, max_id)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    all_results = []
    
    # Process each batch in the chunk
    for batch_idx, batch in enumerate(dataloader):
        results = []
        
        # Use tensors directly without serialization
        input_ids_x = batch["input_ids_x"].to(device)
        input_ids_y = batch["input_ids_y"].to(device)
        shift_x = -batch["tr_x"]
        shift_y = -batch["tr_y"]
        patch_x = int(batch["patch_x"][0])
        patch_y = int(batch["patch_y"][0])
        reference_line = batch["reference_line"]

        attention_mask = None

        # Encode input using the encoder
        batch_size, seq_len = input_ids_x.size()
        position_ids = torch.arange(seq_len, device=input_ids_x.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = model.position_embedding(position_ids)

        embeddings_x = model.embedding_x(input_ids_x)
        embeddings_y = model.embedding_y(input_ids_y)
        encoder_embeddings = torch.cat((embeddings_x, embeddings_y), dim=-1) + position_embeddings

        with torch.no_grad():
            encoder_outputs = model.encoder(inputs_embeds=encoder_embeddings, attention_mask=attention_mask)
            encoder_hidden_states = encoder_outputs.last_hidden_state

        bos_token_id = tokenizer_config["bos_token_id"]
        eos_token_id = tokenizer_config["eos_token_id"]
        pad_token_id = tokenizer_config["pad_token_id"]
        seq_token_id = tokenizer_config["seq_token_id"]

        decoder_input_ids_x = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
        decoder_input_ids_y = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)

        # Generation loop
        for step in range(max_len):
            decoder_position_ids = torch.arange(decoder_input_ids_x.size(1), device=device).unsqueeze(0).expand(batch_size, -1)
            decoder_position_embeddings = model.position_embedding(decoder_position_ids)

            decoder_embeddings_x = model.embedding_x(decoder_input_ids_x)
            decoder_embeddings_y = model.embedding_y(decoder_input_ids_y)
            decoder_embeddings = torch.cat((decoder_embeddings_x, decoder_embeddings_y), dim=-1) + decoder_position_embeddings

            tgt_seq_len = decoder_embeddings.size(1)
            causal_mask = torch.triu(
                torch.ones(tgt_seq_len, tgt_seq_len, device=device), diagonal=1
            ).bool()

            tgt = decoder_embeddings.permute(1, 0, 2)
            memory = encoder_hidden_states.permute(1, 0, 2)
            
            with torch.no_grad():
                decoder_outputs = model.decoder(
                    tgt=tgt,
                    memory=memory,
                    tgt_mask=causal_mask
                )

            decoder_hidden_states = decoder_outputs.permute(1, 0, 2)

            logits_x = model.output_x(decoder_hidden_states[:, -1, :])
            logits_y = model.output_y(decoder_hidden_states[:, -1, :])

            next_token_id_x = torch.argmax(F.log_softmax(logits_x, dim=-1), dim=-1).unsqueeze(1)
            next_token_id_y = torch.argmax(F.log_softmax(logits_y, dim=-1), dim=-1).unsqueeze(1)

            if step > 0 and torch.all(next_token_id_x == bos_token_id) and torch.all(next_token_id_y == bos_token_id):
                break

            decoder_input_ids_x = torch.cat((decoder_input_ids_x, next_token_id_x), dim=1)
            decoder_input_ids_y = torch.cat((decoder_input_ids_y, next_token_id_y), dim=1)

            if torch.all(next_token_id_x == eos_token_id) and torch.all(next_token_id_y == eos_token_id):
                break
        
        # Process results for this batch
        for i in range(batch_size):
            sequence_x = decoder_input_ids_x[i].tolist()[1:]
            sequence_y = decoder_input_ids_y[i].tolist()[1:]
            sequence = []
            for x, y in zip(sequence_x, sequence_y):
                if x == eos_token_id or y == eos_token_id:
                    break
                if x == seq_token_id or y == seq_token_id:
                    sequence.append((seq_token_id,seq_token_id))
                elif x != pad_token_id and y != pad_token_id:
                    sequence.append((x, y))
            
            if len(input_ids_x[i].tolist()) < 30:
                continue

            predicted_multilinestring = convert_prediction_to_multilinestring(sequence, max_id=max_id)
            if predicted_multilinestring == []:
                continue
            
            max_line_curve, max_seg_length = max_curvature_segment_length(predicted_multilinestring)

            if max_line_curve <= 90 or max_seg_length >= 30: 
                continue

            input_sequence_x = input_ids_x[i].tolist()
            input_sequence_y = input_ids_y[i].tolist()
            input_sequence = [(x, y) for x, y in zip(input_sequence_x, input_sequence_y)]
            input_multilinestring = convert_prediction_to_multilinestring(input_sequence, max_id=max_id)

            results.append({
                'patch_xy': (patch_x, patch_y),
                'predicted_line': predicted_multilinestring[0],
                'input_lines': input_multilinestring
            })
        
        all_results.extend(results)

    # Save results to temporary file
    result_file = os.path.join(temp_dir, f"results_chunk_{chunk_id}.pkl")
    with open(result_file, 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"[Process {chunk_id}] Chunk completed! Generated {len(all_results)} valid results ({end_idx - start_idx} samples processed)")
    return result_file

def chunk_indices(total_length, chunk_size):
    """Generate start and end indices for chunks"""
    chunks = []
    for i in range(0, total_length, chunk_size):
        end_idx = min(i + chunk_size, total_length)
        chunks.append((i, end_idx))
    return chunks

def inference_multiprocess(iteration, map_dir, map_name, model_epoch=None, num_processes=3, cuda_devices=[3], chunk_size=100):
    """Main inference function with file-based multiprocessing"""
    
    map_image_path = f"{map_dir}/{map_name}.tif"
    if not os.path.exists(map_image_path):
        map_image_path = f"{map_dir}/{map_name}.png" 
    
    output_visual_dir = f'./inference_output_data/{map_name}_iter{iteration}'
    input_data_dir = "./"
    input_file = f"{input_data_dir}/inference_output_data/{map_name}_iter{iteration}.geojson"
    
    if iteration==0:
        model_weight_path = f"/data4/critical-maas/LineLM/trained_weights/fine_tune_large/LineLM_fine_tune.pth"
    else:
        model_weight_path = f"/data4/critical-maas/LineLM/trained_weights/fine_tune_large/two_dimensional_bert_transformer_e110.pth"

    print(f"model path: {model_weight_path}")
    os.makedirs(output_visual_dir, exist_ok=True)
    os.makedirs(f"{output_visual_dir}/inputs", exist_ok=True)
    os.makedirs(f"{output_visual_dir}/outputs", exist_ok=True)
    
    # Create temporary directory for inter-process communication
    temp_dir = f"{output_visual_dir}/temp_results"
    os.makedirs(temp_dir, exist_ok=True)
    
    geojson_output_path = f"{output_visual_dir}/{map_name}.geojson"

    # Load map image
    map_image = cv2.imread(map_image_path)

    # Get dataset size without loading all data
    input_sequences, shift_list, patch_xy_list, reference_lines = load_linestring_from_geojson_for_finetune(input_file)
    total_samples = len(input_sequences)

    # Define tokenizer
    tokenizer_config = {
        "pad_token_id": max_id + 1,
        "bos_token_id": max_id + 2,
        "eos_token_id": max_id + 3,
        "seq_token_id": max_id + 4,
    }

    # Create index chunks
    index_chunks = chunk_indices(total_samples, chunk_size)
    print(f"Processing {total_samples} samples in {len(index_chunks)} chunks of size {chunk_size}")

    # Prepare arguments for multiprocessing
    chunk_args = []
    for i, (start_idx, end_idx) in enumerate(index_chunks):
        device_id = cuda_devices[i % len(cuda_devices)]
        chunk_args.append((start_idx, end_idx, input_file, model_weight_path, device_id, 
                          tokenizer_config, max_id, max_len, i, temp_dir))

    # Run inference in parallel
    print(f"Running inference with {num_processes} processes on devices {cuda_devices}")
    
    with mp.Pool(processes=min(num_processes, len(index_chunks))) as pool:
        result_files = pool.starmap(process_batch_indices, chunk_args)

    # Load and combine results from temporary files
    print("Loading results from temporary files...")
    flattened_results = []
    for result_file in result_files:
        with open(result_file, 'rb') as f:
            chunk_results = pickle.load(f)
            flattened_results.extend(chunk_results)
        # Clean up temporary file
        os.remove(result_file)
    
    # Remove temporary directory
    os.rmdir(temp_dir)

    print(f"Total results collected: {len(flattened_results)}")

    # Collect results
    generated_patch_sequences = defaultdict(list)
    input_patch_sequences = defaultdict(list)

    for result in flattened_results:
        patch_xy = result['patch_xy']
        generated_patch_sequences[patch_xy].append(result['predicted_line'])
        input_patch_sequences[patch_xy].extend(result['input_lines'])

    # Post-processing and visualization
    refined_lines_on_patches = {}
    for patch_xy, lines in generated_patch_sequences.items():  
        grouped_lists = group_and_select_longest(lines, 10, overlap_threshold=0.8)
        row, col = patch_xy  

        output_vis_image_path = f"{output_visual_dir}/outputs/{row}_{col}.jpg"
        map_patch = copy.deepcopy(map_image[col:col+vis_img_size[0], row:row+vis_img_size[1]])     
        pred_patch = draw_multilinestring_on_image(grouped_lists, vis_img_size,
                                                   output_vis_image_path, base_image=map_patch,
                                                   line_color=(0,0,255), switch_xy=False, random_color=True)

        output_vis_image_path = f"{output_visual_dir}/inputs/{row}_{col}.jpg"
        map_patch = copy.deepcopy(map_image[col:col+vis_img_size[0], row:row+vis_img_size[1]]) 
        input_path = draw_multilinestring_on_image(input_patch_sequences[patch_xy], vis_img_size, 
                                                   output_vis_image_path, base_image=map_patch, 
                                                   line_color=(0,0,255), switch_xy=False, random_color=True)

        refined_lines_on_patches[(row, col)] = grouped_lists
        
    save_line_groups_to_geojson(refined_lines_on_patches, geojson_output_path)
    print(f"Inference completed. Results saved to {geojson_output_path}")
