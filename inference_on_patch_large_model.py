import os
import cv2
import torch
import numpy as np
from model.bert import TwoDimensionalBERTTransformer
from model.decode_prediction import inference_two_dimensional_bert_transformer
from model.dataloader4inference import TwoDimensionalDatasetWithSEQ
from utils import load_linestring_from_geojson_for_finetune, convert_prediction_to_multilinestring, draw_multilinestring_on_image, save_line_groups_to_geojson
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import copy
from process_data.spatial_constraint_operators import max_curvature_segment_length, cal_length_similarity_input_output, cal_orientation_similarity_input_output
from process_data.conflate_lines_on_patch import group_and_select_longest
from collections import defaultdict
from process_data.multiline_to_line_postprocess import MultiLineGraph, break_lines_at_intersections, extract_segments, snap_and_merge_lines


cuda = 1
max_len = 130
max_id = 500
vis_img_size = (500,500)


def inference(iteration, map_dir, map_name, model_epoch=None, cuda=1):
    map_image_path = f"{map_dir}/{map_name}.tif"
    if not os.path.exists(map_image_path):
        map_image_path = f"{map_dir}/{map_name}.png" 
    output_visual_dir = f'./inference_output_data/{map_name}_iter{iteration}'
    input_data_dir = "./"
    input_file = f"{input_data_dir}/inference_input_data/{map_name}_iter{iteration}.geojson"
    if model_epoch is None:
        model_weight_path = f"./trained_weights/fine_tune_large/LineLM_fine_tune_e{model_epoch}.pth"
    else:
        model_weight_path = f"./trained_weights/fine_tune_large/LineLM_fine_tune.pth"


    os.makedirs(output_visual_dir, exist_ok=True)
    os.makedirs(f"{output_visual_dir}/inputs", exist_ok=True)
    os.makedirs(f"{output_visual_dir}/outputs", exist_ok=True)
    geojson_output_path = f"{output_visual_dir}/{map_name}.geojson"


    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    print("Current CUDA Device:", torch.cuda.current_device())


    map_image = cv2.imread(map_image_path)


    input_sequences, shift_list, patch_xy_list = load_linestring_from_geojson_for_finetune(input_file)


    dataset = TwoDimensionalDatasetWithSEQ(input_sequences, shift_list, patch_xy_list, max_len, max_id)


    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)


    # Define tokenizer with special token IDs
    tokenizer = {
        "pad_token_id": max_id + 1,
        "bos_token_id": max_id + 2,
        "eos_token_id": max_id + 3,
        "seq_token_id": max_id + 4,
    }


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




    model.load_state_dict( torch.load(model_weight_path),\
                          strict=False)
    model.to(device)


    model.eval()
    generated_patch_sequences = defaultdict(list)
    input_patch_sequences = defaultdict(list)


    cnt = 0


    for batch_id, batch in enumerate(dataloader):
        input_ids_x = batch["input_ids_x"].to(device)  # Encoder input x
        input_ids_y = batch["input_ids_y"].to(device)  # Encoder input y
        shift_x = -batch["tr_x"]
        shift_y = -batch["tr_y"]
        patch_x = int(batch["patch_x"][0])
        patch_y = int(batch["patch_y"][0])
        attention_mask = None
    #     if patch_x != 1500 or patch_y != 6500:
    #         continue


        # Encode input using the encoder
        batch_size, seq_len = input_ids_x.size()
        position_ids = torch.arange(seq_len, device=input_ids_x.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = model.position_embedding(position_ids)


        embeddings_x = model.embedding_x(input_ids_x)
        embeddings_y = model.embedding_y(input_ids_y)
        encoder_embeddings = torch.cat((embeddings_x, embeddings_y), dim=-1) + position_embeddings


        with torch.no_grad():
            encoder_outputs = model.encoder(inputs_embeds=encoder_embeddings, attention_mask=attention_mask)
            encoder_hidden_states = encoder_outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)


        bos_token_id = tokenizer["bos_token_id"]
        eos_token_id = tokenizer["eos_token_id"]
        pad_token_id = tokenizer["pad_token_id"]
        seq_token_id = tokenizer["seq_token_id"]


        decoder_input_ids_x = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
        decoder_input_ids_y = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)


        for step in range(max_len):
            # Generate decoder embeddings
            decoder_position_ids = torch.arange(decoder_input_ids_x.size(1), device=device).unsqueeze(0).expand(batch_size, -1)
            decoder_position_embeddings = model.position_embedding(decoder_position_ids)


            decoder_embeddings_x = model.embedding_x(decoder_input_ids_x)
            decoder_embeddings_y = model.embedding_y(decoder_input_ids_y)
            decoder_embeddings = torch.cat((decoder_embeddings_x, decoder_embeddings_y), dim=-1) + decoder_position_embeddings


            # Create causal mask
            tgt_seq_len = decoder_embeddings.size(1)
            causal_mask = torch.triu(
                torch.ones(tgt_seq_len, tgt_seq_len, device=device), diagonal=1
            ).bool()


            # Decode the target sequence
            tgt = decoder_embeddings.permute(1, 0, 2)  # (target_seq_len, batch_size, hidden_size)
            memory = encoder_hidden_states.permute(1, 0, 2)  # (input_seq_len, batch_size, hidden_size)
            with torch.no_grad():
                decoder_outputs = model.decoder(
                    tgt=tgt,
                    memory=memory,
                    tgt_mask=causal_mask
                )


            decoder_hidden_states = decoder_outputs.permute(1, 0, 2)  # (batch_size, target_seq_len, hidden_size)


            # Predict next tokens
            logits_x = model.output_x(decoder_hidden_states[:, -1, :])  # (batch_size, vocab_size_x)
            logits_y = model.output_y(decoder_hidden_states[:, -1, :])  # (batch_size, vocab_size_y)


            next_token_id_x = torch.argmax(F.log_softmax(logits_x, dim=-1), dim=-1).unsqueeze(1)  # (batch_size, 1)
            next_token_id_y = torch.argmax(F.log_softmax(logits_y, dim=-1), dim=-1).unsqueeze(1)  # (batch_size, 1)
    #         print(next_token_id_x, next_token_id_y)
            # Sanity check: If all predictions are [BOS], break
            if step > 0 and torch.all(next_token_id_x == bos_token_id) and torch.all(next_token_id_y == bos_token_id):
                print("Warning: All predictions are [BOS]. Check model training or inference settings.")
                break


            # Append predicted tokens
            decoder_input_ids_x = torch.cat((decoder_input_ids_x, next_token_id_x), dim=1)
            decoder_input_ids_y = torch.cat((decoder_input_ids_y, next_token_id_y), dim=1)


            # Stop generation if all batches generate EOS token
            if torch.all(next_token_id_x == eos_token_id) and torch.all(next_token_id_y == eos_token_id):
                break
        # Convert generated token IDs to sequences
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


            # draw prediction on images
            predicted_multilinestring = convert_prediction_to_multilinestring(sequence, max_id=max_id)
            if predicted_multilinestring == []:
                print('Empty Prediction')
                continue


            # Input sequence
            input_sequence_x = input_ids_x[i].tolist()
            input_sequence_y = input_ids_y[i].tolist()
            input_sequence = [(x, y) for x, y in zip(input_sequence_x, input_sequence_y)]


            # draw prediction on images
            input_multilinestring = convert_prediction_to_multilinestring(input_sequence, max_id=max_id)


            max_line_curve, max_seg_length = max_curvature_segment_length(predicted_multilinestring)


            if iteration>-1 and (max_line_curve <= 90 or max_seg_length >= 30): 
#                 print(list(predicted_multilinestring[0]))
#                 print(f'curve and length remove: {max_line_curve}, {max_seg_length}')
                continue
            in_len_ratio, out_len_ratio = cal_length_similarity_input_output(input_multilinestring, predicted_multilinestring, 10)


            if iteration>-1 and (in_len_ratio < 0.6 or out_len_ratio < 0.5):
#                 print('length ratio remove')
                continue
#             print(predicted_multilinestring)
#             print('===='*10)
            generated_patch_sequences[(patch_x, patch_y)].append(predicted_multilinestring[0])
            input_patch_sequences[(patch_x, patch_y)].extend(input_multilinestring)


        cnt += 1


    refined_lines_on_patches = {}
    for patch_xy, lines in generated_patch_sequences.items():  
        grouped_lists = group_and_select_longest(lines, 10, overlap_threshold=0.8)
        row, col = patch_xy  


        output_vis_image_path = f"{output_visual_dir}/outputs/{row}_{col}.jpg"
        map_patch = copy.deepcopy(map_image[col:col+vis_img_size[0], row:row+vis_img_size[1]])   
        pred_patch = draw_multilinestring_on_image(grouped_lists, vis_img_size, \
                                                   output_vis_image_path, base_image=map_patch, \
                                                   line_color=(0,0,255), switch_xy=False, random_color=True)


        output_vis_image_path = f"{output_visual_dir}/inputs/{row}_{col}.jpg"
        map_patch = copy.deepcopy(map_image[col:col+vis_img_size[0], row:row+vis_img_size[1]]) 
        input_path = draw_multilinestring_on_image(input_patch_sequences[patch_xy], vis_img_size, \
                                                   output_vis_image_path, base_image=map_patch, \
                                                   line_color=(0,0,255), switch_xy=False, random_color=True)
        refined_lines_on_patches[(row, col)] = grouped_lists
    save_line_groups_to_geojson(refined_lines_on_patches, geojson_output_path)
