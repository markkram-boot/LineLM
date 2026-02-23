# LineLM: A Language Model for Refining Vector Line Geometries


A transformer-based model for processing and generating vector single-lines. This project implements a LineLM (Bert-encoder and GPT-decoder) architecture for handling geospatial line data.


## Overview


LineLM provides two main functionalities:
1. **Pre-training**: Masked language modeling on vector lines using the encoder of LineLM
2. **Fine-tuning**: Sequence-to-sequence learning using a LineLM model


The models are designed to process geospatial vector lines represented as sequences of coordinate pairs, making them suitable for line refinement to correct distortions, fill gaps, and restore connectivity.


## Architecture


### BERT-encoder (Pre-training)
- **Purpose**: Self-supervised pre-training on vector lines
- **Architecture**: BERT-based encoder with separate embeddings for X and Y coordinates
- **Training**: Masked language modeling with 15% token masking
- **Input**: Single vector lines
- **Output**: Predictions for masked X and Y coordinates


### BERT-encoder + GPT-decoder  (Fine-tuning)
- **Purpose**: Sequence-to-sequence learning for vector lines
- **Architecture**: BERT encoder + GPT decoder
- **Training**: Teacher forcing with cross-entropy loss
- **Input**: Source and noisy line sequences
- **Output**: clean line sequences


## Project Structure


```
LineLM/
├── README.md                    # This file
├── pretrain_bert_large.py      # Pre-training script
├── fine_tune_large.py          # Fine-tuning script
├── utils.py                    # Data loading utilities
└── model/
    ├── bert_pretrain.py        # MaskedBERT model definition
    ├── bert.py                 # LineLM model
    ├── dataloader_mlm.py       # DataLoader for pre-training
    └── dataloader.py           # DataLoader for fine-tuning
```


## Environment Setup Using Poetry 

This project uses [Poetry](https://python-poetry.org/) for dependency management. Install dependencies using:

```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install

# Activate the virtual environment
poetry shell
```


## Data Format


The models expect GeoJSON files with the following structure:


### Pre-training Data
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "LineString",
        "coordinates": [[x1, y1], [x2, y2], ..., [xn, yn]]
      }
    }
  ]
}
```


### Fine-tuning Data
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": [
        [[x1, y1], [x2, y2], ..., [xn, yn]],  # First trajectory
        [[x1, y1], [x2, y2], ..., [xm, ym]]   # Second trajectory
      ]
    }
  ]
}
```


## Usage


### 1. Data Preparation


Create a `data/` directory and place your GeoJSON files:


```bash
mkdir data/
```


Place your GeoJSON files for pretraining in the `data/` directory.
Place your input and ground truth GeoJSON files for fine-tuning in the `data/` directory.


You can download the GeoJSON files for both pretraining and fine-tuning from [this link](https://drive.google.com/drive/folders/1of4o8HmkL5kuBJghoU_i8doujeG97ELr?usp=sharing).




### 2. Pre-training


Run the pre-training script to learn general trajectory representations:


```bash
python pretrain_bert_large.py
```


### 3. Fine-tuning


Run the fine-tuning script for sequence-to-sequence learning:


```bash
python fine_tune_large.py
```


## Output


### Pre-training
- Model checkpoints saved every 20 epochs in `./trained_weights/pretrain_trainset_large/`
- Format: `bert_pretrain_e{epoch}.pth`


### Fine-tuning  
- Model checkpoints saved every 20 epoch in `./trained_weights/fine_tune_large/`
- Format: `LineLM_fine_tune_e{epoch}.pth`




## Model Weights


### Pre-trained Models


| Model | Description | Download Link | Size |
|-------|-------------|---------------|------|
| Pretrain| BERT encoder pre-trained on vector lines | [Download](https://drive.google.com/file/d/1URW3_fCFvG2n8tBJzoZaA6a8pPAUwpoD/view?usp=sharing) | 642.7MB |
| Fine-tune | Large model fine-tuned for line refinement | [Download](https://drive.google.com/file/d/1oyG5gUpbNfkEiF08WofpGyZMsDEyrFUo/view?usp=sharing) | 1.13GB |


### Model Card


**Model Name**: LineLM (Line Language Model)  
**Model Type**: Transformer-based encoder-decoder for geospatial vector lines  
**Architecture**: BERT encoder + GPT decoder  
**Training Data**: Vector line geometries from geospatial datasets  
**Use Cases**: Line refinement, gap filling, distortion correction, connectivity restoration  
**Input Format**: Sequences of coordinate pairs  
**Output Format**: Refined coordinate sequences  
**License**: MIT  


**Notes**: 
- Maximum sequence length: 512 coordinate pairs
- Coordinate range: [0, 500] 
- Performance may degrade on highly complex multi-line geometries


## Inference

The inference process is designed to be iterative, allowing the model to refine line geometries over multiple passes. Each iteration consists of generating input data from the previous step's output, running the model, and stitching the results back together.

### Step 1: Initial Inference (Iteration 0)

The first step processes an initial input GeoJSON file, breaks it into patches, and runs the model on them.

```bash
python iterative_inference.py \
    --iteration 0 \
    --map_dir ./data/maps \
    --in_geojson_dir ./data/inference_input_data \
    --out_geojson_dir ./inference_output_data \
    --in_geojson_name my_map_processed \
    --map_name my_map \
    --model_version 100 \
    --cuda 0
```

- **Output**: This will create a directory `./inference_output_data/my_map_iter0/` containing the processed results, including `my_map_post.geojson`.

### Step 2: Iterative Refinement (Iteration > 0)

Subsequent iterations use the output of the previous step (`_post.geojson`) as the new input, allowing for progressive refinement.

```bash
python iterative_inference.py \
    --iteration 1 \
    --map_dir ./data/maps \
    --extract_geojson_dir ./data/inference_input_data \
    --out_geojson_dir ./inference_output_data \
    --in_geojson_name my_map_processed \
    --map_name my_map \
    --model_version 100 \
    --cuda 0
```
- **How it works**: For `iteration 1`, the script automatically looks for the output from `iteration 0` (i.e., in `./inference_output_data/my_map_iter0/my_map_post.geojson`) to use as its input.
- **`--extract_geojson_dir`**: This is required for iterative steps to reference the original, unprocessed lines for context.

```
### Parameters Explained
- `--iteration`: The current pass of the inference process (e.g., 0, 1, 2...).
- `--map_dir`: Directory containing the base map image files (`.tif` or `.png`).
- `--in_geojson_dir`: **(Iteration 0 only)** Directory containing the initial input GeoJSON file.
- `--extract_geojson_dir`: **(Iteration > 0)** Directory of the *original* unprocessed GeoJSON, used for context in refinement steps.
- `--out_geojson_dir`: The base directory where all output folders will be created (e.g., `my_map_iter0`, `my_map_iter1`).
- `--in_geojson_name`: The name of the input GeoJSON file (without extension).
- `--map_name`: The base name of the map, used for creating output directories and files.
- `--model_version`: The specific epoch/version of the fine-tuned model to use.
- `--cuda`: The ID of the CUDA device to use for GPU acceleration.




### Inference Data
You can download the GeoJSON files for inference from [this link](https://drive.google.com/drive/folders/1QHs0uDjItz47S_q8X3MoCzeJwuL61lF9?usp=sharing).




## Troubleshooting


### Common Issues


1. **CUDA Out of Memory**: Reduce `batch_size` in the training scripts
2. **File Not Found**: Ensure data files are in the correct `./data/` directory
3. **Token Index Errors**: Verify that coordinate values fall within the valid vocab_size range, i.e., [0, 500]


### Memory Optimization


For limited GPU memory:
- Reduce `batch_size`
- Reduce `hidden_size` 
- Reduce `num_hidden_layers`
- Use gradient accumulation


## License


This project is licensed under the MIT License - see the LICENSE file for details.
