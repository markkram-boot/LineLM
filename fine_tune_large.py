from model.dataloader import TwoDimensionalDatasetWithSEQ
from model.bert import TwoDimensionalBERTTransformer
from torch.utils.data import DataLoader, Dataset
from utils import load_linestring_from_geojson_for_finetune
from transformers import AdamW, get_scheduler
import torch
import torch.nn as nn
from tqdm import tqdm
import os
from torch.nn import DataParallel
from collections import OrderedDict


# Set specific GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


# Create dataset
max_len = 130 
max_id = 500
epochs = 200
batch_size = 70


# Load Dataset and DataLoader
input_file = "./data/fine_tune/input_line_finetune.geojson"
target_file = "./data/fine_tune/target_line_finetune.geojson"
pretrained_weights = None
s_epoch = 0
saved_dir = "./trained_weights/fine_tune_large"
os.makedirs(saved_dir, exist_ok=True)


encoder_pretrained_weights ="./trained_weights/LineLM_pretrain.pth"


input_sequences, _, _ = load_linestring_from_geojson_for_finetune(input_file)
target_sequences, _, _ = load_linestring_from_geojson_for_finetune(target_file)


dataset = TwoDimensionalDatasetWithSEQ(input_sequences, target_sequences, max_len, max_id)


# Create DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Model Initialization
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




# Check if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = DataParallel(model)
# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device) 


if pretrained_weights is not None:
    print('Resume the fine-tune')
    if torch.cuda.device_count() > 1:
        state_dict = torch.load(pretrained_weights)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = "module." + k  # Add 'module.' to key
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(torch.load(pretrained_weights))


# Load pretrained weights into the encoder only
elif encoder_pretrained_weights is not None:
    print('Load encoder pretrained weights')
    state_dict = torch.load(encoder_pretrained_weights, map_location='cpu')
    encoder_dict = {k: v for k, v in state_dict.items() if 'encoder' in k}
    if torch.cuda.device_count() > 1:
        new_state_dict = OrderedDict()
        for k, v in encoder_dict.items():
            new_key = 'module.' + k if not k.startswith('module.') else k
            new_state_dict[new_key] = v
        model.module.encoder.load_state_dict(new_state_dict, strict=False)
    else:
        model.encoder.load_state_dict(encoder_dict, strict=False)


encoder_params = model.module.encoder.parameters()
decoder_params = list(model.module.decoder.parameters())\
                + list(model.module.output_x.parameters()) + list(model.module.output_y.parameters())


# Set up separate parameter groups
param_groups = [
    {'params': encoder_params, 'lr': 5e-6},
    {'params': decoder_params, 'lr': 5e-5}
]


# Initialize the optimizer
optimizer = torch.optim.Adam(param_groups)


loss_fn = nn.CrossEntropyLoss(ignore_index=max_id+1)


# Define tokenizer with special token IDs
tokenizer = {
    "pad_token_id": max_id + 1,
    "bos_token_id": max_id + 2,
    "eos_token_id": max_id + 3,
    "seq_token_id": max_id + 4,
}


for epoch in range(s_epoch, epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
        optimizer.zero_grad()
        input_ids_x = batch["input_ids_x"].to(device)
        input_ids_y = batch["input_ids_y"].to(device)
        decoder_input_ids_x = batch["decoder_input_ids_x"].to(device)
        decoder_input_ids_y = batch["decoder_input_ids_y"].to(device)
        labels_x = batch["labels_x"].to(device)
        labels_y = batch["labels_y"].to(device)
#         print(input_ids_x.shape, labels_x.shape, decoder_input_ids_x.shape)
#         print(input_ids_y.shape, labels_y.shape, decoder_input_ids_y.shape)


        if isinstance(model, torch.nn.DataParallel):
            # Check for out-of-bound token IDs
            vocab_size_x = model.module.embedding_x.num_embeddings
            vocab_size_y = model.module.embedding_y.num_embeddings
        else:
            # Check for out-of-bound token IDs
            vocab_size_x = model.embedding_x.num_embeddings
            vocab_size_y = model.embedding_y.num_embeddings


        invalid_x = (input_ids_x >= vocab_size_x).any().item()
        invalid_y = (input_ids_y >= vocab_size_y).any().item()


        if invalid_x:
            print("Error: input_ids_x contains out-of-bound indices")
        if invalid_y:
            print("Error: input_ids_y contains out-of-bound indices")
        
        # Debugging checks
        assert input_ids_x.dim() == 2, f"Expected 2D tensor for input_ids_x, got {input_ids_x.dim()}D"
        assert input_ids_y.dim() == 2, f"Expected 2D tensor for input_ids_y, got {input_ids_y.dim()}D"
        assert labels_x.max().item() < vocab_size_x, f"labels_x out of range: {labels_x.max().item()}"
        assert labels_y.max().item() < vocab_size_y, f"labels_y out of range: {labels_y.max().item()}"
        
        logits_x, logits_y = model(
            input_ids_x=input_ids_x,
            input_ids_y=input_ids_y,
            attention_mask=None,
            decoder_input_ids_x=decoder_input_ids_x,
            decoder_input_ids_y=decoder_input_ids_y,
        )


        logits_x = logits_x.view(-1, logits_x.size(-1))
        logits_y = logits_y.view(-1, logits_y.size(-1))
        labels_x = labels_x.view(-1)
        labels_y = labels_y.view(-1)
        
        # Get predictions from logits
        predicted_ids_x = torch.argmax(logits_x, dim=-1)  # (batch_size * seq_len)
        predicted_ids_y = torch.argmax(logits_y, dim=-1)  # (batch_size * seq_len)


        # Example: Calculate accuracy (optional for debugging)
        correct_x = (predicted_ids_x == labels_x).sum().item()
        correct_y = (predicted_ids_y == labels_y).sum().item()
        total_x = labels_x.ne(tokenizer["pad_token_id"]).sum().item()  # Exclude padding
        total_y = labels_y.ne(tokenizer["pad_token_id"]).sum().item()  # Exclude padding


        accuracy_x = correct_x / total_x if total_x > 0 else 0
        accuracy_y = correct_y / total_y if total_y > 0 else 0
#         print(logits_x.shape, labels_x.shape, input_ids_x.shape)
        loss_x = loss_fn(logits_x, labels_x)
        loss_y = loss_fn(logits_y, labels_y)
        loss = loss_x + loss_y
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}: Loss = {total_loss / len(dataloader)}, Accuracy X: {accuracy_x:.4f}, Accuracy Y: {accuracy_y:.4f}")
    if (epoch+1)%1 == 0:
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), f"{saved_dir}/LineLM_fine_tune_e{epoch+1}.pth")
        else:
            torch.save(model.state_dict(), f"{saved_dir}/LineLM_fine_tune_e{epoch+1}.pth")
        print("Model training complete and saved.")
