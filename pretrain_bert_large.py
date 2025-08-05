import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm import tqdm
from model.dataloader_mlm import TwoDimensionalDatasetWithSEQForMLM
from model.bert_pretrain import MaskedBERT
from utils import load_linestring_from_geojson_for_pretrain


input_file = "./data/pretrain/target_line_pretrain.geojson"
input_sequences = load_linestring_from_geojson_for_pretrain(input_file)
vocab_size = 500 
max_len = 130
saved_model_dir = './trained_weights/pretrain_trainset_large'
os.makedirs(saved_model_dir, exist_ok=True)
print(input_sequences[0])


# Define dataset
dataset = TwoDimensionalDatasetWithSEQForMLM(input_sequences, max_len=max_len, max_id=vocab_size, mask_prob=0.15)


# Define dataloader
dataloader = DataLoader(dataset, batch_size=260, shuffle=True)


# Initialize MaskedBERT model
model = MaskedBERT(
    vocab_size=vocab_size+6, # max_id + 5 (for PAD, BOS, EOS, SEQ, MASK tokens)
    hidden_size=1024,
    num_hidden_layers=8,  # Adjust as needed
    num_attention_heads=16,  # Adjust as needed
    intermediate_size=6*1024, #4 * hidden_size,
    max_position_embeddings=max_len
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for training!")
    model = torch.nn.DataParallel(model)  # Enable multi-GPU
model.to(device)




# Define optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = len(dataloader) * 50  # 50 epochs
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)


# Define loss function
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=dataset.pad_token_id)


# Training loop
epochs = 400
for epoch in range(epochs):
    model.train()
    total_loss = 0
    total_correct_x, total_correct_y = 0, 0
    total_tokens_x, total_tokens_y = 0, 0
    
    # Loop through batches
    for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
        input_ids_x = batch["input_ids_x"].to(device)
        input_ids_y = batch["input_ids_y"].to(device)
        labels_x = batch["labels_x"].to(device)
        labels_y = batch["labels_y"].to(device)
#         # Check input IDs for invalid indices
#         print(f"Max input_ids_x: {input_ids_x.max().item()}, Num embeddings: {model.embedding_x.num_embeddings}")
#         print(f"Max input_ids_y: {input_ids_y.max().item()}, Num embeddings: {model.embedding_y.num_embeddings}")
        actual_model = model.module if torch.cuda.device_count() > 1 else model


        # Get vocab sizes
        vocab_size_x = actual_model.embedding_x.num_embeddings
        vocab_size_y = actual_model.embedding_y.num_embeddings


        # Check for invalid token indices
        invalid_x = (input_ids_x >= vocab_size_x).any().item()
        invalid_y = (input_ids_y >= vocab_size_y).any().item()
        
        # Debugging checks
        assert input_ids_x.dim() == 2, f"Expected 2D tensor for input_ids_x, got {input_ids_x.dim()}D"
        assert input_ids_y.dim() == 2, f"Expected 2D tensor for input_ids_y, got {input_ids_y.dim()}D"
        assert labels_x.max().item() < vocab_size_x, f"labels_x out of range: {labels_x.max().item()}"
        assert labels_y.max().item() < vocab_size_y, f"labels_y out of range: {labels_y.max().item()}"
        # Forward pass
        optimizer.zero_grad()
        logits_x, logits_y = \
            model(input_ids_x=input_ids_x, input_ids_y=input_ids_y, attention_mask=(input_ids_x != dataset.pad_token_id))


         # Reshape logits and labels for loss computation
        logits_x = logits_x.view(-1, logits_x.size(-1))
        logits_y = logits_y.view(-1, logits_y.size(-1))
        labels_x = labels_x.view(-1)
        labels_y = labels_y.view(-1)


        # Compute loss
        loss_x = loss_fn(logits_x, labels_x)
        loss_y = loss_fn(logits_y, labels_y)
        loss = (loss_x + loss_y) / 2


        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        total_loss += loss.item()
        
        # Compute accuracy
        preds_x = logits_x.argmax(dim=-1)
        preds_y = logits_y.argmax(dim=-1)


        mask_x = labels_x != dataset.pad_token_id
        mask_y = labels_y != dataset.pad_token_id


        total_correct_x += (preds_x == labels_x).masked_select(mask_x).sum().item()
        total_correct_y += (preds_y == labels_y).masked_select(mask_y).sum().item()


        total_tokens_x += mask_x.sum().item()
        total_tokens_y += mask_y.sum().item()


    avg_loss = total_loss / len(dataloader)
    accuracy_x = total_correct_x / total_tokens_x if total_tokens_x > 0 else 0
    accuracy_y = total_correct_y / total_tokens_y if total_tokens_y > 0 else 0


    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy X: {accuracy_x:.4f}, Accuracy Y: {accuracy_y:.4f}")
    
    if (epoch+1)%20 == 0:
        save_path = f"{saved_model_dir}/bert_pretrain_e{epoch+1}.pth"
        torch.save(model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(), save_path)
        print(f"Model saved at {save_path}")
