import torch
from torch import nn
from transformers import BertConfig, BertModel
from torch.nn import DataParallel


class TwoDimensionalBERTTransformer(nn.Module):
    """
    A sequence-to-sequence model with TwoDimensionalBERT as the encoder and
    a Transformer-based decoder for autoregressive sequence generation.
    """
    def __init__(self, vocab_size_x, vocab_size_y, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, max_position_embeddings=512, decoder_layers=6, nhead=8):
        super(TwoDimensionalBERTTransformer, self).__init__()

        # Encoder: TwoDimensionalBERT
        self.embedding_x = nn.Embedding(vocab_size_x, hidden_size // 2)
        self.embedding_y = nn.Embedding(vocab_size_y, hidden_size // 2)
        # self.embedding_x = nn.Embedding(vocab_size_x, hidden_size)
        # self.embedding_y = nn.Embedding(vocab_size_y, hidden_size)
        self.position_embedding = nn.Embedding(max_position_embeddings, hidden_size)

        # Adjust the hidden_size to account for concatenation
        config = BertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings
        )
        self.encoder = BertModel(config)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=nhead, dim_feedforward=4 * hidden_size
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)

        # Output layers for id_x and id_y
        self.output_x = nn.Linear(hidden_size, vocab_size_x)
        self.output_y = nn.Linear(hidden_size, vocab_size_y)

    def forward(self, input_ids_x, input_ids_y, attention_mask, decoder_input_ids_x, decoder_input_ids_y):
        """
        Forward pass for the model.

        Args:
            input_ids_x: Token IDs for the x dimension of the input.
            input_ids_y: Token IDs for the y dimension of the input.
            attention_mask: Attention mask for the encoder input.
            decoder_input_ids_x: Token IDs for the x dimension of the target (decoder input).
            decoder_input_ids_y: Token IDs for the y dimension of the target (decoder input).

        Returns:
            Tuple of logits for id_x and id_y predictions.
        """
        batch_size, seq_len = input_ids_x.size()

        # Positional embeddings
        position_ids = torch.arange(seq_len, device=input_ids_x.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embedding(position_ids)

        # Concatenate embeddings for encoder input
        embeddings_x = self.embedding_x(input_ids_x)  # Shape: (batch_size, seq_len, hidden_size // 2)
        embeddings_y = self.embedding_y(input_ids_y)  # Shape: (batch_size, seq_len, hidden_size // 2)
        encoder_embeddings = torch.cat((embeddings_x, embeddings_y), dim=-1) + position_embeddings
        # encoder_embeddings = embeddings_x + embeddings_y + position_embeddings

        # Encode the input sequence
        encoder_outputs = self.encoder(inputs_embeds=encoder_embeddings, attention_mask=attention_mask)
        encoder_hidden_states = encoder_outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)

        # Prepare decoder embeddings
        decoder_position_ids = torch.arange(decoder_input_ids_x.size(1), device=decoder_input_ids_x.device).unsqueeze(0)
        decoder_position_embeddings = self.position_embedding(decoder_position_ids)

        decoder_embeddings_x = self.embedding_x(decoder_input_ids_x)
        decoder_embeddings_y = self.embedding_y(decoder_input_ids_y)
        decoder_embeddings = torch.cat((decoder_embeddings_x, decoder_embeddings_y), dim=-1) + decoder_position_embeddings
        # decoder_embeddings = decoder_embeddings_x + decoder_embeddings_y + decoder_position_embeddings

        # Create causal mask for decoder (autoregressive)
        tgt_seq_len = decoder_embeddings.size(1)
        causal_mask = torch.triu(
            torch.ones(tgt_seq_len, tgt_seq_len, device=decoder_embeddings.device), diagonal=1
        ).bool()

        # Decode the target sequence
        decoder_outputs = self.decoder(
            tgt=decoder_embeddings.permute(1, 0, 2),  # (target_seq_len, batch_size, hidden_size)
            memory=encoder_hidden_states.permute(1, 0, 2),  # (input_seq_len, batch_size, hidden_size)
            tgt_mask=causal_mask
        )

        decoder_hidden_states = decoder_outputs.permute(1, 0, 2)  # (batch_size, target_seq_len, hidden_size)

        # Project decoder outputs to vocab sizes
        logits_x = self.output_x(decoder_hidden_states)  # (batch_size, target_seq_len, vocab_size_x)
        logits_y = self.output_y(decoder_hidden_states)  # (batch_size, target_seq_len, vocab_size_y)

        return logits_x, logits_y
    
def load_pretrained_encoder(model, pretrained_state_dict_path, gpu_count=1):
    """
    Load the pretrained encoder into the model, handling both single and multi-GPU cases.

    Args:
        model (torch.nn.Module): The model instance containing the encoder.
        pretrained_state_dict_path (str): Path to the pretrained state_dict for the encoder.
        gpu_count (int): Number of GPUs available for training.

    Returns:
        torch.nn.Module: The model with the pretrained encoder loaded.
    """
    # Load pretrained state_dict
    pretrained_state_dict = torch.load(pretrained_state_dict_path, map_location="cpu")

    # Check if using multiple GPUs
    if gpu_count > 1 and isinstance(model, DataParallel):
        print(f"Using {gpu_count} GPUs with DataParallel")

        # Load state_dict into the encoder of the wrapped model
        model.module.encoder.load_state_dict(pretrained_state_dict, strict=False)
    else:
        print("Using single GPU or CPU")
        
        # Load state_dict into the encoder directly
        model.encoder.load_state_dict(pretrained_state_dict, strict=False)

    return model
    
#     # Check if we are using multiple GPUs
#     if gpu_count > 1:
#         print(f"Using {gpu_count} GPUs with DataParallel")

#         # Wrap model in DataParallel
#         model = DataParallel(model)
        
#         # Modify state_dict keys to remove "module." prefix if necessary
#         new_state_dict = {}
#         for k, v in pretrained_state_dict.items():
#             if k.startswith("module."):
#                 new_state_dict[k[7:]] = v  # Remove "module."
#             else:
#                 new_state_dict[k] = v
        
#         # Load the modified state_dict into the encoder
#         model.module.encoder.load_state_dict(new_state_dict, strict=False)

#     else:
#         print("Using single GPU or CPU")

        # Load state_dict directly for a single GPU setup
    model.encoder.load_state_dict(pretrained_state_dict, strict=False)

    return model
