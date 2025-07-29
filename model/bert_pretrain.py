import torch
from torch import nn
from transformers import BertConfig, BertModel

class MaskedBERT(nn.Module):
    """
    Masked BERT model for two-dimensional sequences (x and y).
    Predicts masked x and y tokens separately.
    """
    def __init__(self, vocab_size, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, max_position_embeddings=512):
        """
        Args:
            vocab_size (int): Vocabulary size for both x and y dimensions.
            hidden_size (int): Hidden size of the BERT model.
            num_hidden_layers (int): Number of hidden layers in the BERT model.
            num_attention_heads (int): Number of attention heads.
            intermediate_size (int): Size of the intermediate (feed-forward) layers.
            max_position_embeddings (int): Maximum sequence length for positional embeddings.
        """
        super(MaskedBERT, self).__init__()

        # Embeddings for x, y, and positional encoding
        self.embedding_x = nn.Embedding(vocab_size, hidden_size // 2)
        self.embedding_y = nn.Embedding(vocab_size, hidden_size // 2)
        self.position_embedding = nn.Embedding(max_position_embeddings, hidden_size)

        # BERT encoder configuration
        config = BertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings
        )
        self.encoder = BertModel(config)

        # Output layers for x and y predictions
        self.output_x = nn.Linear(hidden_size, vocab_size)
        self.output_y = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids_x, input_ids_y, attention_mask):
        """
        Forward pass for the model.

        Args:
            input_ids_x (torch.Tensor): Input IDs for x-dimension (batch_size, seq_len).
            input_ids_y (torch.Tensor): Input IDs for y-dimension (batch_size, seq_len).
            attention_mask (torch.Tensor): Attention mask for the input (batch_size, seq_len).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Logits for x and y predictions.
        """
        batch_size, seq_len = input_ids_x.size()

        # Positional embeddings
        position_ids = torch.arange(seq_len, device=input_ids_x.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embedding(position_ids)

        # Combine embeddings for x and y dimensions
        embeddings_x = self.embedding_x(input_ids_x)  # (batch_size, seq_len, hidden_size // 2)
        embeddings_y = self.embedding_y(input_ids_y)  # (batch_size, seq_len, hidden_size // 2)
        combined_embeddings = torch.cat([embeddings_x, embeddings_y], dim=-1)  # (batch_size, seq_len, hidden_size)

        # Add positional embeddings
        combined_embeddings += position_embeddings
        # Pass through BERT encoder
        encoder_outputs = self.encoder(
            inputs_embeds=combined_embeddings, attention_mask=attention_mask
        )
        hidden_states = encoder_outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

        # Predict x and y logits
        logits_x = self.output_x(hidden_states)  # (batch_size, seq_len, vocab_size)
        logits_y = self.output_y(hidden_states)  # (batch_size, seq_len, vocab_size)
        return logits_x, logits_y