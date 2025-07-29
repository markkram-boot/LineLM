import torch
from torch.utils.data import Dataset
import random

class TwoDimensionalDatasetWithSEQForMLM(Dataset):
    """
    Dataset for TwoDimensionalBERTTransformer, handling x and y dimensions separately,
    and adding [SEQ] tokens between lists within a single sample, with masking for MLM.
    """
    def __init__(self, input_sequences, max_len=512, max_id=100, mask_prob=0.15):
        """
        Args:
            input_sequences (list of list of list of tuples): Input sequences with multiple lists per sample.
            max_len (int): Maximum sequence length.
            max_id (int): Maximum valid ID for id_x and id_y.
            mask_prob (float): Probability of masking tokens for MLM.
        """
        self.input_sequences = input_sequences
        self.max_len = max_len
        self.mask_prob = mask_prob

        # Assign special token IDs
        self.pad_token_id = max_id + 1
        self.bos_token_id = max_id + 2
        self.eos_token_id = max_id + 3
        self.seq_token_id = max_id + 4
        self.mask_token_id = max_id + 5

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.input_sequences)

    def pad_and_truncate(self, sequence):
        """
        Pad and truncate a sequence to the maximum length.

        Args:
            sequence (list of tuples): Sequence of (id_x, id_y) pairs.

        Returns:
            tuple: Padded and truncated x and y sequences.
        """
        seq_x = [x for x, _ in sequence]
        seq_y = [y for _, y in sequence]

        # Pad/truncate sequences
        seq_x = seq_x[:self.max_len] + [self.pad_token_id] * max(0, self.max_len - len(seq_x))
        seq_y = seq_y[:self.max_len] + [self.pad_token_id] * max(0, self.max_len - len(seq_y))

        return torch.tensor(seq_x, dtype=torch.long), torch.tensor(seq_y, dtype=torch.long)

    def insert_seq_tokens(self, sequence):
        """
        Insert [SEQ] tokens between lists in a sequence.

        Args:
            sequence (list of list of tuples): Sequence with multiple lists.

        Returns:
            list of tuples: Sequence with [SEQ] tokens added.
        """
        result = sequence
#         for idx, sublist in enumerate(sequence):
#             result.extend(sublist)
#             if idx < len(sequence) - 1:  # Add [SEQ] token between lists
#                 result.append((self.seq_token_id, self.seq_token_id))
        return result

    def mask_tokens(self, sequence):
        """
        Randomly mask tokens in the sequence for MLM.

        Args:
            sequence (list of tuples): Sequence of (id_x, id_y) pairs.

        Returns:
            list of tuples: Sequence with randomly masked tokens.
        """
        masked_sequence = []
        for x, y in sequence:
            if random.random() < self.mask_prob:
                masked_sequence.append((self.mask_token_id, self.mask_token_id))  # Mask both x and y
            else:
                masked_sequence.append((x, y))
        return masked_sequence

    def __getitem__(self, idx):
        """
        Process and return a single data sample.

        Returns:
            dict: A dictionary containing tokenized, masked, and padded inputs.
        """
        input_seq = self.insert_seq_tokens(self.input_sequences[idx])

        # Mask tokens in the input sequence
        masked_input_seq = self.mask_tokens(input_seq)

        # Pad and truncate sequences
        input_ids_x, input_ids_y = self.pad_and_truncate(masked_input_seq)
        labels_x, labels_y = self.pad_and_truncate(input_seq)  # Use unmasked input as labels

        return {
            "input_ids_x": input_ids_x,  # Input x-dimension
            "input_ids_y": input_ids_y,  # Input y-dimension
            "labels_x": labels_x,  # Labels for x-dimension
            "labels_y": labels_y,  # Labels for y-dimension
        }