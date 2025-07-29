import torch
from torch.utils.data import Dataset

class TwoDimensionalDatasetWithSEQ(Dataset):
    """
    Dataset for TwoDimensionalBERTTransformer, handling x and y dimensions separately,
    and adding [SEQ] tokens between lists within a single sample.
    """
    def __init__(self, input_sequences, target_sequences, max_len=512, max_id=100):
        """
        Args:
            input_sequences (list of list of list of tuples): Input sequences with multiple lists per sample.
            target_sequences (list of list of list of tuples): Target sequences with multiple lists per sample.
            max_len (int): Maximum sequence length.
            max_id (int): Maximum valid ID for id_x and id_y.
        """
        self.input_sequences = input_sequences
        self.target_sequences = target_sequences
        self.max_len = max_len

        # Assign special token IDs
        self.pad_token_id = max_id + 1
        self.bos_token_id = max_id + 2
        self.eos_token_id = max_id + 3
        self.seq_token_id = max_id + 4
    
    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.input_sequences)

    def pad_and_truncate(self, sequence, is_target=False):
        """
        Pad and truncate a sequence to the maximum length.

        Args:
            sequence (list of tuples): Sequence of (id_x, id_y) pairs.
            is_target (bool): Whether this is a target sequence (add [BOS] and [EOS]).

        Returns:
            tuple: Padded and truncated x and y sequences.
        """
        if is_target:
            sequence = [(self.bos_token_id, self.bos_token_id)] + sequence + [(self.eos_token_id, self.eos_token_id)]

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
        result = []
        for idx, sublist in enumerate(sequence):
            result.extend(sublist)
            if idx < len(sequence) - 1:  # Add [SEQ] token between lists
                result.append((self.seq_token_id, self.seq_token_id))
        return result

    def __getitem__(self, idx):
        """
        Process and return a single data sample.

        Returns:
            dict: A dictionary containing tokenized and padded inputs and targets.
        """
        input_seq = self.insert_seq_tokens(self.input_sequences[idx])
        target_seq = self.insert_seq_tokens(self.target_sequences[idx])

        # Add [BOS] and [EOS] to the target sequence
        target_seq = [(self.bos_token_id, self.bos_token_id)] + target_seq + [(self.eos_token_id, self.eos_token_id)]

        # Pad and truncate sequences
        input_ids_x, input_ids_y = self.pad_and_truncate(input_seq)
        target_ids_x, target_ids_y = self.pad_and_truncate(target_seq, is_target=True)

        # Prepare decoder inputs and labels
        decoder_input_ids_x = target_ids_x[:-1]
        decoder_input_ids_y = target_ids_y[:-1]
        labels_x = target_ids_x[1:]
        labels_y = target_ids_y[1:]

        return {
            "input_ids_x": input_ids_x,  # Encoder input x-dimension
            "input_ids_y": input_ids_y,  # Encoder input y-dimension
            "decoder_input_ids_x": decoder_input_ids_x,  # Decoder input x-dimension
            "decoder_input_ids_y": decoder_input_ids_y,  # Decoder input y-dimension
            "labels_x": labels_x,  # Labels for x-dimension
            "labels_y": labels_y,  # Labels for y-dimension
        }