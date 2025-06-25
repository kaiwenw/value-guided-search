import math
import torch
from torch.utils.data import Dataset, Sampler
from typing import Iterator
import random
from collections import defaultdict
import inspect
from tqdm import tqdm


class FlattenedDataset(Dataset):
    """
    Wraps a grouped dataset where each sample (row) is a dict containing:
      - Grouped columns: keys that map to lists of values which need to be flattened.
      - Shared columns: keys that remain the same for all flattened entries from that row.

    The flattened view is built such that each new sample is a dict combining:
      - The i-th element from each grouped column.
      - The original value for each shared column.

    Attributes:
        flattened_data: A list of dicts, each representing one flattened sample.
        group_boundaries: A list of tuples (start_idx, end_idx) where each tuple indicates
                          the contiguous range in flattened_data corresponding to each
                          original row.
    """
    def __init__(self, grouped_dataset, grouped_keys: list[str], shared_keys: list[str], big_group_size: int, small_group_size: int):
        """
        Args:
            grouped_dataset: An iterable of dicts, where each dict represents a row.
            grouped_keys: A list of keys whose values are lists to be flattened.
            shared_keys: A list of keys whose values are to be shared across the flattened entries.
        """
        assert big_group_size % small_group_size == 0, f"{big_group_size=}, {small_group_size=}"
        self.grouped_dataset = grouped_dataset
        self.grouped_keys = grouped_keys
        self.shared_keys = shared_keys
        self.big_group_size = big_group_size
        self.small_group_size = small_group_size
        current_index = 0
        self.group_boundaries = []  # one tuple per row in grouped_dataset
        num_small_groups = len(grouped_dataset) * big_group_size // small_group_size
        for i in tqdm(range(num_small_groups), desc="Flattening dataset"):
            # for k in grouped_keys:
            #     assert len(grouped_dataset[i][k]) == num_elems_per_group, f"{i=},{k=}: Expected {num_elems_per_group} elements, got {len(grouped_dataset[i][k])}."
            start = current_index
            current_index += small_group_size
            self.group_boundaries.append((start, current_index))
        self.length = current_index
        assert len(grouped_dataset) * big_group_size == self.length, f"{len(grouped_dataset)=}, {big_group_size=}, {self.length=}"

    def __getitem__(self, idx):
        big_group_idx = idx // self.big_group_size
        elem_idx = idx % self.big_group_size
        sample = {}
        for k in self.shared_keys:
            sample[k] = self.grouped_dataset[big_group_idx][k]
        for k in self.grouped_keys:
            sample[k] = self.grouped_dataset[big_group_idx][k][elem_idx]
        return sample

    def __len__(self):
        return self.length


class EndlessSampler(Sampler):
    """
    An endless sampler for a FlattenedDataset that shuffles by groups.

    The sampler works as follows:
      1. It shuffles the groups (as given by flattened_dataset.group_boundaries).
      2. It concatenates the indices from each group (keeping each group's indices consecutive).
      3. It yields batches of indices such that each batch has exactly `batch_size` elements.
         If the last batch is incomplete, it is dropped.
      4. A group may be split across batches if its number of elements exceeds batch_size.

    Attributes:
        flattened_dataset: The dataset with a group_boundaries attribute.
        batch_size: The number of elements per batch.
        group_boundaries: A list of tuples (start, end) indicating the contiguous range
                          in flattened_dataset.flattened_data for each original row.
    """
    def __init__(self, flattened_dataset, batch_size, process_rank: int = 0, num_processes: int = 1, shuffle=True, endless=True):
        """
        Args:
            flattened_dataset: An instance of FlattenedDataset.
            batch_size (int): The number of flattened elements per batch.
            seed (int, optional): A random seed for reproducibility.
        """
        self.flattened_dataset = flattened_dataset
        self.batch_size = batch_size
        self.group_boundaries = flattened_dataset.group_boundaries
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.shuffle = shuffle
        self.endless = endless
        self.epoch = 0
        self.current_position = self.batch_size * self.process_rank
        self.group_indices = list(range(len(self.group_boundaries)))

    def __iter__(self) -> Iterator[list[int]]:
        while True:
            # Shuffle the order of groups.
            group_indices = list(self.group_indices)
            if self.shuffle:
                # same shuffle seed for each process
                # ensures deterministic replay when resuming from preemption
                rng = random.Random((self.epoch+1) * 1337)
                rng.shuffle(group_indices)

            # Concatenate the flattened indices from each shuffled group.
            indices_order = []
            for group_idx in group_indices:
                start, end = self.group_boundaries[group_idx]
                indices_order.extend(range(start, end))
            assert len(indices_order) == len(self.flattened_dataset)

            # Drop the last incomplete batch.
            shift_by = self.batch_size * self.num_processes
            num_batches = len(indices_order) // shift_by

            # Yield batches of exactly batch_size elements.
            for _ in range(num_batches):
                yield indices_order[self.current_position : self.current_position + self.batch_size]
                self.current_position += shift_by

            # advance epoch and restart
            self.epoch += 1
            self.current_position = self.batch_size * self.process_rank

            if not self.endless:
                break

    def __len__(self):
        # The length is defined as the number of complete batches available in one epoch.
        total_elements = len(self.flattened_dataset)
        return total_elements // (self.batch_size * self.num_processes)


class RollInOutCollator:
    """
    Collate function that combines 'roll_in_ids' and 'roll_out_ids' from each sample,
    pads the sequences, creates corresponding attention and roll_out masks, and also
    copies over all additional scalar keys (e.g., "reward") as tensors.

    Attributes:
        tokenizer: A tokenizer with a `pad_token_id` attribute.
        roll_in_key: The key in the sample dict for the roll-in token list.
        roll_out_key: The key in the sample dict for the roll-out token list.
    """
    def __init__(self, tokenizer, roll_in_key, roll_out_key, max_length: int | None = None, pad_multiple: int | None = None):
        """
        Args:
            tokenizer: A tokenizer instance with a `pad_token_id` attribute.
            roll_in_key (str): Key for the roll-in tokens.
            roll_out_key (str): Key for the roll-out tokens.
        """
        self.tokenizer = tokenizer
        self.roll_in_key = roll_in_key
        self.roll_out_key = roll_out_key
        self.max_length = max_length
        self.pad_multiple = pad_multiple

    def __call__(self, batch):
        """
        Processes a list of samples (each a dict) into a batch.

        For each sample:
          - Concatenates the roll_in_ids and roll_out_ids.
          - Creates a roll_out_mask: 0 for tokens from roll_in_ids, 1 for tokens from roll_out_ids.
          - Pads the sequences on the right using tokenizer.pad_token_id.
          - Creates an attention_mask where non-pad tokens have value 1 and pad tokens have value 0.
          - Copies over any additional scalar keys into the batch as lists and converts them to tensors.

        Returns:
            A dict with keys:
              - "input_ids": Tensor of padded concatenated sequences.
              - "attention_mask": Tensor indicating non-pad tokens.
              - "roll_out_mask": Tensor indicating which tokens are from roll_out_ids.
              - Any additional keys present in the original samples, collated as tensors.
        """
        batch_input_ids = []
        batch_roll_out_masks = []
        other_keys = defaultdict(list)

        # Process each sample. Left pad roll_in.
        for sample in batch:
            roll_in = sample[self.roll_in_key]
            roll_out = sample[self.roll_out_key]
            # Concatenate the roll_in and roll_out tokens.
            combined = roll_in + roll_out
            # Create roll_out_mask: 0 for roll_in tokens, 1 for roll_out tokens.
            roll_out_mask = [0] * len(roll_in) + [1] * len(roll_out)
            if self.max_length is not None and len(combined) > self.max_length:
                combined = combined[:self.max_length]
                roll_out_mask = roll_out_mask[:self.max_length]

            batch_input_ids.append(combined)
            batch_roll_out_masks.append(roll_out_mask)

            # Copy over additional scalar keys.
            for key, value in sample.items():
                if key not in (self.roll_in_key, self.roll_out_key):
                    other_keys[key].append(value)

        # Determine maximum sequence length in the batch for padding.
        max_len = max(len(seq) for seq in batch_input_ids)
        if self.pad_multiple is not None:
            max_len = int(math.ceil(max_len / self.pad_multiple)) * self.pad_multiple
            assert max_len <= self.max_length, f"{max_len=}, {self.max_length=}"

        padded_input_ids = []
        padded_attention_masks = []
        padded_roll_out_masks = []

        # Right pad sequences and corresponding masks.
        for seq, rmask in zip(batch_input_ids, batch_roll_out_masks):
            seq_len = len(seq)
            pad_length = max_len - seq_len
            padded_seq = seq + [self.tokenizer.pad_token_id] * pad_length
            padded_rmask = rmask + [0] * pad_length
            attention_mask = [1] * seq_len + [0] * pad_length

            padded_input_ids.append(padded_seq)
            padded_attention_masks.append(attention_mask)
            padded_roll_out_masks.append(padded_rmask)

        # Build final output dictionary.
        output = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            # attention_mask includes roll_in and roll_out
            "attention_mask": torch.tensor(padded_attention_masks, dtype=torch.long),
            # roll_out_mask only includes roll_out
            "roll_out_mask": torch.tensor(padded_roll_out_masks, dtype=torch.long)
        }
        # Add additional keys as tensors.
        for key, values in other_keys.items():
            output[key] = torch.tensor(values)
        return output


def configure_optimizer(model, lr, weight_decay, betas, device_type):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas, eps=1e-6, **extra_args)
    print(f"using AdamW fused: {use_fused} | lr: {lr} | betas: {betas}")
    return optimizer


def get_lr(it, max_lr, warmup_iters, lr_decay_iters, min_lr):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return max_lr * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + coeff * (max_lr - min_lr)


class DistributedMetrics:
    @staticmethod
    def local_compute(predictions: torch.Tensor, labels: torch.Tensor) -> dict:
        """
        Compute local partial statistics on the current GPU.

        Args:
            predictions (torch.Tensor): Local predictions.
            labels (torch.Tensor): Local ground truth labels.

        Returns:
            dict: Contains local statistics:
                - "count": number of samples (scalar tensor)
                - "labels_sum": sum of labels
                - "labels_sum_sq": sum of squared labels
                - "error_sum": sum of errors (predictions - labels)
                - "error_sum_sq": sum of squared errors
        """
        assert predictions.shape == labels.shape
        # assume labels are binary
        labels = labels.to(torch.int64)
        assert ((labels == 0) | (labels == 1)).all(), f"{labels=}"

        count = torch.tensor(labels.numel(), device=labels.device, dtype=torch.int64)
        labels_sum = torch.sum(labels).to(torch.int64)
        labels_sum_sq = torch.sum(labels ** 2).to(torch.int64)

        error = predictions - labels
        error_sum = torch.sum(error).to(torch.float64)
        error_sum_sq = torch.sum(error ** 2).to(torch.float64)

        # Compute confusion matrix components
        pred_labels = torch.round(predictions).to(torch.int64)
        tp = torch.sum((pred_labels == 1) & (labels == 1)).to(torch.int64)
        fp = torch.sum((pred_labels == 1) & (labels == 0)).to(torch.int64)
        tn = torch.sum((pred_labels == 0) & (labels == 0)).to(torch.int64)
        fn = torch.sum((pred_labels == 0) & (labels == 1)).to(torch.int64)
        return {
            "count": count,
            "labels_sum": labels_sum,
            "labels_sum_sq": labels_sum_sq,
            "error_sum": error_sum,
            "error_sum_sq": error_sum_sq,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }

    @staticmethod
    def global_combine(reduced_stats: dict) -> dict:
        """
        Computes the global regression metrics from reduced statistics.
        The expected keys in reduced_stats are:
            - "count", "labels_sum", "labels_sum_sq", "error_sum", "error_sum_sq".

        Returns a dictionary with:
            - "explained_var": 1 - Var(error)/Var(labels)
            - "R2": 1 - sum((labels - predictions)**2) / sum((labels - labels.mean())**2)
            - "MSE": Mean squared error.
        """
        global_count = reduced_stats["count"]
        global_labels_sum = reduced_stats["labels_sum"]
        global_labels_sum_sq = reduced_stats["labels_sum_sq"]
        global_error_sum = reduced_stats["error_sum"]
        global_error_sum_sq = reduced_stats["error_sum_sq"]
        tp = reduced_stats["tp"]
        fp = reduced_stats["fp"]
        tn = reduced_stats["tn"]
        fn = reduced_stats["fn"]
        assert global_count == tp + fp + tn + fn, f"{global_count=}, {tp=}, {fp=}, {tn=}, {fn=}"

        # Compute global means.
        global_labels_mean = global_labels_sum / global_count
        global_error_mean = global_error_sum / global_count

        # Variance: E[x^2] - (E[x])^2.
        global_labels_var = global_labels_sum_sq / global_count - global_labels_mean ** 2
        global_error_var = global_error_sum_sq / global_count - global_error_mean ** 2

        # Compute explained_var: 1 - Var(error)/Var(labels).
        if global_labels_var.item() == 0:
            explained_var = float('nan')
        else:
            explained_var = 1.0 - (global_error_var / global_labels_var).item()

        # Mean Squared Error.
        mse = (global_error_sum_sq / global_count).item()

        # R2: 1 - SSE/TSS, where TSS = n * Var(labels)
        total_labels_ss = global_count * global_labels_var
        if total_labels_ss.item() == 0:
            r2 = float('nan')
        else:
            r2 = 1.0 - (global_error_sum_sq / total_labels_ss).item()

        # Calculate accuracy
        accuracy = ((tp + tn) / global_count).item()

        # Precision: TP / (TP + FP)
        if (tp + fp).item() == 0:
            precision = float('nan')
        else:
            precision = (tp / (tp + fp)).item()

        # Recall/Sensitivity: TP / (TP + FN)
        if (tp + fn).item() == 0:
            recall = float('nan')
        else:
            recall = (tp / (tp + fn)).item()

        # Specificity: TN / (TN + FP)
        if (tn + fp).item() == 0:
            specificity = float('nan')
        else:
            specificity = (tn / (tn + fp)).item()

        # F1 Score: 2 * precision * recall / (precision + recall)
        if precision == float('nan') or recall == float('nan') or (precision + recall) == 0:
            f1 = float('nan')
        else:
            f1 = 2 * precision * recall / (precision + recall)

        # Area Under ROC Curve (approximation using sensitivity and specificity)
        # This is a simplified approximation, not a true AUC
        if recall == float('nan') or specificity == float('nan'):
            auc_approx = float('nan')
        else:
            auc_approx = (recall + specificity) / 2
        return {
            "n_tokens": global_count,
            "ExpVar": explained_var,
            "R2": r2,
            "MSE": mse,
            "Acc": accuracy,
            "Precision": precision,
            "Recall": recall,
            "Specificity": specificity,
            "F1": f1,
            "ApxAUC": auc_approx,
            "TP": tp.item(),
            "FP": fp.item(),
            "TN": tn.item(),
            "FN": fn.item(),
        }


def create_pos_mask(mask, start_pos, end_pos):
    """
    Create a position mask based on the cumulative count of 1s in the input mask.

    Args:
        mask (torch.Tensor): Binary mask of shape (batch_size, seq_len)
        start_pos (int): Minimum number of 1s required (inclusive)
        end_pos (int): Maximum number of 1s allowed (exclusive)

    Returns:
        torch.Tensor: Position mask of the same shape as input mask
    """
    # Get cumulative sum of 1s along the sequence dimension
    cumsum = torch.cumsum(mask, dim=1)

    # Create the position mask based on cumulative counts
    # 1 if:
    # - the original mask position is 1 AND
    # - cumsum is at least start_pos AND
    # - cumsum is less than end_pos
    pos_mask = mask * (cumsum >= start_pos) * (cumsum < end_pos)

    return pos_mask