
from torch.utils.data import Dataset

class TaxonomyDataset(Dataset):
    def __init__(self, X, X_taxonomy, df_taxonomy, taxonomy_list, max_taxa=104, return_abundances=False):
        """
        Args:
            X (scipy.sparse matrix): Matrix of OTUs (num_samples x num_otus).
            X_taxonomy (scipy.sparse matrix): Matrix of taxonomies (num_samples x num_taxonomies).
            df_taxonomy (pandas.DataFrame): Contains mapping from OTU indices to taxonomy strings.
            taxonomy_list (list): Unique taxonomy strings for columns in X_taxonomy.
            max_taxa (int): Fixed length to pad/truncate each sample's taxonomy list.
        """
        super().__init__()
        self.X = X
        self.X_taxonomy = X_taxonomy
        self.df_taxonomy = df_taxonomy
        self.taxonomy_list = taxonomy_list
        self.max_taxa = max_taxa
        self.num_samples = X.shape[0]
        self.return_abundances = return_abundances


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns:
            A single list of taxonomies of length 'max_taxa', sorted by abundance (descending),
            padded (with '<pad>') or truncated if necessary.
        """
        # 1) Identify non-zero taxonomy indices + values
        nonzero_tax_indices = self.X_taxonomy[idx].nonzero()[1]
        nonzero_tax_values  = self.X_taxonomy[idx, nonzero_tax_indices].toarray().flatten()
        taxonomies = [self.taxonomy_list[i] for i in nonzero_tax_indices]

        # 2) Pair each taxonomy with its abundance, then sort descending
        tax_val_pairs = list(zip(taxonomies, nonzero_tax_values))
        tax_val_pairs_sorted = sorted(tax_val_pairs, key=lambda x: x[1], reverse=True)

        # 3) Keep only the sorted taxonomy names
        sorted_taxonomies = [tv[0] for tv in tax_val_pairs_sorted]
        sorted_abundances = [tv[1] for tv in tax_val_pairs_sorted]

		# 4) Pad or truncate to 'max_taxa'
        cur_len = len(sorted_taxonomies)
        if cur_len < self.max_taxa:
            pad_len = self.max_taxa - cur_len
            sorted_taxonomies += ["<pad>"] * pad_len
            sorted_abundances += [0.0] * pad_len
        elif cur_len > self.max_taxa:
            sorted_taxonomies = sorted_taxonomies[:self.max_taxa]
            sorted_abundances = sorted_abundances[:self.max_taxa]

        # 5) Return depending on 'return_abundances'
        if self.return_abundances:
            return {
                "taxonomies": sorted_taxonomies,
                "abundances": sorted_abundances,
            }
        else:
            return sorted_taxonomies
def no_collation(batch):
    # 'batch' will be a list of length = batch_size
    # each element is exactly what your Dataset's __getitem__ returned
    # i.e. a list of (up to) 104 strings.
    # Just return it directly:
    return batch


#### Colapse or truncate tree in a random level

def truncate_taxonomy_up_to_rank(taxonomy_str, rank_code="f"):
    """
    Given a full taxonomy like:
      d__Bacteria;p__Firmicutes;c__Bacilli;o__Haloplasmatales;f__Turicibacteraceae;g__Turicibacter;s__Turicibacter
    and a rank_code like 'f', return:
      'd__Bacteria;p__Firmicutes;c__Bacilli;o__Haloplasmatales;f__Turicibacteraceae'
    If the rank_code is not found, return the original string.
    """
    parts = taxonomy_str.split(';')
    # parts = [
    #   'd__Bacteria', 'p__Firmicutes', 'c__Bacilli', 
    #   'o__Haloplasmatales', 'f__Turicibacteraceae', 
    #   'g__Turicibacter', 's__Turicibacter'
    # ]

    # We look for the first part that starts with e.g. 'f__'
    target_prefix = rank_code + "__"

    truncated_parts = []
    for p in parts:
        truncated_parts.append(p)
        if p.startswith(target_prefix):
            # Stop once we've included the target rank
            break

    return ";".join(truncated_parts)


def collapse_taxa_up_to_rank(taxonomies, abundances, possible_ranks=("p","c","o","f","g","s"), randomize=True):
    """
    taxonomies: list of full taxonomy strings (e.g. d__...;p__...;...;s__...)
    abundances: same-length list of abundance floats
    possible_ranks: which ranks we might choose from
    randomize: if True, pick one rank randomly each time; 
               otherwise, pick e.g. the *last* in possible_ranks or a fixed rank

    Returns:
      collapsed_tax_list, collapsed_abundance_list
    """
    if randomize:
        rank_code = random.choice(possible_ranks)
    else:
        rank_code = possible_ranks[-1]  # or a fixed choice

    # Dictionary to accumulate sums
    collapsed_map = {}

    for tax_str, ab in zip(taxonomies, abundances):
        # 1) Truncate up to the chosen rank
        truncated = truncate_taxonomy_up_to_rank(tax_str, rank_code)

        # 2) Sum abundance for identical truncated taxonomies
        collapsed_map[truncated] = collapsed_map.get(truncated, 0.0) + ab

    # Convert to lists
    collapsed_tax_list = list(collapsed_map.keys())
    collapsed_abundance_list = list(collapsed_map.values())
    return collapsed_tax_list, collapsed_abundance_list


class CollapsingTaxonomyDataset(Dataset):
    def __init__(
        self,
        X_taxonomy,
        taxonomy_list,
        max_taxa=104,
        ranks=('p','c','o','f','g','s'),
        do_augmentation=True
    ):
        """
        Args:
          X_taxonomy: SciPy sparse matrix (num_samples x num_taxonomies).
          taxonomy_list: List of strings, each is a full taxonomy string for a column in X_taxonomy.
          max_taxa: how many to pad/truncate to
          ranks: tuple/list of rank codes for random collapsing (p, c, o, f, g, s)
          do_augmentation: if True, pick a random rank each time; 
                           if False, do no collapse (or a fixed approach).
        """
        super().__init__()
        self.X_taxonomy = X_taxonomy
        self.taxonomy_list = taxonomy_list
        self.max_taxa = max_taxa
        self.ranks = ranks
        self.do_augmentation = do_augmentation
        self.num_samples = X_taxonomy.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1) Non-zero indices + values
        nonzero_tax_indices = self.X_taxonomy[idx].nonzero()[1]
        nonzero_tax_values  = self.X_taxonomy[idx, nonzero_tax_indices].toarray().flatten()
        
		# # debugging:
        # # Let's check if all are strings:
        # for i, t in zip(nonzero_tax_indices, taxonomies):
        #     if not isinstance(t, str):
        #         print(f"Index {i} yields a non-string taxonomy: {t} (type={type(t)})")
		# 		# You could raise an error or skip here:
		# 		# raise ValueError(f"Taxonomy {i} is not a string!")`



		###############

        # Original full taxonomies
        original_taxa = [self.taxonomy_list[i] for i in nonzero_tax_indices]

        # 2) Maybe collapse at random rank
        if self.do_augmentation:
            collapsed_taxa, collapsed_ab = collapse_taxa_up_to_rank(
                original_taxa, nonzero_tax_values,
                possible_ranks=self.ranks,
                randomize=True
            )
        else:
            # No augmentation => keep original
            collapsed_taxa, collapsed_ab = original_taxa, nonzero_tax_values

        # 3) Sort by abundance descending
        tax_val_pairs = list(zip(collapsed_taxa, collapsed_ab))
        tax_val_pairs_sorted = sorted(tax_val_pairs, key=lambda x: x[1], reverse=True)

        sorted_taxonomies  = [p[0] for p in tax_val_pairs_sorted]
        sorted_abundances  = [p[1] for p in tax_val_pairs_sorted]

        # 4) Pad or truncate
        cur_len = len(sorted_taxonomies)
        if cur_len < self.max_taxa:
            pad_len = self.max_taxa - cur_len
            sorted_taxonomies += ["<pad>"] * pad_len
            sorted_abundances += [0.0] * pad_len
        elif cur_len > self.max_taxa:
            sorted_taxonomies = sorted_taxonomies[:self.max_taxa]
            sorted_abundances = sorted_abundances[:self.max_taxa]

        # Return both
        return {
            "taxonomies": sorted_taxonomies,
            "abundances": sorted_abundances
        }
    
### Colapse or truncate tree in a random paths of the tree

import random

def random_collapse_per_path(taxonomies, abundances, possible_ranks=('p','c','o','f','g','s')):
    """
    For each taxonomy path (e.g., d__...;p__...;c__...), pick a random rank from 'possible_ranks'
    that actually appears in that path. Truncate the taxonomy at that rank, merging everything below.
    
    Then group (sum) all truncated paths that ended up identical.
    
    Args:
        taxonomies: list of taxonomy strings
        abundances: parallel list of floats
        possible_ranks: tuple of rank codes (e.g. 'p','c','o','f','g','s')
        
    Returns:
        collapsed_tax_list, collapsed_abundance_list
    """

    collapsed_map = {}

    for tax_str, ab in zip(taxonomies, abundances):
        # Parse the path
        ranks = tax_str.split(';')  # e.g. ["d__Bacteria", "p__Firmicutes", "c__Bacilli", ...]

        # Find which ranks in possible_ranks are actually present in this path
        # e.g. "p__Firmicutes" => rank_code = 'p'
        candidate_positions = []
        for i, r in enumerate(ranks):
            # e.g. r might be "p__Firmicutes"
            # rank_code is what's before "__"
            if '__' in r:
                code = r.split('__', 1)[0]  # "p"
                if code in possible_ranks:
                    candidate_positions.append(i)

        if len(candidate_positions) == 0:
            # No overlap with the chosen ranks => keep full path as is
            truncated_path = tax_str
        else:
            # Pick a random position among those that match
            chosen_i = random.choice(candidate_positions)
            # Keep everything up to that position (inclusive)
            truncated_path = ";".join(ranks[: chosen_i + 1])

        # Merge/sum the abundance
        collapsed_map[truncated_path] = collapsed_map.get(truncated_path, 0.0) + ab

    # Convert dict back to parallel lists
    collapsed_tax_list = list(collapsed_map.keys())
    collapsed_ab_list  = list(collapsed_map.values())
    return collapsed_tax_list, collapsed_ab_list

class PartialMergingTaxonomyDataset(Dataset):
    def __init__(
        self,
        X_taxonomy,         # SciPy sparse matrix: (num_samples, num_taxonomies)
        taxonomy_list,      # list of taxonomy strings (unique columns from your mapping)
        token_dict,         # token dictionary mapping taxonomy string -> integer token
        max_taxa=104,
        possible_ranks=('p','c','o','f','g','s'),
        do_augmentation=True
    ):
        """
        For each sample, extract nonzero taxonomy values from the matrix, then (optionally) apply random augmentation.
        Finally, sort by abundance and pad/truncate to fixed length. Convert the taxonomy strings to token IDs.
        """
        super().__init__()
        self.X_taxonomy = X_taxonomy
        self.taxonomy_list = taxonomy_list
        self.token_dict = token_dict
        self.max_taxa = max_taxa
        self.possible_ranks = possible_ranks
        self.do_augmentation = do_augmentation
        self.num_samples = X_taxonomy.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1) Get non-zero taxonomy indices and abundances for the sample
        nonzero_tax_indices = self.X_taxonomy[idx].nonzero()[1]
        nonzero_tax_values  = self.X_taxonomy[idx, nonzero_tax_indices].toarray().flatten()
        original_taxa = [self.taxonomy_list[i] for i in nonzero_tax_indices]
        
        # 2) Apply dynamic augmentation if desired
        if self.do_augmentation:
            collapsed_taxa, _ = random_collapse_per_path(
                original_taxa, nonzero_tax_values,
                possible_ranks=self.possible_ranks
            )
        else:
            collapsed_taxa = original_taxa

        # 3) Sort by abundance descending (using the original abundance values)
        tax_val_pairs = list(zip(collapsed_taxa, nonzero_tax_values))
        tax_val_pairs_sorted = sorted(tax_val_pairs, key=lambda x: x[1], reverse=True)
        sorted_taxonomies = [p[0] for p in tax_val_pairs_sorted]

        # 4) Pad or truncate to fixed length
        cur_len = len(sorted_taxonomies)
        if cur_len < self.max_taxa:
            pad_len = self.max_taxa - cur_len
            sorted_taxonomies += ["<pad>"] * pad_len
        elif cur_len > self.max_taxa:
            sorted_taxonomies = sorted_taxonomies[:self.max_taxa]

        # 5) Convert taxonomy strings to token IDs dynamically
        tokens = [self.token_dict.get(t, self.token_dict["<mask>"]) for t in sorted_taxonomies]
        
        # 6) Return a dictionary with dynamic data and original sample index for tracking
        return {
            "input_ids": tokens,
            "length": len(tokens),
            "sample_idx": idx
        }


#############################################    
####### Hugging face dataloader #############
#############################################    

from datasets import Dataset as HFDataset
def dataset_to_hf_dataset(pt_dataset, token_dict):
    """
    Converts a PartialMergingTaxonomyDataset to a Hugging Face Dataset.
    We also store each sample's index in a column called "sample_idx."
    """
    import numpy as np
    from datasets import Dataset

    input_ids_list = []
    length_list = []
    sample_idx_list = []

    for i in range(len(pt_dataset)):
        sample = pt_dataset[i]  
        # sample["taxonomies"] = padded list of length max_taxa
        taxonomies = sample["taxonomies"]

        # Convert each taxonomy to an integer token
        tokens = []
        for t_str in taxonomies:
            # handle "<pad>" or missing tokens
            if t_str in token_dict:
                tokens.append(token_dict[t_str])
            else:
                tokens.append(token_dict["<mask>"])  # or <unk>

        input_ids_list.append(tokens)
        length_list.append(len(tokens))   # typically max_taxa
        sample_idx_list.append(i)         # store the original dataset index

    data_dict = {
        "input_ids": input_ids_list,
        "length": length_list,
        "sample_idx": sample_idx_list
    }

    hf_dataset = Dataset.from_dict(data_dict)
    return hf_dataset

## data colator
import torch
import os, json
from transformers import DataCollatorForLanguageModeling, BatchEncoding
from transformers.tokenization_utils_base import PaddingStrategy

class TaxonomyPreCollator:
    """
    'Tokenizer-like' object with all methods needed by DataCollatorForLanguageModeling.
    """
    def __init__(
        self, 
        token_dict, 
        pad_token="<pad>", 
        mask_token="<mask>", 
        cls_token="<cls>", 
        eos_token="<eos>"
    ):
        self.token_dict = token_dict
        
        # The HF DataCollator checks these string attributes
        self.pad_token   = pad_token
        self.mask_token  = mask_token
        self.cls_token   = cls_token
        self.eos_token   = eos_token
        
        # Their integer IDs (assuming they exist in token_dict)
        self.pad_token_id  = token_dict[pad_token]
        self.mask_token_id = token_dict[mask_token]
        self.cls_token_id  = token_dict.get(cls_token, -1)
        self.eos_token_id  = token_dict.get(eos_token, -1)

        self.model_input_names = ["input_ids"]
        self.padding_side = "right"

    def convert_tokens_to_ids(self, tokens):
        """
        The DataCollator calls this to convert string tokens (like "<mask>")
        into integer IDs. In practice, it mostly needs to do:
           convert_tokens_to_ids(self.mask_token)
        to get the ID for "[MASK]" replacements.
        """
        # If tokens is a single string, return a single ID
        if isinstance(tokens, str):
            return self.token_dict.get(tokens, self.token_dict[self.mask_token])
        # If tokens is a list of strings, convert each
        elif isinstance(tokens, list):
            return [
                self.token_dict.get(t, self.token_dict[self.mask_token]) 
                for t in tokens
            ]
        else:
            raise TypeError(
                f"convert_tokens_to_ids expected str or List[str], got {type(tokens)}"
            )

    def get_special_tokens_mask(self, token_ids, already_has_special_tokens=True):
        """
        Return a 0/1 list indicating which positions are special (1)
        so they won't be masked. 
        """
        special_ids = {
            self.pad_token_id, 
            self.mask_token_id, 
            self.cls_token_id, 
            self.eos_token_id
        }
        return [1 if t in special_ids else 0 for t in token_ids]

    def pad(
        self, 
        encoded_inputs, 
        padding=True, 
        max_length=None, 
        pad_to_multiple_of=None, 
        return_tensors="pt",
        **kwargs
    ):
        # If it's a list of row dicts, unify them into a dict of columns
        if isinstance(encoded_inputs, list):
            batch_dict = {"input_ids": []}
            for ex in encoded_inputs:
                batch_dict["input_ids"].append(ex["input_ids"])
        else:
            batch_dict = encoded_inputs

        # Find the longest sequence
        max_seq_len = max(len(seq) for seq in batch_dict["input_ids"])

        # Pad each sequence
        padded_input_ids = []
        for seq in batch_dict["input_ids"]:
            padding_needed = max_seq_len - len(seq)
            new_seq = seq + [self.pad_token_id]*padding_needed
            padded_input_ids.append(new_seq)

        batch_dict["input_ids"] = padded_input_ids

        # Construct attention_mask
        attention_masks = []
        for seq in padded_input_ids:
            attn = [1 if t != self.pad_token_id else 0 for t in seq]
            attention_masks.append(attn)
        batch_dict["attention_mask"] = attention_masks

        # Convert to PyTorch
        if return_tensors == "pt":
            batch_dict["input_ids"]       = torch.tensor(batch_dict["input_ids"],       dtype=torch.long)
            batch_dict["attention_mask"]  = torch.tensor(batch_dict["attention_mask"],  dtype=torch.long)

        return BatchEncoding(batch_dict)

    def __call__(self, batch):
        return self.pad(batch)

    def __len__(self):
        # The data collator checks len(...) 
        return len(self.token_dict)
    
    def save_pretrained(self, output_dir):
        path = os.path.join(output_dir, "token_dict.json")
        with open(path, "w") as f:
            json.dump(self.token_dict, f)


def create_taxonomy_data_collator(token_dict, mlm_probability=0.15, mask_replace_prob=0.8):
    pre_collator = TaxonomyPreCollator(token_dict)
    mlm_collator = DataCollatorForLanguageModeling(
        tokenizer=pre_collator, 
        mlm=True, 
        mlm_probability=0.15,
        mask_replace_prob=mask_replace_prob,
    )
    return mlm_collator
