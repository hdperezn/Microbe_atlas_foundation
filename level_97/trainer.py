import os
import sys
import json

# Add the parent directory of level_97 (i.e., MicrobeAtlas) to sys.path
project_root = os.path.abspath(os.path.join(os.getcwd(), "../MetaFormer"))
sys.path.append(project_root)

import math
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns 
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
#from dataloaders import random_collapse_per_path, PartialMergingTaxonomyDataset
from torch.utils.data import random_split

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

from datasets import Dataset as HFDataset
from transformers import DataCollatorForLanguageModeling, BatchEncoding
from dataloaders import dataset_to_hf_dataset, TaxonomyPreCollator, create_taxonomy_data_collator
from metrics_and_callback import LogTrainBatchMetricsCallback, compute_custom_metrics, LogValMetricsCallback

import random
import torch
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

from torch.utils.data import DataLoader, Dataset


class OTUSequenceJSONLDataset(Dataset):
    def __init__(self, jsonl_path, token_dict):
        self.jsonl_path = jsonl_path
        self.token_dict = token_dict

        # Step 1: Index file line positions (byte offsets)
        self.offsets = []
        with open(jsonl_path, "rb") as f:
            offset = 0
            for line in f:
                self.offsets.append(offset)
                offset += len(line)

        self.num_samples = len(self.offsets)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        offset = self.offsets[idx]
        with open(self.jsonl_path, "r") as f:
            f.seek(offset)
            line = f.readline()
            otu_sequence = json.loads(line)

        # Convert OTUs to token IDs (strict lookup)
        try:
            tokens = [self.token_dict[otu] for otu in otu_sequence]
        except KeyError as e:
            raise KeyError(f"OTU not found in token_dict: {e.args[0]} (in sample {idx})")

        return {
            "input_ids": tokens,
            "length": len(tokens),
            "sample_idx": idx
        }
    
# Adjust paths
OTU_matrix_path = "/home/hernan_melmoth/Documents/phd_work/Bio_ontology/MicrobeAtlas/level_97/otus_sequences.jsonl"
OTU_mapping_path = "/home/hernan_melmoth/Documents/phd_work/Bio_ontology/MicrobeAtlas/level_97/sample_ids.json"
token_dict_path = "/home/hernan_melmoth/Documents/phd_work/Bio_ontology/MicrobeAtlas/level_97/taxonomy_token_dict.pkl"

# Load sample ID list (List[str])
with open(OTU_mapping_path, "r") as f:
    sample_ids = json.load(f)

# Load token dictionary (Dict[str, int])
with open(token_dict_path, "rb") as f:
    token_dict = pickle.load(f)

# # Initialize dataset and dataloader
dataset = OTUSequenceJSONLDataset(OTU_matrix_path, token_dict)


# Assume partial_dataset is your instance of PartialMergingTaxonomyDataset
total_samples = len(dataset)
val_size = int(0.2 * total_samples)
train_size = total_samples - val_size


g = torch.Generator().manual_seed(123)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=g)

print(f"Total samples: {total_samples}")
print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")


original_val_size = len(val_dataset)
small_val_size = 1000#original_val_size // 2
test_size = original_val_size - small_val_size

# Split validation dataset into small val and test datasets
val_dataset, test_dataset = random_split(
    val_dataset, [small_val_size, test_size], generator=torch.Generator().manual_seed(123)
)

print(f"Original validation size: {original_val_size}")
print(f"Small validation dataset size (for periodic eval): {len(val_dataset)}")
print(f"Test dataset size (final evaluation): {len(test_dataset)}")

from transformers import (
    BertConfig, 
    BertForMaskedLM, 
    TrainingArguments, 
    Trainer
)
from transformers import TrainerCallback, TrainerState, TrainerControl

save_model_path = "/home/hernan_melmoth/Documents/phd_work/Bio_ontology/MicrobeAtlas/level_97/OTUs_model"

vocab_size = len(token_dict)  # token_dict includes <pad>, <mask>, etc.
print(vocab_size)


config = BertConfig(
    vocab_size=vocab_size,
    hidden_size=256,        # embedding dim
    num_hidden_layers=6,    # number of Transformer layers
    num_attention_heads=4,
    max_position_embeddings=480,  # or your max sequence length
    pad_token_id=token_dict["<pad>"],
)
model = BertForMaskedLM(config)
model.to(device)


training_args = TrainingArguments(
    output_dir= save_model_path,
    overwrite_output_dir=True,
    num_train_epochs=40 ,#50,
    per_device_train_batch_size= 16, #int(256/4),
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    logging_steps=1000,
    fp16=True,
    save_strategy="epoch",           # Automatically saves every epoch
    eval_strategy="no",
    #eval_steps=10,                 # Evaluate every 1000 steps
    #eval_accumulation_steps=10,
    logging_strategy="steps",
    logging_dir=f"{save_model_path}/logs",
)

# 3) data collator (the one you tested)
collator = create_taxonomy_data_collator(token_dict)

# 4) Build the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collator,
    compute_metrics=compute_custom_metrics,
)
eval_every_steps = 1000
trainer.add_callback(LogValMetricsCallback(
    eval_dataset=val_dataset,
    collator=collator,
    compute_metrics_fn=compute_custom_metrics,
    batch_size=16,
    eval_every_steps=eval_every_steps
))

trainer.add_callback(LogTrainBatchMetricsCallback(
    compute_metrics_fn=compute_custom_metrics,
    trainer_ref=trainer,
    eval_every_steps=eval_every_steps
))
# 5) Train
trainer.train()
trainer.save_model(save_model_path)
