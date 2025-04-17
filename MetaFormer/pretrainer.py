from transformers import (
    BertConfig, 
    BertForMaskedLM, 
    TrainingArguments, 
    Trainer
)
from transformers import TrainerCallback, TrainerState, TrainerControl
import pickle 
import os
import math

class TaxonomyPretrainer(Trainer):
    """
    Minimal imitation of GeneformerPretrainer. 
    Optionally handles length-based sampling if you store a 'length' for each item in your dataset.
    """
    def __init__(self, *args, example_lengths_file=None, **kwargs):
        self.example_lengths = None
        
        # If user provides a file of precomputed lengths
        if example_lengths_file is not None:
            with open(example_lengths_file, "rb") as f:
                self.example_lengths = pickle.load(f)
        
        super().__init__(*args, **kwargs)
    
    def _get_train_sampler(self):
        """
        If you want length-based grouping, you can override here 
        using the self.example_lengths data. 
        If not, just rely on default Trainer sampler logic.
        """
        # For simplicity, let's use default super() so no length grouping
        return super()._get_train_sampler()
    

class SaveEveryNEpochsCallback(TrainerCallback):
    def __init__(self, save_every=5, output_dir=None):
        super().__init__()
        self.save_every = save_every
        self.output_dir = output_dir

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # Convert epoch (which may be a float) to an integer epoch number.
        epoch_int = int(math.floor(state.epoch))
        print(f"[DEBUG] epoch end, state.epoch={state.epoch} => epoch_int={epoch_int}")

        # Save if this is epoch 1 or a multiple of self.save_every.
        if epoch_int == 1 or (epoch_int > 1 and epoch_int % self.save_every == 0):
            checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-epoch{epoch_int}")
            # First, try to get the trainer from kwargs.
            trainer = kwargs.get("trainer", None)
            if trainer is not None:
                trainer.save_model(checkpoint_dir)
            else:
                # Fallback: try to get the model and save it directly.
                model = kwargs.get("model", None)
                if model is not None:
                    model.save_pretrained(checkpoint_dir)
                else:
                    print("Warning: No trainer or model found; checkpoint not saved.")
            print(f"*** Saved checkpoint at epoch {epoch_int} to {checkpoint_dir} ***")
        return control