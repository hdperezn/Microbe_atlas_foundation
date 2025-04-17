import numpy as np
import torch

def compute_custom_metrics(eval_pred):
    #print("We are in compute_custom_metrics") 
    pred_ids, labels = eval_pred  # these should be already argmax'ed
    total_masked, correct, total_repeated = 0, 0, 0
    batch_repetition_fractions = []

    for pred, label in zip(pred_ids, labels):
        masked_positions = (label != -100)
        true_tokens = label[masked_positions]
        predicted_tokens = pred[masked_positions]

        total_masked += len(true_tokens)
        correct += (predicted_tokens == true_tokens).sum().item()

        # Repetition
        if len(predicted_tokens) > 0:
            repetition = 1 - len(torch.unique(predicted_tokens)) / len(predicted_tokens)
        else:
            repetition = 0
        batch_repetition_fractions.append(repetition)
        total_repeated += len(predicted_tokens) - len(torch.unique(predicted_tokens))

    return {
        "accuracy": correct / total_masked if total_masked else 0,
        "mean_repetition": np.mean(batch_repetition_fractions),
        "total_masked": total_masked,
        "total_repeated": total_repeated,
    }

from transformers import TrainerCallback
from torch.utils.data import DataLoader
from torch.amp import autocast  
from torch import no_grad
from tqdm import tqdm



## this callback is used to log the metrics on the training batch

class LogTrainBatchMetricsCallback(TrainerCallback):
    def __init__(self, compute_metrics_fn, trainer_ref, eval_every_steps=1000):
        self.compute_metrics_fn = compute_metrics_fn
        self.trainer = trainer_ref
        self.eval_every_steps = eval_every_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_every_steps == 0:
            self._log_train_metrics(args, state)

    def _log_train_metrics(self, args, state):
        print(f"\n Evaluating training batch at step {state.global_step}...")

        try:
            train_batch = next(iter(self.trainer.get_train_dataloader()))
        except StopIteration:
            print("Could not retrieve a training batch.")
            return

        device = self.trainer.model.device
        batch = {k: v.to(device) for k, v in train_batch.items()}

        self.trainer.model.eval()
        with torch.no_grad(), torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            outputs = self.trainer.model(**batch)

        pred_ids = outputs.logits.argmax(-1).cpu()
        labels = batch["labels"].cpu()

        metrics = self.compute_metrics_fn((pred_ids, labels))
        metrics = {f"train_batch_{k}": v for k, v in metrics.items()}

        if self.trainer.is_world_process_zero():
            self.trainer.log(metrics)

        self.trainer.model.train()
        torch.cuda.empty_cache()

# This callback is used to log the metrics on the evaluation set (because its too big to evaluate directly on the GPU)
class LogValMetricsCallback(TrainerCallback):
    def __init__(self, eval_dataset, collator, compute_metrics_fn, batch_size=16, eval_every_steps=1000):
        self.eval_dataset = eval_dataset
        self.collator = collator
        self.compute_metrics_fn = compute_metrics_fn
        self.batch_size = batch_size
        self.eval_every_steps = eval_every_steps

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.eval_every_steps == 0:
            self._run_streaming_eval(args, state, control, model)

    def _run_streaming_eval(self, args, state, control, model):
        #print(f"\n🔍 Streaming evaluation at step {state.global_step}...")

        model.eval()
        dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collator
        )

        total_masked, correct, total_repeated = 0, 0, 0
        batch_repetition_fractions = []

        with no_grad():
            for batch in tqdm(dataloader, desc="Eval", leave=False):
                batch = {k: v.to(model.device) for k, v in batch.items()}
                with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                    outputs = model(**batch)

                pred_ids = outputs.logits.argmax(-1)
                labels = batch["labels"]

                for pred, label in zip(pred_ids, labels):
                    masked_positions = (label != -100)
                    true_tokens = label[masked_positions]
                    predicted_tokens = pred[masked_positions]

                    total_masked += len(true_tokens)
                    correct += (predicted_tokens == true_tokens).sum().item()

                    if len(predicted_tokens) > 0:
                        repetition = 1 - len(torch.unique(predicted_tokens)) / len(predicted_tokens)
                    else:
                        repetition = 0.0
                    batch_repetition_fractions.append(repetition)
                    total_repeated += len(predicted_tokens) - len(torch.unique(predicted_tokens))

        metrics = {
            "accuracy": correct / total_masked if total_masked else 0,
            "mean_repetition": np.mean(batch_repetition_fractions),
            "total_masked": total_masked,
            "total_repeated": total_repeated,
        }

        #print("Eval metrics:", metrics)

        if model is not None and hasattr(model, "trainer"):
            model.trainer.log(metrics)

        model.train()