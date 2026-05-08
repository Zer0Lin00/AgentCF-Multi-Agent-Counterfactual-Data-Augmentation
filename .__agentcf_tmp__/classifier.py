from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("sample_weight", None)
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        losses = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
        if weights is not None:
            weights = weights.to(losses.device).float()
            loss = (losses * weights).sum() / weights.sum().clamp_min(1e-6)
        else:
            loss = losses.mean()
        return (loss, outputs) if return_outputs else loss


@dataclass
class HFClassifier:
    model_name: str
    max_length: int

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)

    def _to_dataset(self, df):
        cols = ["text", "label"] + (["sample_weight"] if "sample_weight" in df.columns else [])
        ds = Dataset.from_pandas(df[cols], preserve_index=False)
        return ds.map(
            lambda x: self.tokenizer(x["text"], truncation=True, max_length=self.max_length),
            batched=True,
            remove_columns=["text"],
        )

    @staticmethod
    def _compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro"),
        }

    def train_and_eval(self, train_df, val_df, cfg: dict[str, Any], out_dir: str) -> dict[str, float]:
        train_ds = self._to_dataset(train_df)
        val_ds = self._to_dataset(val_df)
        collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        args = TrainingArguments(
            output_dir=out_dir,
            per_device_train_batch_size=int(cfg["batch_size"]),
            per_device_eval_batch_size=int(cfg["batch_size"]),
            num_train_epochs=float(cfg["epochs"]),
            learning_rate=float(cfg["learning_rate"]),
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=20,
            report_to="none",
            seed=int(cfg["seed"]),
            do_train=True,
            do_eval=True,
            weight_decay=float(cfg.get("training_regularization", {}).get("weight_decay", 0.0)),
        )
        trainer_cls = WeightedTrainer if "sample_weight" in train_df.columns else Trainer
        trainer = trainer_cls(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=self.tokenizer,
            data_collator=collator,
            compute_metrics=self._compute_metrics,
        )
        trainer.train()
        metrics = trainer.evaluate()
        return {
            "acc": float(metrics.get("eval_accuracy", 0.0)),
            "f1": float(metrics.get("eval_macro_f1", 0.0)),
        }

    def evaluate_df(self, df, cfg: dict[str, Any], out_dir: str) -> dict[str, float]:
        eval_ds = self._to_dataset(df)
        collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        args = TrainingArguments(
            output_dir=out_dir,
            per_device_eval_batch_size=int(cfg["batch_size"]),
            report_to="none",
            do_train=False,
            do_eval=True,
            seed=int(cfg["seed"]),
        )
        trainer = Trainer(
            model=self.model,
            args=args,
            eval_dataset=eval_ds,
            processing_class=self.tokenizer,
            data_collator=collator,
            compute_metrics=self._compute_metrics,
        )
        metrics = trainer.evaluate()
        return {
            "acc": float(metrics.get("eval_accuracy", 0.0)),
            "f1": float(metrics.get("eval_macro_f1", 0.0)),
        }
