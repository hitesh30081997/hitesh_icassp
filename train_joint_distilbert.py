"""
Train a joint DistilBERT model for intent classification + slot filling on
the preprocessed Fluent Speech Commands (FSC) dataset produced by
`prepare_fsc_dataset.py`.

Expects:
    fsc_processed/dataset/          (HF DatasetDict: train/validation/test)
    fsc_processed/intent2id.json
    fsc_processed/slot2id.json

Install deps first:
    pip install transformers datasets seqeval torch --break-system-packages
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_from_disk
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import (
    DistilBertConfig,
    DistilBertModel,
    DistilBertPreTrainedModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

MODEL_NAME = "distilbert-base-uncased"
DATA_DIR = Path("fsc_processed")
OUTPUT_DIR = Path("fsc_joint_model")


# --------------------------------------------------------------------------
# Model: DistilBERT backbone + intent head (on [CLS]) + slot head (per-token)
# --------------------------------------------------------------------------
class JointDistilBertForIntentSlot(DistilBertPreTrainedModel):
    def __init__(self, config, num_intents, num_slots, intent_loss_weight=1.0,
                 slot_loss_weight=1.0, dropout=0.1):
        super().__init__(config)
        self.num_intents = num_intents
        self.num_slots = num_slots
        self.intent_loss_weight = intent_loss_weight
        self.slot_loss_weight = slot_loss_weight

        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(dropout)
        self.intent_classifier = nn.Linear(config.hidden_size, num_intents)
        self.slot_classifier = nn.Linear(config.hidden_size, num_slots)

        self.post_init()

    def forward(self, input_ids=None, attention_mask=None,
                intent_label=None, slot_labels=None):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state          # (B, T, H)
        pooled_output = sequence_output[:, 0]                 # [CLS] token

        intent_logits = self.intent_classifier(self.dropout(pooled_output))
        slot_logits = self.slot_classifier(self.dropout(sequence_output))

        loss = None
        if intent_label is not None and slot_labels is not None:
            intent_loss = nn.CrossEntropyLoss()(intent_logits, intent_label)
            slot_loss = nn.CrossEntropyLoss(ignore_index=-100)(
                slot_logits.view(-1, self.num_slots), slot_labels.view(-1)
            )
            loss = self.intent_loss_weight * intent_loss + self.slot_loss_weight * slot_loss

        return {"loss": loss, "intent_logits": intent_logits, "slot_logits": slot_logits}


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def build_compute_metrics(id2slot):
    def compute_metrics(eval_pred):
        (intent_logits, slot_logits), (intent_labels, slot_labels) = eval_pred.predictions, eval_pred.label_ids

        intent_preds = np.argmax(intent_logits, axis=-1)
        intent_acc = float((intent_preds == intent_labels).mean())

        slot_preds = np.argmax(slot_logits, axis=-1)
        true_slots, pred_slots = [], []
        for pred_row, label_row in zip(slot_preds, slot_labels):
            t_seq, p_seq = [], []
            for p, l in zip(pred_row, label_row):
                if l == -100:
                    continue
                t_seq.append(id2slot[int(l)])
                p_seq.append(id2slot[int(p)])
            true_slots.append(t_seq)
            pred_slots.append(p_seq)

        exact_match = float(np.mean([
            (intent_preds[i] == intent_labels[i]) and (true_slots[i] == pred_slots[i])
            for i in range(len(true_slots))
        ]))

        return {
            "intent_accuracy": intent_acc,
            "slot_f1": f1_score(true_slots, pred_slots),
            "slot_precision": precision_score(true_slots, pred_slots),
            "slot_recall": recall_score(true_slots, pred_slots),
            "sentence_exact_match": exact_match,
        }
    return compute_metrics


def main():
    # ---- load data & label maps ----
    ds = load_from_disk(str(DATA_DIR / "dataset"))
    intent2id = json.load(open(DATA_DIR / "intent2id.json"))
    slot2id = json.load(open(DATA_DIR / "slot2id.json"))
    id2slot = {v: k for k, v in slot2id.items()}
    id2intent = {v: k for k, v in intent2id.items()}

    keep_cols = ["input_ids", "attention_mask", "slot_labels", "intent_label"]
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in keep_cols])
    ds.set_format(type="torch", columns=keep_cols)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ---- model ----
    config = DistilBertConfig.from_pretrained(MODEL_NAME)
    model = JointDistilBertForIntentSlot.from_pretrained(
        MODEL_NAME,
        config=config,
        num_intents=len(intent2id),
        num_slots=len(slot2id),
    )

    # ---- training args ----
    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="slot_f1",
        greater_is_better=True,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=10,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        label_names=["intent_label", "slot_labels"],  # tells Trainer these are targets, not inputs
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=default_data_collator,
        compute_metrics=build_compute_metrics(id2slot),
    )

    # ---- train ----
    trainer.train()

    # ---- validation metrics (best checkpoint, already loaded) ----
    val_metrics = trainer.evaluate(ds["validation"])
    print("\n=== Validation metrics ===")
    for k, v in val_metrics.items():
        print(f"{k}: {v}")

    # ---- test set: predict + full seqeval report ----
    test_output = trainer.predict(ds["test"])
    intent_logits, slot_logits = test_output.predictions
    intent_labels, slot_labels = test_output.label_ids

    intent_preds = np.argmax(intent_logits, axis=-1)
    slot_preds = np.argmax(slot_logits, axis=-1)

    true_slots, pred_slots = [], []
    for pred_row, label_row in zip(slot_preds, slot_labels):
        t_seq, p_seq = [], []
        for p, l in zip(pred_row, label_row):
            if l == -100:
                continue
            t_seq.append(id2slot[int(l)])
            p_seq.append(id2slot[int(p)])
        true_slots.append(t_seq)
        pred_slots.append(p_seq)

    print("\n=== Test metrics ===")
    print("Intent accuracy:", float((intent_preds == intent_labels).mean()))
    print("\nSlot classification report (seqeval):")
    print(classification_report(true_slots, pred_slots, digits=3))

    # ---- save everything needed for inference ----
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    trainer.save_model(str(OUTPUT_DIR / "final_model"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final_model"))
    json.dump(intent2id, open(OUTPUT_DIR / "intent2id.json", "w"), indent=2)
    json.dump(slot2id, open(OUTPUT_DIR / "slot2id.json", "w"), indent=2)
    print(f"\nSaved model + tokenizer + label maps to {OUTPUT_DIR / 'final_model'}")


if __name__ == "__main__":
    main()
