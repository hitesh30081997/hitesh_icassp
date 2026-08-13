"""
Inference on a saved Semi-Cascaded SLU checkpoint, for audio you've already
noised yourself and stored in the same directory/CSV layout as the original
Fluent Speech Commands dataset.

No offline ASR cache is used here -- transcripts don't exist for the noisy
audio, so this script decodes them live: for every batch, the log-mel
features are computed once and reused for BOTH (a) the Whisper ASR decode
that produces the transcript fed to the text branch, and (b) the frozen
acoustic encoder inside the model. That keeps the two branches consistent
(same underlying noisy signal) and avoids running the feature extractor
twice.

Expected layout (same as original FSC):
    <root_dir>/
        wavs/speakers/.../*.wav      (your noised audio, same relative paths)
        data/train_data.csv
        data/valid_data.csv
        data/test_data.csv           (or whichever split you point at)

CSV columns expected: path, action, object, location
(a `transcription` column may exist but is ignored -- it would be the
CLEAN-speech ground truth text, not a valid transcript of your noisy audio)

Usage:
    python infer_fsc_noisy.py \\
        --root_dir noisy_fluent_speech_commands_snr0 \\
        --csv data/test_data.csv \\
        --checkpoint fsc_semi_cascaded_slu_best.pt \\
        --vocab fsc_label_vocab.json \\
        --output_csv predictions_snr0.csv
"""

import argparse
import json
import os
from typing import Dict, List, Optional

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    BertTokenizerFast,
)

from fsc_semi_cascaded_slu import (
    FSCLabelVocab,
    SemiCascadedSLU_FSC,
    load_and_resample,
)


# =========================================================================== #
# 1. Minimal dataset: just loads raw (already-noisy) waveforms + any labels
# =========================================================================== #
class NoisyAudioInferenceDataset(Dataset):
    """
    Reads wav paths (and, if present, action/object/location labels) from a
    CSV in the standard FSC layout. Returns raw waveforms only -- no text,
    since the noisy-audio transcripts don't exist and will be generated at
    inference time instead.
    """

    def __init__(self, csv_path: str, root_dir: str, vocab: FSCLabelVocab, sample_rate: int = 16000):
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.vocab = vocab
        self.sample_rate = sample_rate
        self.has_labels = all(c in self.df.columns for c in ("action", "object", "location"))
        # a `transcription` column, if present, is the CLEAN-speech ground
        # truth text. It is NOT a valid transcript of the noisy audio, but
        # it's still useful as a REFERENCE to measure how much the ASR
        # branch degrades under noise (word/character error rate).
        self.has_reference_transcript = "transcription" in self.df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        wav_rel_path = row["path"]
        waveform = load_and_resample(os.path.join(self.root_dir, wav_rel_path), self.sample_rate)

        item = {"waveform": waveform, "path": wav_rel_path}
        if self.has_labels:
            item["action"] = row["action"]
            item["object"] = row["object"]
            item["location"] = row["location"]
        if self.has_reference_transcript:
            item["reference_transcript"] = str(row["transcription"])
        return item


def collate_raw(batch: List[Dict]) -> Dict:
    """No padding/tokenization here -- that happens per-batch in the
    inference loop once we have the feature extractor and tokenizer handy."""
    out = {
        "waveforms": [b["waveform"] for b in batch],
        "paths": [b["path"] for b in batch],
    }
    if "action" in batch[0]:
        out["actions"] = [b["action"] for b in batch]
        out["objects"] = [b["object"] for b in batch]
        out["locations"] = [b["location"] for b in batch]
    if "reference_transcript" in batch[0]:
        out["reference_transcripts"] = [b["reference_transcript"] for b in batch]
    return out


# =========================================================================== #
# 2. Inference engine
# =========================================================================== #
class FSCNoisyInference:
    def __init__(
        self,
        checkpoint_path: str,
        vocab_path: str,
        whisper_model_name: str = "openai/whisper-small",
        bert_model_name: str = "bert-base-uncased",
        whisper_layer: int = 6,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        asr_language: str = "en",
        max_text_len: int = 32,
    ):
        self.device = device
        self.max_text_len = max_text_len

        self.vocab = FSCLabelVocab.load(vocab_path)

        # -- ASR components (live transcription of the noisy audio) --
        self.asr_processor = WhisperProcessor.from_pretrained(whisper_model_name)
        self.asr_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_name).to(device)
        self.asr_model.eval()
        self.asr_model.generation_config.forced_decoder_ids = None
        self.asr_language = asr_language

        # feature extractor is shared between ASR decode and the frozen
        # acoustic encoder inside the classification model -- computed once
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model_name)
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)

        # -- classification model --
        self.model = SemiCascadedSLU_FSC(
            num_actions=self.vocab.num_actions,
            num_objects=self.vocab.num_objects,
            num_locations=self.vocab.num_locations,
            whisper_model_name=whisper_model_name,
            bert_model_name=bert_model_name,
            whisper_layer=whisper_layer,
            freeze_whisper=True,
            freeze_bert=False,
        ).to(device)
        state_dict = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.no_grad()
    def transcribe(self, waveforms: List, sample_rate: int = 16000):
        """Compute log-mel features once, decode a transcript from them."""
        audio_inputs = self.feature_extractor(
            waveforms, sampling_rate=sample_rate, return_tensors="pt", return_attention_mask=True
        )
        input_features = audio_inputs.input_features.to(self.device)
        attention_mask = audio_inputs.attention_mask.to(self.device)

        generated_ids = self.asr_model.generate(
            input_features,
            attention_mask=attention_mask,
            language=self.asr_language,
            task="transcribe",
        )
        transcripts = [
            t.strip() for t in self.asr_processor.batch_decode(generated_ids, skip_special_tokens=True)
        ]
        return input_features, transcripts

    @torch.no_grad()
    def predict_batch(self, waveforms: List, sample_rate: int = 16000) -> Dict:
        input_features, transcripts = self.transcribe(waveforms, sample_rate)

        text_inputs = self.tokenizer(
            transcripts, padding=True, truncation=True, max_length=self.max_text_len, return_tensors="pt"
        )
        input_ids = text_inputs["input_ids"].to(self.device)
        text_attention_mask = text_inputs["attention_mask"].to(self.device)

        out = self.model(
            input_features=input_features,
            input_ids=input_ids,
            text_attention_mask=text_attention_mask,
        )

        pred_action_ids = out["action_logits"].argmax(-1).cpu().tolist()
        pred_object_ids = out["object_logits"].argmax(-1).cpu().tolist()
        pred_location_ids = out["location_logits"].argmax(-1).cpu().tolist()

        return {
            "transcripts": transcripts,
            "pred_action": [self.vocab.id2action[i] for i in pred_action_ids],
            "pred_object": [self.vocab.id2object[i] for i in pred_object_ids],
            "pred_location": [self.vocab.id2location[i] for i in pred_location_ids],
        }


# =========================================================================== #
# 2b. Metrics helpers
# =========================================================================== #
def _edit_distance(ref: List[str], hyp: List[str]) -> int:
    """Standard Levenshtein distance (insert/delete/substitute, unit cost)."""
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            tmp = dp[j]
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = tmp
    return dp[m]


def word_error_rate(refs: List[str], hyps: List[str]) -> Dict[str, float]:
    """
    Corpus-level WER = total edits / total reference words (standard ASR
    metric), plus the mean of per-utterance WER for reference. Measures how
    much the ASR branch degrades on your noisy audio, using the clean-speech
    `transcription` column as the reference text.
    """
    total_edits, total_words = 0, 0
    per_utt = []
    for r, h in zip(refs, hyps):
        r_tok, h_tok = r.lower().split(), h.lower().split()
        edits = _edit_distance(r_tok, h_tok)
        total_edits += edits
        total_words += len(r_tok)
        per_utt.append(edits / len(r_tok) if len(r_tok) > 0 else 0.0)
    corpus_wer = total_edits / total_words if total_words > 0 else float("nan")
    mean_utt_wer = sum(per_utt) / len(per_utt) if per_utt else float("nan")
    return {"wer_corpus": corpus_wer, "wer_mean_per_utterance": mean_utt_wer}


def char_error_rate(refs: List[str], hyps: List[str]) -> float:
    total_edits, total_chars = 0, 0
    for r, h in zip(refs, hyps):
        r_ch, h_ch = list(r.lower()), list(h.lower())
        total_edits += _edit_distance(r_ch, h_ch)
        total_chars += len(r_ch)
    return total_edits / total_chars if total_chars > 0 else float("nan")


def field_classification_metrics(y_true: List[str], y_pred: List[str], prefix: str) -> Dict[str, float]:
    """
    Accuracy alone hides class imbalance. Macro-avg P/R/F1 weight every
    class equally (exposes rare-class collapse); weighted-avg accounts for
    class frequency. sklearn accepts the label strings directly, so no id
    mapping is needed here.
    """
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score

    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        f"{prefix}_acc": acc,
        f"{prefix}_precision_macro": p_macro,
        f"{prefix}_recall_macro": r_macro,
        f"{prefix}_f1_macro": f1_macro,
        f"{prefix}_precision_weighted": p_weighted,
        f"{prefix}_recall_weighted": r_weighted,
        f"{prefix}_f1_weighted": f1_weighted,
    }


def print_field_classification_report(y_true: List[str], y_pred: List[str], field_name: str):
    """Full per-class precision/recall/F1/support table, using the string
    labels directly -- shows exactly which action/object/location classes
    the model struggles with on this noisy set."""
    from sklearn.metrics import classification_report

    labels = sorted(set(y_true) | set(y_pred))
    print(f"\n=== {field_name} classification report ===")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))


# =========================================================================== #
# 3. Batch inference over a full CSV split, writes a predictions file
# =========================================================================== #
def run_inference(
    root_dir: str,
    csv_path: str,
    checkpoint_path: str,
    vocab_path: str,
    output_csv: str,
    whisper_model_name: str = "openai/whisper-small",
    bert_model_name: str = "bert-base-uncased",
    batch_size: int = 8,
    num_workers: int = 4,
    sample_rate: int = 16000,
    device: Optional[str] = None,
    print_classification_reports: bool = False,
    metrics_output_json: Optional[str] = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    engine = FSCNoisyInference(
        checkpoint_path=checkpoint_path,
        vocab_path=vocab_path,
        whisper_model_name=whisper_model_name,
        bert_model_name=bert_model_name,
        device=device,
    )

    dataset = NoisyAudioInferenceDataset(csv_path, root_dir, engine.vocab, sample_rate)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_raw,
    )

    rows = []
    true_actions, pred_actions = [], []
    true_objects, pred_objects = [], []
    true_locations, pred_locations = [], []
    reference_transcripts, asr_transcripts = [], []

    for batch in loader:
        preds = engine.predict_batch(batch["waveforms"], sample_rate)

        for i, path in enumerate(batch["paths"]):
            row = {
                "path": path,
                "asr_transcript": preds["transcripts"][i],
                "pred_action": preds["pred_action"][i],
                "pred_object": preds["pred_object"][i],
                "pred_location": preds["pred_location"][i],
            }
            if "actions" in batch:
                true_a, true_o, true_l = batch["actions"][i], batch["objects"][i], batch["locations"][i]
                row.update({"true_action": true_a, "true_object": true_o, "true_location": true_l})

                a_ok = preds["pred_action"][i] == true_a
                o_ok = preds["pred_object"][i] == true_o
                l_ok = preds["pred_location"][i] == true_l
                row.update({"action_correct": a_ok, "object_correct": o_ok, "location_correct": l_ok,
                            "exact_match": a_ok and o_ok and l_ok})

                true_actions.append(true_a); pred_actions.append(preds["pred_action"][i])
                true_objects.append(true_o); pred_objects.append(preds["pred_object"][i])
                true_locations.append(true_l); pred_locations.append(preds["pred_location"][i])

            if "reference_transcripts" in batch:
                ref = batch["reference_transcripts"][i]
                row["reference_transcript"] = ref
                reference_transcripts.append(ref)
                asr_transcripts.append(preds["transcripts"][i])

            rows.append(row)

        print(f"  ...{len(rows)}/{len(dataset)} processed")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv, index=False)
    print(f"\nPredictions written to {output_csv}")

    metrics: Dict = {}

    # --- action/object/location classification metrics ---
    if true_actions:
        exact_match = sum(
            1 for a, o, l, ta, to, tl in zip(pred_actions, pred_objects, pred_locations,
                                              true_actions, true_objects, true_locations)
            if a == ta and o == to and l == tl
        ) / len(true_actions)

        metrics["exact_match_acc"] = exact_match
        metrics.update(field_classification_metrics(true_actions, pred_actions, "action"))
        metrics.update(field_classification_metrics(true_objects, pred_objects, "object"))
        metrics.update(field_classification_metrics(true_locations, pred_locations, "location"))

        print(f"\nexact_match_acc={exact_match:.4f}  (n={len(true_actions)})")
        for field, y_t, y_p in [("action", true_actions, pred_actions),
                                 ("object", true_objects, pred_objects),
                                 ("location", true_locations, pred_locations)]:
            print(
                f"{field:>9}: acc={metrics[f'{field}_acc']:.4f}  "
                f"f1_macro={metrics[f'{field}_f1_macro']:.4f}  "
                f"f1_weighted={metrics[f'{field}_f1_weighted']:.4f}"
            )
            if print_classification_reports:
                print_field_classification_report(y_t, y_p, field)
    else:
        print("\nNo action/object/location columns found in CSV -- predictions only, no classification metrics.")

    # --- ASR robustness metrics (only meaningful if a clean-speech reference exists) ---
    if reference_transcripts:
        wer_metrics = word_error_rate(reference_transcripts, asr_transcripts)
        cer = char_error_rate(reference_transcripts, asr_transcripts)
        metrics.update(wer_metrics)
        metrics["cer"] = cer
        print(
            f"\nASR robustness vs. clean-speech reference text:\n"
            f"  WER (corpus)          = {wer_metrics['wer_corpus']:.4f}\n"
            f"  WER (mean per-utt)    = {wer_metrics['wer_mean_per_utterance']:.4f}\n"
            f"  CER                   = {cer:.4f}"
        )
    else:
        print("\nNo `transcription` column in CSV -- skipping WER/CER (no clean-speech reference to compare against).")

    if metrics_output_json:
        with open(metrics_output_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics written to {metrics_output_json}")

    return out_df, metrics


# =========================================================================== #
# 4. CLI
# =========================================================================== #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True, help="Root dir of your noisy dataset copy")
    parser.add_argument("--csv", type=str, default="data/test_data.csv", help="Relative to root_dir")
    parser.add_argument("--checkpoint", type=str, default="fsc_semi_cascaded_slu_best.pt")
    parser.add_argument("--vocab", type=str, default="fsc_label_vocab.json")
    parser.add_argument("--output_csv", type=str, default="predictions.csv")
    parser.add_argument("--whisper_model_name", type=str, default="openai/whisper-small")
    parser.add_argument("--bert_model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--print_classification_reports", action="store_true",
                         help="Print full per-class precision/recall/F1 tables for action/object/location.")
    parser.add_argument("--metrics_output_json", type=str, default=None,
                         help="Optional path to save all computed metrics as JSON.")
    args = parser.parse_args()

    csv_path = os.path.join(args.root_dir, args.csv)
    run_inference(
        root_dir=args.root_dir,
        csv_path=csv_path,
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        output_csv=args.output_csv,
        whisper_model_name=args.whisper_model_name,
        bert_model_name=args.bert_model_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        print_classification_reports=args.print_classification_reports,
        metrics_output_json=args.metrics_output_json,
    )
