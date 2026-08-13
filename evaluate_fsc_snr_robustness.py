"""
Evaluate a saved Semi-Cascaded SLU checkpoint on the Fluent Speech Commands
test (or valid) split under different additive-noise SNR conditions.

Assumes:
  * `fsc_semi_cascaded_slu.py` (the training script) is importable from the
    same directory / on PYTHONPATH.
  * A checkpoint saved by that script, e.g. `fsc_semi_cascaded_slu_best.pt`.
  * A vocab file saved by that script, e.g. `fsc_label_vocab.json`.
  * The dataset root_dir / train / valid / test CSV layout is the SAME as
    the original Fluent Speech Commands release (this script only reads
    the wavs and re-injects noise on the fly -- it does not need a
    pre-noised copy of the dataset on disk).

For each requested SNR level (in dB) this script:
  1. Builds a deterministic noisy version of every test utterance
     (same noise realization every run, so results are reproducible).
  2. Decodes a fresh Whisper ASR hypothesis on THAT noisy audio and caches
     it separately per SNR level (low-SNR audio should produce worse ASR
     transcripts -- that degradation is exactly what the acoustic branch
     is supposed to help recover from).
  3. Runs the saved model and reports action/object/location accuracy +
     precision/recall/F1, and the exact_match_acc headline metric.
  4. Prints a summary table across all SNR levels (+ a clean baseline).

Usage:
    python evaluate_fsc_snr_robustness.py \\
        --root_dir fluent_speech_commands_dataset \\
        --test_csv data/test_data.csv \\
        --checkpoint fsc_semi_cascaded_slu_best.pt \\
        --vocab fsc_label_vocab.json \\
        --snr_levels clean,20,15,10,5,0,-5 \\
        --noise_dir /path/to/noise_wavs   # optional; omit for white noise
"""

import argparse
import hashlib
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader

from fsc_semi_cascaded_slu import (
    FSCConfig,
    FSCLabelVocab,
    FluentSpeechCommandsDataset,
    FSCCollator,
    SemiCascadedSLU_FSC,
    ASRTranscriber,
    load_and_resample,
    evaluate,
    print_classification_reports,
)
from transformers import WhisperFeatureExtractor, AutoTokenizer


# =========================================================================== #
# 1. SNR-controlled additive noise
# =========================================================================== #
class SNRNoiseInjector:
    """
    Adds noise to a waveform so the resulting signal has a specified SNR
    (dB). If `noise_dir` is given, noise is sampled from real recordings in
    that folder (e.g. MUSAN/NOISEX-92 -- more realistic); otherwise falls
    back to white Gaussian noise.
    """

    def __init__(self, noise_dir: Optional[str] = None, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.noise_files: List[str] = []
        if noise_dir is not None:
            self.noise_files = [
                os.path.join(noise_dir, f)
                for f in os.listdir(noise_dir)
                if f.lower().endswith((".wav", ".flac"))
            ]
            if not self.noise_files:
                raise ValueError(f"No .wav/.flac files found in noise_dir={noise_dir}")

    def _get_noise_segment(self, length: int, rng: np.random.RandomState) -> np.ndarray:
        if self.noise_files:
            noise_path = self.noise_files[rng.randint(len(self.noise_files))]
            noise, sr = torchaudio.load(noise_path)
            if noise.shape[0] > 1:
                noise = noise.mean(dim=0, keepdim=True)
            if sr != self.sample_rate:
                noise = torchaudio.functional.resample(noise, sr, self.sample_rate)
            noise = noise.squeeze(0).numpy()
            if len(noise) < length:
                reps = int(np.ceil(length / len(noise)))
                noise = np.tile(noise, reps)
            start = rng.randint(0, max(1, len(noise) - length + 1))
            return noise[start : start + length].astype(np.float32)
        else:
            return rng.randn(length).astype(np.float32)

    def add_noise(self, waveform: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
        rng = np.random.RandomState(seed)
        noise = self._get_noise_segment(len(waveform), rng)

        sig_power = float(np.mean(waveform ** 2)) + 1e-10
        noise_power = float(np.mean(noise ** 2)) + 1e-10

        target_noise_power = sig_power / (10 ** (snr_db / 10))
        scale = np.sqrt(target_noise_power / noise_power)
        noisy = waveform + scale * noise
        return noisy.astype(np.float32)


def _snr_seed(path: str, snr_db) -> int:
    """Deterministic per-(utterance, SNR) seed so noise is reproducible
    across the ASR-cache-building pass and the later evaluation pass."""
    key = f"{path}|{snr_db}"
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2 ** 32)


# =========================================================================== #
# 2. Noisy dataset variant
# =========================================================================== #
class NoisyFluentSpeechCommandsDataset(FluentSpeechCommandsDataset):
    """
    Identical to FluentSpeechCommandsDataset, except every waveform gets
    noise injected at a fixed target SNR before being returned. `snr_db`
    can also be `None` / "clean" to skip noise injection entirely (useful
    as the 0-degradation baseline point in the sweep).
    """

    def __init__(self, *args, noise_injector: Optional[SNRNoiseInjector], snr_db, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_injector = noise_injector
        self.snr_db = snr_db

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        wav_rel_path = row["path"]
        wav_path = os.path.join(self.root_dir, wav_rel_path)

        waveform = load_and_resample(wav_path, self.sample_rate)

        if self.noise_injector is not None and self.snr_db not in (None, "clean"):
            seed = _snr_seed(wav_rel_path, self.snr_db)
            waveform = self.noise_injector.add_noise(waveform, float(self.snr_db), seed)

        if len(waveform) > self.max_samples:
            waveform = waveform[: self.max_samples]  # deterministic crop for eval (no random offset)

        if self.use_ground_truth_transcript:
            text = str(row["transcription"])
        else:
            text = self.asr_cache.get(wav_rel_path, str(row["transcription"]))

        labels = self.vocab.encode(row["action"], row["object"], row["location"])
        return {
            "waveform": waveform,
            "text": text,
            "action_id": labels["action_id"],
            "object_id": labels["object_id"],
            "location_id": labels["location_id"],
            "intent_id": labels["intent_id"],
            "path": wav_rel_path,
        }


# =========================================================================== #
# 3. Per-SNR ASR cache building (decode the *noisy* audio, not clean)
# =========================================================================== #
def build_asr_cache_for_snr(
    transcriber: ASRTranscriber,
    wav_paths: List[str],
    root_dir: str,
    sample_rate: int,
    snr_db,
    noise_injector: Optional[SNRNoiseInjector],
    cache_path: str,
    batch_size: int = 16,
) -> Dict[str, str]:
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)

    todo = [p for p in wav_paths if p not in cache]
    print(f"[SNR={snr_db}] {len(todo)} / {len(wav_paths)} utterances need ASR decoding")

    for i in range(0, len(todo), batch_size):
        batch_paths = todo[i : i + batch_size]
        waveforms = []
        for p in batch_paths:
            wav = load_and_resample(os.path.join(root_dir, p), sample_rate)
            if noise_injector is not None and snr_db not in (None, "clean"):
                wav = noise_injector.add_noise(wav, float(snr_db), _snr_seed(p, snr_db))
            waveforms.append(wav)

        hyps = transcriber.transcribe_batch(waveforms, sample_rate)
        for p, h in zip(batch_paths, hyps):
            cache[p] = h

        if (i // batch_size) % 20 == 0:
            with open(cache_path, "w") as f:
                json.dump(cache, f)
            print(f"  ...{i + len(batch_paths)}/{len(todo)} decoded")

    with open(cache_path, "w") as f:
        json.dump(cache, f)
    return cache


# =========================================================================== #
# 4. Model loading
# =========================================================================== #
def load_trained_model(checkpoint_path: str, vocab: FSCLabelVocab, cfg: FSCConfig) -> SemiCascadedSLU_FSC:
    model = SemiCascadedSLU_FSC(
        num_actions=vocab.num_actions,
        num_objects=vocab.num_objects,
        num_locations=vocab.num_locations,
        whisper_model_name=cfg.whisper_model_name,
        bert_model_name=cfg.bert_model_name,
        whisper_layer=cfg.whisper_layer,
        freeze_whisper=True,
        freeze_bert=False,
    ).to(cfg.device)

    state_dict = torch.load(checkpoint_path, map_location=cfg.device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# =========================================================================== #
# 5. Main sweep
# =========================================================================== #
def evaluate_across_snr_levels(
    cfg: FSCConfig,
    checkpoint_path: str,
    vocab_path: str,
    snr_levels: List,
    noise_dir: Optional[str] = None,
    cache_dir: str = ".",
    print_per_class_report_for: Optional[List] = None,
):
    vocab = FSCLabelVocab.load(vocab_path)
    model = load_trained_model(checkpoint_path, vocab, cfg)

    feature_extractor = WhisperFeatureExtractor.from_pretrained(cfg.whisper_model_name, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.bert_model_name, local_files_only=True)
    collator = FSCCollator(feature_extractor, tokenizer, sample_rate=cfg.sample_rate)

    noise_injector = SNRNoiseInjector(noise_dir=noise_dir, sample_rate=cfg.sample_rate)

    test_csv_path = os.path.join(cfg.root_dir, cfg.test_csv)
    all_paths = None  # loaded once below

    transcriber = None
    if not cfg.use_ground_truth_transcript:
        transcriber = ASRTranscriber(cfg.whisper_model_name, cfg.device)

    results_table = []
    print_per_class_report_for = print_per_class_report_for or []

    for snr in snr_levels:
        label = "clean" if snr in (None, "clean") else f"{snr}dB"
        print(f"\n{'=' * 60}\nEvaluating at SNR = {label}\n{'=' * 60}")

        asr_cache = {}
        if transcriber is not None:
            import pandas as pd

            if all_paths is None:
                all_paths = pd.read_csv(test_csv_path)["path"].tolist()

            cache_path = os.path.join(cache_dir, f"fsc_asr_cache_test_snr_{label}.json")
            asr_cache = build_asr_cache_for_snr(
                transcriber,
                all_paths,
                cfg.root_dir,
                cfg.sample_rate,
                snr,
                noise_injector,
                cache_path,
            )

        test_ds = NoisyFluentSpeechCommandsDataset(
            test_csv_path,
            root_dir=cfg.root_dir,
            vocab=vocab,
            sample_rate=cfg.sample_rate,
            max_audio_seconds=cfg.max_audio_seconds,
            asr_cache=asr_cache,
            use_ground_truth_transcript=cfg.use_ground_truth_transcript,
            noise_injector=noise_injector,
            snr_db=snr,
        )
        test_loader = DataLoader(
            test_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, collate_fn=collator,
        )

        metrics = evaluate(model, test_loader, cfg.device)
        metrics["snr"] = label
        results_table.append(metrics)
        print(f"[SNR={label}] exact_match_acc={metrics['exact_match_acc']:.4f} "
              f"action_acc={metrics['action_acc']:.4f} "
              f"object_acc={metrics['object_acc']:.4f} "
              f"location_acc={metrics['location_acc']:.4f} "
              f"loss={metrics['loss']:.4f}")

        if snr in print_per_class_report_for:
            print_classification_reports(model, test_loader, cfg.device, vocab)

    if transcriber is not None:
        del transcriber
        if cfg.device == "cuda":
            torch.cuda.empty_cache()

    _print_summary_table(results_table)
    return results_table


def _print_summary_table(results_table: List[Dict]):
    cols = ["snr", "exact_match_acc", "action_acc", "object_acc", "location_acc",
            "action_f1_macro", "object_f1_macro", "location_f1_macro", "loss"]
    print(f"\n{'=' * 100}\nSUMMARY: robustness vs. SNR\n{'=' * 100}")
    header = " | ".join(f"{c:>18}" for c in cols)
    print(header)
    print("-" * len(header))
    for row in results_table:
        print(" | ".join(
            f"{row[c]:>18}" if isinstance(row[c], str) else f"{row[c]:>18.4f}"
            for c in cols
        ))


# =========================================================================== #
# 6. CLI entry point
# =========================================================================== #
def _parse_snr_levels(raw: str) -> List:
    levels = []
    for tok in raw.split(","):
        tok = tok.strip()
        levels.append("clean" if tok.lower() == "clean" else float(tok))
    return levels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="fluent_speech_commands_dataset")
    parser.add_argument("--test_csv", type=str, default="data/test_data.csv")
    parser.add_argument("--checkpoint", type=str, default="fsc_semi_cascaded_slu_best.pt")
    parser.add_argument("--vocab", type=str, default="fsc_label_vocab.json")
    parser.add_argument("--snr_levels", type=str, default="clean,20,15,10,5,0,-5")
    parser.add_argument("--noise_dir", type=str, default=None,
                         help="Optional folder of real noise .wav/.flac files. Omit for white noise.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--use_ground_truth_transcript", action="store_true",
                         help="Skip ASR decoding, feed clean ground-truth text (not a realistic robustness test).")
    parser.add_argument("--print_report_for", type=str, default="",
                         help="Comma-separated SNR values (or 'clean') to also print a full per-class report for.")
    args = parser.parse_args()

    cfg = FSCConfig(
        root_dir=args.root_dir,
        test_csv=args.test_csv,
        batch_size=args.batch_size,
        use_ground_truth_transcript=args.use_ground_truth_transcript,
    )

    snr_levels = _parse_snr_levels(args.snr_levels)
    print_report_for = _parse_snr_levels(args.print_report_for) if args.print_report_for else []

    evaluate_across_snr_levels(
        cfg=cfg,
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        snr_levels=snr_levels,
        noise_dir=args.noise_dir,
        print_per_class_report_for=print_report_for,
    )
