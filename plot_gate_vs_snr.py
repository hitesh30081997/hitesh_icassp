"""
Diagnostic: Acoustic-Reliance Gate vs. SNR
===========================================

Runs a saved checkpoint (base SemiCascadedSLU_FSC or the confidence-gated
variant) across a sweep of SNR levels and plots the model's own gate value
`g` (how much it leans on the acoustic branch vs. the text/ASR branch)
alongside exact_match_acc.

This turns the gate from invisible plumbing into an interpretability
result: if g rises monotonically as SNR drops, that's direct evidence the
model is doing what the architecture claims -- falling back on raw audio
precisely when the ASR transcript becomes unreliable. If it's flat, that's
also a useful (negative) finding worth reporting.

Reuses the noise-injection + per-SNR ASR caching machinery from
evaluate_fsc_snr_robustness.py, and (for --model_type confidence_gated)
the confidence extraction from confidence_gated_fusion.py.

Usage (base model):
    python plot_gate_vs_snr.py \\
        --root_dir fluent_speech_commands_dataset \\
        --checkpoint fsc_semi_cascaded_slu_best.pt \\
        --vocab fsc_label_vocab.json \\
        --model_type base \\
        --snr_levels clean,20,15,10,5,0,-5

Usage (confidence-gated model):
    python plot_gate_vs_snr.py \\
        --root_dir fluent_speech_commands_dataset \\
        --checkpoint fsc_confidence_gated_best.pt \\
        --vocab fsc_label_vocab.json \\
        --model_type confidence_gated \\
        --snr_levels clean,20,15,10,5,0,-5
"""

import argparse
import json
import os
from typing import Dict, List, Optional

# Must run before any `transformers`/`huggingface_hub` import.
from hf_offline_utils import enable_offline_mode
enable_offline_mode()

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import WhisperFeatureExtractor, AutoTokenizer

from fsc_semi_cascaded_slu import (
    FSCConfig,
    FSCLabelVocab,
    FSCCollator,
    SemiCascadedSLU_FSC,
    ASRTranscriber,
    load_and_resample,
)
from evaluate_fsc_snr_robustness import (
    SNRNoiseInjector,
    NoisyFluentSpeechCommandsDataset,
    build_asr_cache_for_snr,
    _snr_seed,
)
from confidence_gated_fusion import (
    SemiCascadedSLU_FSC_ConfidenceGated,
    ASRTranscriberWithConfidence,
    FSCConfidenceCollator,
)


# =========================================================================== #
# 1. Noisy dataset variant that also carries a per-utterance ASR confidence
#    (only needed for the confidence-gated model type)
# =========================================================================== #
class NoisyDatasetWithConfidence(NoisyFluentSpeechCommandsDataset):
    def __init__(self, *args, confidence_cache: Optional[Dict[str, float]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.confidence_cache = confidence_cache or {}

    def __getitem__(self, idx: int) -> Dict:
        item = super().__getitem__(idx)
        item["asr_confidence"] = float(self.confidence_cache.get(item["path"], 0.5))
        return item


def build_asr_and_confidence_cache_for_snr(
    transcriber: ASRTranscriberWithConfidence,
    wav_paths: List[str],
    root_dir: str,
    sample_rate: int,
    snr_db,
    noise_injector: Optional[SNRNoiseInjector],
    cache_path: str,
    batch_size: int = 16,
) -> Dict[str, Dict]:
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)

    todo = [p for p in wav_paths if p not in cache]
    print(f"[SNR={snr_db}] {len(todo)} / {len(wav_paths)} need ASR+confidence decoding")

    for i in range(0, len(todo), batch_size):
        batch_paths = todo[i : i + batch_size]
        waveforms = []
        for p in batch_paths:
            wav = load_and_resample(os.path.join(root_dir, p), sample_rate)
            if noise_injector is not None and snr_db not in (None, "clean"):
                wav = noise_injector.add_noise(wav, float(snr_db), _snr_seed(p, snr_db))
            waveforms.append(wav)

        texts, confs = transcriber.transcribe_batch(waveforms, sample_rate)
        for p, t, c in zip(batch_paths, texts, confs):
            cache[p] = {"text": t, "confidence": c}

        if (i // batch_size) % 20 == 0:
            with open(cache_path, "w") as f:
                json.dump(cache, f)
            print(f"  ...{i + len(batch_paths)}/{len(todo)} decoded")

    with open(cache_path, "w") as f:
        json.dump(cache, f)
    return cache


# =========================================================================== #
# 2. Model loading (either variant)
# =========================================================================== #
def load_model(model_type: str, checkpoint_path: str, vocab: FSCLabelVocab, cfg: FSCConfig):
    model_cls = SemiCascadedSLU_FSC_ConfidenceGated if model_type == "confidence_gated" else SemiCascadedSLU_FSC
    model = model_cls(
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
# 3. Forward pass that extracts the mean acoustic-reliance gate value
# =========================================================================== #
@torch.no_grad()
def evaluate_with_gate(model, loader: DataLoader, device: str, model_type: str) -> Dict[str, float]:
    total = 0
    correct_all = 0
    gate_sum = 0.0
    conf_sum = 0.0
    have_confidence = False

    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        kwargs = dict(
            input_features=batch["input_features"],
            input_ids=batch["input_ids"],
            text_attention_mask=batch["text_attention_mask"],
        )
        if model_type == "confidence_gated":
            kwargs["asr_confidence"] = batch["asr_confidence"]
            have_confidence = True

        out = model(**kwargs)

        pred_action = out["action_logits"].argmax(-1)
        pred_object = out["object_logits"].argmax(-1)
        pred_location = out["location_logits"].argmax(-1)
        bsz = batch["action_labels"].size(0)

        correct_all += (
            (pred_action == batch["action_labels"])
            & (pred_object == batch["object_labels"])
            & (pred_location == batch["location_labels"])
        ).sum().item()
        total += bsz

        # mean gate value per sample, over valid (non-pad) text tokens and
        # over the fusion_dim channel axis, so we get one scalar per sample
        text_mask = batch["text_attention_mask"].unsqueeze(-1).float()   # (B, L, 1)
        gate = out["acoustic_gate"]                                     # (B, L, fusion_dim)
        fusion_dim = gate.shape[-1]
        valid_tokens = text_mask.sum(dim=1)                              # (B, 1)
        gate_per_sample = (gate * text_mask).sum(dim=(1, 2)) / (valid_tokens.squeeze(-1) * fusion_dim).clamp(min=1e-6)
        gate_sum += gate_per_sample.sum().item()

        if have_confidence:
            conf_sum += batch["asr_confidence"].sum().item()

    metrics = {
        "exact_match_acc": correct_all / total,
        "mean_acoustic_gate": gate_sum / total,
    }
    if have_confidence:
        metrics["mean_asr_confidence"] = conf_sum / total
    return metrics


# =========================================================================== #
# 4. Sweep across SNR levels
# =========================================================================== #
def run_gate_sweep(
    cfg: FSCConfig,
    checkpoint_path: str,
    vocab_path: str,
    model_type: str,
    snr_levels: List,
    noise_dir: Optional[str] = None,
) -> List[Dict]:
    vocab = FSCLabelVocab.load(vocab_path)
    model = load_model(model_type, checkpoint_path, vocab, cfg)

    feature_extractor = WhisperFeatureExtractor.from_pretrained(cfg.whisper_model_name, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.bert_model_name, local_files_only=True)

    noise_injector = SNRNoiseInjector(noise_dir=noise_dir, sample_rate=cfg.sample_rate)
    test_csv_path = os.path.join(cfg.root_dir, cfg.test_csv)
    all_paths = pd.read_csv(test_csv_path)["path"].tolist()

    if model_type == "confidence_gated":
        transcriber = ASRTranscriberWithConfidence(cfg.whisper_model_name, cfg.device)
        collator = FSCConfidenceCollator(feature_extractor, tokenizer, sample_rate=cfg.sample_rate)
    else:
        transcriber = ASRTranscriber(cfg.whisper_model_name, cfg.device)
        collator = FSCCollator(feature_extractor, tokenizer, sample_rate=cfg.sample_rate)

    results = []
    for snr in snr_levels:
        label = "clean" if snr in (None, "clean") else f"{snr}dB"
        print(f"\n{'=' * 60}\nSNR = {label}\n{'=' * 60}")

        if model_type == "confidence_gated":
            cache_path = f"fsc_asr_confidence_cache_test_snr_{label}.json"
            combined_cache = build_asr_and_confidence_cache_for_snr(
                transcriber, all_paths, cfg.root_dir, cfg.sample_rate, snr, noise_injector, cache_path
            )
            text_cache = {p: v["text"] for p, v in combined_cache.items()}
            confidence_cache = {p: v["confidence"] for p, v in combined_cache.items()}

            test_ds = NoisyDatasetWithConfidence(
                test_csv_path, root_dir=cfg.root_dir, vocab=vocab, sample_rate=cfg.sample_rate,
                max_audio_seconds=cfg.max_audio_seconds, asr_cache=text_cache,
                use_ground_truth_transcript=False, noise_injector=noise_injector, snr_db=snr,
                confidence_cache=confidence_cache,
            )
        else:
            cache_path = f"fsc_asr_cache_test_snr_{label}.json"
            text_cache = build_asr_cache_for_snr(
                transcriber, all_paths, cfg.root_dir, cfg.sample_rate, snr, noise_injector, cache_path
            )
            test_ds = NoisyFluentSpeechCommandsDataset(
                test_csv_path, root_dir=cfg.root_dir, vocab=vocab, sample_rate=cfg.sample_rate,
                max_audio_seconds=cfg.max_audio_seconds, asr_cache=text_cache,
                use_ground_truth_transcript=False, noise_injector=noise_injector, snr_db=snr,
            )

        test_loader = DataLoader(
            test_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, collate_fn=collator,
        )

        metrics = evaluate_with_gate(model, test_loader, cfg.device, model_type)
        metrics["snr"] = label
        metrics["snr_numeric"] = 30.0 if snr in (None, "clean") else float(snr)  # 'clean' plotted far right
        results.append(metrics)
        print(f"[SNR={label}] mean_acoustic_gate={metrics['mean_acoustic_gate']:.4f}  "
              f"exact_match_acc={metrics['exact_match_acc']:.4f}"
              + (f"  mean_asr_confidence={metrics['mean_asr_confidence']:.4f}" if "mean_asr_confidence" in metrics else ""))

    del transcriber
    if cfg.device == "cuda":
        torch.cuda.empty_cache()

    return results


# =========================================================================== #
# 5. Plotting
# =========================================================================== #
def plot_gate_vs_snr(results: List[Dict], output_path: str, model_type: str):
    import matplotlib.pyplot as plt

    results_sorted = sorted(results, key=lambda r: r["snr_numeric"], reverse=True)
    labels = [r["snr"] for r in results_sorted]
    x = list(range(len(labels)))
    gate_vals = [r["mean_acoustic_gate"] for r in results_sorted]
    acc_vals = [r["exact_match_acc"] for r in results_sorted]

    fig, ax1 = plt.subplots(figsize=(8, 5.5))

    color_gate = "#c8781f"
    ax1.set_xlabel("SNR condition (decreasing noise \u2192 right = clean)")
    ax1.set_ylabel("Mean acoustic-reliance gate  $g$", color=color_gate)
    l1, = ax1.plot(x, gate_vals, marker="o", color=color_gate, linewidth=2.2, label="Mean gate $g$")
    ax1.tick_params(axis="y", labelcolor=color_gate)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    color_acc = "#3b6fa0"
    ax2.set_ylabel("Exact-match accuracy", color=color_acc)
    l2, = ax2.plot(x, acc_vals, marker="s", linestyle="--", color=color_acc, linewidth=2.0, label="Exact-match acc.")
    ax2.tick_params(axis="y", labelcolor=color_acc)
    ax2.set_ylim(0, 1)

    title_variant = "Confidence-Gated" if model_type == "confidence_gated" else "Base"
    plt.title(f"Acoustic-Reliance Gate vs. SNR ({title_variant} Model)", fontsize=12.5, fontweight="bold")
    ax1.legend(handles=[l1, l2], loc="lower left", frameon=False, fontsize=9.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    svg_path = os.path.splitext(output_path)[0] + ".svg"
    plt.savefig(svg_path, bbox_inches="tight")
    print(f"\nPlot saved to {output_path} and {svg_path}")


# =========================================================================== #
# 6. CLI
# =========================================================================== #
def _parse_snr_levels(raw: str) -> List:
    levels = []
    for tok in raw.split(","):
        tok = tok.strip()
        levels.append("clean" if tok.lower() == "clean" else float(tok))
    return levels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--test_csv", type=str, default="data/test_data.csv")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--vocab", type=str, default="fsc_label_vocab.json")
    parser.add_argument("--model_type", type=str, choices=["base", "confidence_gated"], default="base")
    parser.add_argument("--snr_levels", type=str, default="clean,20,15,10,5,0,-5")
    parser.add_argument("--noise_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_plot", type=str, default="gate_vs_snr.png")
    parser.add_argument("--metrics_output_json", type=str, default="gate_vs_snr_metrics.json")
    args = parser.parse_args()

    cfg = FSCConfig(root_dir=args.root_dir, test_csv=args.test_csv, batch_size=args.batch_size)
    snr_levels = _parse_snr_levels(args.snr_levels)

    results = run_gate_sweep(
        cfg=cfg,
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        model_type=args.model_type,
        snr_levels=snr_levels,
        noise_dir=args.noise_dir,
    )

    with open(args.metrics_output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw metrics saved to {args.metrics_output_json}")

    plot_gate_vs_snr(results, args.output_plot, args.model_type)
