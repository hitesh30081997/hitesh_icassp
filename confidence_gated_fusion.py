"""
Confidence-Gated Fusion for Semi-Cascaded SLU
==============================================

Extends the base SemiCascadedSLU_FSC model so the acoustic/text gate is
conditioned on Whisper's own ASR decoding confidence, not learned purely
from the fused/text representations in isolation.

Motivation: the original gate
    g = sigmoid(W [fused ; text_proj])
has no explicit signal about *how much to trust the transcript it just
received*. It has to infer unreliability indirectly from the text/acoustic
representations themselves. Whisper already computes this directly during
decoding (per-token probabilities) and throws it away after generating the
final string. This script recovers it and feeds it back in:

    confidence = geometric mean of per-token probabilities of the
                 generated ASR transcript (mean log-prob, exponentiated)
    g = sigmoid(W [fused ; text_proj ; confidence])

This makes the gate's behavior falsifiable and interpretable: you can now
directly test "does the model down-weight the transcript when Whisper
itself is unsure?", not just observe an accuracy number.

Everything not related to confidence extraction / gating is reused as-is
from fsc_semi_cascaded_slu.py (encoders, fusion block, dataset, config).
"""

import json
import os
from typing import Dict, List, Optional, Tuple

# Must run before any `transformers`/`huggingface_hub` import.
from hf_offline_utils import enable_offline_mode
enable_offline_mode()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    AutoTokenizer,
)

from fsc_semi_cascaded_slu import (
    FSCConfig,
    FSCLabelVocab,
    FluentSpeechCommandsDataset,
    FSCCollator,
    SemiCascadedSLU_FSC,
    load_and_resample,
    _field_metrics,
)


# =========================================================================== #
# 1. ASR transcriber that also returns a per-utterance confidence score
# =========================================================================== #
class ASRTranscriberWithConfidence:
    """
    Same role as fsc_semi_cascaded_slu.ASRTranscriber, but also extracts a
    confidence score per decoded utterance from Whisper's own generation
    scores -- no separate model or extra forward pass needed, this comes
    for free from generate(output_scores=True).

    confidence = exp( mean_t log P(token_t | token_<t, audio) )
    i.e. the geometric mean of per-token probabilities along the greedy/beam
    decoding path. 1.0 = fully confident, values closer to 0 = the decoder
    was picking low-probability tokens (typically under noise/ambiguity).
    """

    def __init__(self, model_name: str, device: str, language: str = "en"):
        self.processor = WhisperProcessor.from_pretrained(model_name, local_files_only=True)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name, local_files_only=True).to(device)
        self.model.eval()
        self.model.generation_config.forced_decoder_ids = None
        # Set these directly on the generation_config (rather than passing
        # as generate() kwargs every call) so transformers' internal config
        # validation sees them already merged and doesn't emit a spurious
        # "generation flags are not valid" warning -- output_scores DOES
        # get applied correctly either way, this just silences the
        # cosmetic false-positive.
        self.model.generation_config.output_scores = True
        self.model.generation_config.return_dict_in_generate = True
        self.device = device
        self.language = language

    @torch.no_grad()
    def transcribe_batch(self, waveforms: List, sample_rate: int) -> Tuple[List[str], List[float]]:
        inputs = self.processor(
            waveforms, sampling_rate=sample_rate, return_tensors="pt", return_attention_mask=True
        )
        input_features = inputs.input_features.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        gen_out = self.model.generate(
            input_features,
            attention_mask=attention_mask,
            language=self.language,
            task="transcribe",
        )

        texts = [
            t.strip() for t in self.processor.batch_decode(gen_out.sequences, skip_special_tokens=True)
        ]
        confidences = self._compute_confidence(gen_out)
        return texts, confidences

    def _compute_confidence(self, gen_out) -> List[float]:
        if not gen_out.scores:
            # nothing generated (e.g. immediate EOS) -- treat as neutral
            return [0.5] * gen_out.sequences.size(0)

        # scores: tuple of length T_gen, each (B, vocab) logits for that step
        scores = torch.stack(gen_out.scores, dim=1)          # (B, T_gen, vocab)
        log_probs = F.log_softmax(scores, dim=-1)

        # the generated tokens corresponding to those T_gen steps are the
        # LAST T_gen tokens of the full output sequence (the earlier ones
        # are the fixed decoder-start / language / task prompt tokens)
        gen_tokens = gen_out.sequences[:, -scores.shape[1]:]   # (B, T_gen)
        token_logp = torch.gather(log_probs, 2, gen_tokens.unsqueeze(-1)).squeeze(-1)  # (B, T_gen)

        # mask out anything at/after the first EOS so padding doesn't dilute
        # the confidence estimate
        eos_id = self.processor.tokenizer.eos_token_id
        mask = torch.ones_like(gen_tokens, dtype=torch.bool)
        for b in range(gen_tokens.size(0)):
            eos_pos = (gen_tokens[b] == eos_id).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                mask[b, eos_pos[0].item() + 1:] = False

        token_logp = token_logp.masked_fill(~mask, 0.0)
        valid_counts = mask.sum(dim=1).clamp(min=1)
        mean_logp = token_logp.sum(dim=1) / valid_counts       # (B,)

        confidence = torch.exp(mean_logp).clamp(0.0, 1.0)
        return confidence.cpu().tolist()

    def build_cache(
        self, wav_paths: List[str], root_dir: str, sample_rate: int, cache_path: str, batch_size: int = 16
    ) -> Dict[str, Dict]:
        """Writes {path: {"text": ..., "confidence": ...}} to cache_path."""
        cache = {}
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                cache = json.load(f)

        todo = [p for p in wav_paths if p not in cache]
        print(f"[ASRTranscriberWithConfidence] {len(todo)} / {len(wav_paths)} need decoding")

        for i in range(0, len(todo), batch_size):
            batch_paths = todo[i : i + batch_size]
            waveforms = [load_and_resample(os.path.join(root_dir, p), sample_rate) for p in batch_paths]
            texts, confs = self.transcribe_batch(waveforms, sample_rate)
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
# 2. Dataset / collator that carry the confidence score through to the model
# =========================================================================== #
class FluentSpeechCommandsDatasetWithConfidence(FluentSpeechCommandsDataset):
    """
    Same as FluentSpeechCommandsDataset, but each item also carries an
    `asr_confidence` scalar sourced from the confidence cache built above.
    Falls back to a neutral 0.5 for any utterance not (yet) in the cache.
    """

    def __init__(self, *args, confidence_cache: Optional[Dict[str, float]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.confidence_cache = confidence_cache or {}

    def __getitem__(self, idx: int) -> Dict:
        item = super().__getitem__(idx)
        item["asr_confidence"] = float(self.confidence_cache.get(item["path"], 0.5))
        return item


class FSCConfidenceCollator(FSCCollator):
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        out = super().__call__(batch)
        out["asr_confidence"] = torch.tensor([b["asr_confidence"] for b in batch], dtype=torch.float)
        return out


# =========================================================================== #
# 3. Confidence-gated model
# =========================================================================== #
class SemiCascadedSLU_FSC_ConfidenceGated(SemiCascadedSLU_FSC):
    """
    Identical to SemiCascadedSLU_FSC except the gate additionally sees a
    scalar ASR confidence per utterance, broadcast across text-token
    positions:

        g = sigmoid(W [fused ; text_proj ; confidence])

    `asr_confidence` is optional in forward() -- if omitted, a neutral 0.5
    is substituted so the model degrades gracefully to a confidence-blind
    gate (useful for quick sanity checks or reusing eval code that doesn't
    know about this input).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        fusion_dim = self.fusion.text_proj.out_features
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2 + 1, fusion_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        input_features: torch.Tensor,
        input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        acoustic_attention_mask: torch.Tensor = None,
        asr_confidence: torch.Tensor = None,
        action_labels: torch.Tensor = None,
        object_labels: torch.Tensor = None,
        location_labels: torch.Tensor = None,
        **kwargs,
    ):
        acoustic_states = self.acoustic_encoder(input_features)
        text_states = self.text_encoder(input_ids, text_attention_mask)

        fused, attn_weights = self.fusion(
            text_states, text_attention_mask, acoustic_states, acoustic_attention_mask
        )
        text_proj = self.fusion.text_proj(text_states)

        B, L, _ = fused.shape
        if asr_confidence is None:
            asr_confidence = torch.full((B,), 0.5, device=fused.device, dtype=fused.dtype)
        conf_broadcast = asr_confidence.view(B, 1, 1).expand(B, L, 1)

        g = self.gate(torch.cat([fused, text_proj, conf_broadcast], dim=-1))
        mixed = g * fused + (1 - g) * text_proj
        mixed = self.dropout(mixed)

        mask = text_attention_mask.unsqueeze(-1).float()
        pooled = (mixed * mask).sum(1) / mask.sum(1).clamp(min=1e-6)

        action_logits = self.action_head(pooled)
        object_logits = self.object_head(pooled)
        location_logits = self.location_head(pooled)

        output = {
            "action_logits": action_logits,
            "object_logits": object_logits,
            "location_logits": location_logits,
            "cross_attn_weights": attn_weights,
            "acoustic_gate": g,               # (B, L, fusion_dim) reliance score, now confidence-aware
            "asr_confidence": asr_confidence,  # kept in output for logging/diagnostics
        }

        if action_labels is not None and object_labels is not None and location_labels is not None:
            action_loss = F.cross_entropy(action_logits, action_labels)
            object_loss = F.cross_entropy(object_logits, object_labels)
            location_loss = F.cross_entropy(location_logits, location_labels)
            output["loss"] = action_loss + object_loss + location_loss
            output["action_loss"] = action_loss
            output["object_loss"] = object_loss
            output["location_loss"] = location_loss

        return output


# =========================================================================== #
# 4. Data pipeline (builds text + confidence cache, then dataloaders)
# =========================================================================== #
def build_confidence_dataloaders(
    cfg: FSCConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader, FSCLabelVocab]:
    import pandas as pd

    train_csv = os.path.join(cfg.root_dir, cfg.train_csv)
    valid_csv = os.path.join(cfg.root_dir, cfg.valid_csv)
    test_csv = os.path.join(cfg.root_dir, cfg.test_csv)

    vocab = FSCLabelVocab([train_csv, valid_csv, test_csv])
    vocab.save("fsc_label_vocab.json")

    transcriber = ASRTranscriberWithConfidence(cfg.whisper_model_name, cfg.device)
    all_paths = pd.concat(
        [pd.read_csv(p)["path"] for p in [train_csv, valid_csv, test_csv]]
    ).tolist()
    combined_cache_path = cfg.asr_cache_path.replace(".json", "_with_confidence.json")
    combined_cache = transcriber.build_cache(
        all_paths, cfg.root_dir, cfg.sample_rate, combined_cache_path
    )
    del transcriber
    if cfg.device == "cuda":
        torch.cuda.empty_cache()

    text_cache = {p: v["text"] for p, v in combined_cache.items()}
    confidence_cache = {p: v["confidence"] for p, v in combined_cache.items()}

    feature_extractor = WhisperFeatureExtractor.from_pretrained(cfg.whisper_model_name, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.bert_model_name, local_files_only=True)
    collator = FSCConfidenceCollator(feature_extractor, tokenizer, sample_rate=cfg.sample_rate)

    common_kwargs = dict(
        root_dir=cfg.root_dir,
        vocab=vocab,
        sample_rate=cfg.sample_rate,
        max_audio_seconds=cfg.max_audio_seconds,
        asr_cache=text_cache,
        use_ground_truth_transcript=False,
        confidence_cache=confidence_cache,
    )
    train_ds = FluentSpeechCommandsDatasetWithConfidence(train_csv, **common_kwargs)
    valid_ds = FluentSpeechCommandsDatasetWithConfidence(valid_csv, **common_kwargs)
    test_ds = FluentSpeechCommandsDatasetWithConfidence(test_csv, **common_kwargs)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collator, drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collator,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collator,
    )
    return train_loader, valid_loader, test_loader, vocab


# =========================================================================== #
# 5. Evaluation (mirrors fsc_semi_cascaded_slu.evaluate, but passes confidence)
# =========================================================================== #
@torch.no_grad()
def evaluate_confidence_gated(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    total, correct_all = 0, 0
    total_loss = 0.0
    all_action_true, all_action_pred = [], []
    all_object_true, all_object_pred = [], []
    all_location_true, all_location_pred = [], []
    mean_gate_sum, mean_conf_sum = 0.0, 0.0

    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(
            input_features=batch["input_features"],
            input_ids=batch["input_ids"],
            text_attention_mask=batch["text_attention_mask"],
            asr_confidence=batch["asr_confidence"],
            action_labels=batch["action_labels"],
            object_labels=batch["object_labels"],
            location_labels=batch["location_labels"],
        )
        bsz = batch["action_labels"].size(0)
        total_loss += out["loss"].item() * bsz

        pred_action = out["action_logits"].argmax(-1)
        pred_object = out["object_logits"].argmax(-1)
        pred_location = out["location_logits"].argmax(-1)

        correct_all += (
            (pred_action == batch["action_labels"])
            & (pred_object == batch["object_labels"])
            & (pred_location == batch["location_labels"])
        ).sum().item()
        total += bsz

        all_action_true += batch["action_labels"].cpu().tolist()
        all_action_pred += pred_action.cpu().tolist()
        all_object_true += batch["object_labels"].cpu().tolist()
        all_object_pred += pred_object.cpu().tolist()
        all_location_true += batch["location_labels"].cpu().tolist()
        all_location_pred += pred_location.cpu().tolist()

        text_mask = batch["text_attention_mask"].unsqueeze(-1).float()
        gate_mean_per_sample = (out["acoustic_gate"] * text_mask).sum(dim=(1, 2)) / (
            text_mask.sum(dim=(1, 2)) * out["acoustic_gate"].shape[-1] / text_mask.shape[-1]
        ).clamp(min=1e-6)
        mean_gate_sum += gate_mean_per_sample.sum().item()
        mean_conf_sum += batch["asr_confidence"].sum().item()

    metrics = {
        "loss": total_loss / total,
        "exact_match_acc": correct_all / total,
        "mean_acoustic_gate": mean_gate_sum / total,   # avg acoustic reliance score
        "mean_asr_confidence": mean_conf_sum / total,  # avg Whisper decode confidence
    }
    metrics.update(_field_metrics(all_action_true, all_action_pred, "action"))
    metrics.update(_field_metrics(all_object_true, all_object_pred, "object"))
    metrics.update(_field_metrics(all_location_true, all_location_pred, "location"))
    return metrics


# =========================================================================== #
# 6. Training loop
# =========================================================================== #
def train_confidence_gated(cfg: FSCConfig):
    train_loader, valid_loader, test_loader, vocab = build_confidence_dataloaders(cfg)

    model = SemiCascadedSLU_FSC_ConfidenceGated(
        num_actions=vocab.num_actions,
        num_objects=vocab.num_objects,
        num_locations=vocab.num_locations,
        whisper_model_name=cfg.whisper_model_name,
        bert_model_name=cfg.bert_model_name,
        whisper_layer=cfg.whisper_layer,
        freeze_whisper=True,
        freeze_bert=False,
    ).to(cfg.device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr)

    best_valid_acc = -1.0
    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            batch = {k: (v.to(cfg.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

            out = model(
                input_features=batch["input_features"],
                input_ids=batch["input_ids"],
                text_attention_mask=batch["text_attention_mask"],
                asr_confidence=batch["asr_confidence"],
                action_labels=batch["action_labels"],
                object_labels=batch["object_labels"],
                location_labels=batch["location_labels"],
            )
            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

        valid_metrics = evaluate_confidence_gated(model, valid_loader, cfg.device)
        print(f"[epoch {epoch}] train_loss={running_loss / len(train_loader):.4f} valid={valid_metrics}")

        if valid_metrics["exact_match_acc"] > best_valid_acc:
            best_valid_acc = valid_metrics["exact_match_acc"]
            torch.save(model.state_dict(), "fsc_confidence_gated_best.pt")
            print(f"  -> new best ({best_valid_acc:.4f}), checkpoint saved")

    test_metrics = evaluate_confidence_gated(model, test_loader, cfg.device)
    print(f"[final test] {test_metrics}")
    return model, vocab, test_metrics


if __name__ == "__main__":
    cfg = FSCConfig(
        root_dir="fluent_speech_commands_dataset",
        batch_size=8,
        epochs=10,
    )
    train_confidence_gated(cfg)
