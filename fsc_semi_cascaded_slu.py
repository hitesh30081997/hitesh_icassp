"""
Semi-Cascaded SLU on Fluent Speech Commands (FSC)
==================================================

Adapts the frozen-Whisper-acoustic + ASR-transcript-BERT cross-attention
architecture to the Fluent Speech Commands dataset.

Key differences from a token-BIO-slot setup (e.g. SLURP):

  * FSC has NO word-level slot tags. Each utterance is labeled with three
    utterance-level categorical fields: `action`, `object`, `location`
    (e.g. "increase" / "heat" / "kitchen"). Some papers also use the
    triple joined as a single 31-way "intent" -- both options are
    supported below.
  * FSC ships a ground-truth `transcription` column, but the whole point
    of a *semi-cascaded* model is robustness to ASR errors, so we
    generate our own (possibly wrong) ASR hypothesis with Whisper and
    feed THAT to the text branch, not the ground-truth text. Ground
    truth transcription is only used as a fallback / for sanity checks.
  * The model's slot head is dropped (nothing to supervise it with) and
    replaced by three parallel heads: action / object / location.

Expected directory layout (standard FSC release):

    fluent_speech_commands_dataset/
        wavs/
            speakers/
                <speakerId>/
                    <utterance>.wav
        data/
            train_data.csv
            valid_data.csv
            test_data.csv

Each CSV has columns:
    ,path,speakerId,transcription,action,object,location

`path` is relative to the dataset root, e.g.
    "wavs/speakers/2BqVo8kVB2Skwgyb/de9ff364-6d1e-11e9....wav"
"""

import os
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Must run before the `transformers` import below -- forces every
# from_pretrained() call in this file to skip the network entirely.
# See hf_offline_utils.py for the one-time local model download steps.
from hf_offline_utils import enable_offline_mode
enable_offline_mode()

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import (
    WhisperModel,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    AutoModel,
    AutoTokenizer,
)


# =========================================================================== #
# 0. Config
# =========================================================================== #
@dataclass
class FSCConfig:
    root_dir: str = "fluent_speech_commands_dataset"
    train_csv: str = "data/train_data.csv"
    valid_csv: str = "data/valid_data.csv"
    test_csv: str = "data/test_data.csv"

    sample_rate: int = 16000
    max_audio_seconds: float = 5.0     # FSC utterances are short (~1-5s); pad/trim to this

    whisper_model_name: str = "openai/whisper-small"
    bert_model_name: str = "bert-base-uncased"
    whisper_layer: int = 6

    # ASR hypothesis cache (JSON: {relative_wav_path: hypothesis_text})
    asr_cache_path: str = "fsc_asr_cache.json"
    use_ground_truth_transcript: bool = False  # set True to bypass ASR decoding entirely

    batch_size: int = 8
    num_workers: int = 4
    lr: float = 3e-5
    epochs: int = 10
    slot_loss_weight: float = 1.0  # kept for API symmetry; used as object/location weight

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================================== #
# 1. Label vocabularies (action / object / location, built from train split)
# =========================================================================== #
class FSCLabelVocab:
    """
    Builds and holds action / object / location -> id maps, plus (optionally)
    a joined 'intent' vocab over the observed (action, object, location)
    triples, matching how some FSC papers report a single accuracy number.
    """

    def __init__(self, csv_paths: List[str]):
        dfs = [pd.read_csv(p) for p in csv_paths]
        full = pd.concat(dfs, ignore_index=True)

        self.action2id = self._build_vocab(full["action"])
        self.object2id = self._build_vocab(full["object"])
        self.location2id = self._build_vocab(full["location"])

        triples = list(
            zip(full["action"], full["object"], full["location"])
        )
        unique_triples = sorted(set(triples))
        self.intent2id = {t: i for i, t in enumerate(unique_triples)}

        self.id2action = {v: k for k, v in self.action2id.items()}
        self.id2object = {v: k for k, v in self.object2id.items()}
        self.id2location = {v: k for k, v in self.location2id.items()}
        self.id2intent = {v: k for k, v in self.intent2id.items()}

    @staticmethod
    def _build_vocab(series: pd.Series) -> Dict[str, int]:
        values = sorted(series.dropna().unique().tolist())
        return {v: i for i, v in enumerate(values)}

    @property
    def num_actions(self) -> int:
        return len(self.action2id)

    @property
    def num_objects(self) -> int:
        return len(self.object2id)

    @property
    def num_locations(self) -> int:
        return len(self.location2id)

    @property
    def num_intents(self) -> int:
        return len(self.intent2id)

    def encode(self, action: str, obj: str, location: str) -> Dict[str, int]:
        return {
            "action_id": self.action2id[action],
            "object_id": self.object2id[obj],
            "location_id": self.location2id[location],
            "intent_id": self.intent2id[(action, obj, location)],
        }

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(
                {
                    "action2id": self.action2id,
                    "object2id": self.object2id,
                    "location2id": self.location2id,
                    "intent2id": {f"{a}|{o}|{l}": i for (a, o, l), i in self.intent2id.items()},
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: str) -> "FSCLabelVocab":
        with open(path) as f:
            d = json.load(f)
        obj = cls.__new__(cls)
        obj.action2id = d["action2id"]
        obj.object2id = d["object2id"]
        obj.location2id = d["location2id"]
        obj.intent2id = {
            tuple(k.split("|")): v for k, v in d["intent2id"].items()
        }
        obj.id2action = {v: k for k, v in obj.action2id.items()}
        obj.id2object = {v: k for k, v in obj.object2id.items()}
        obj.id2location = {v: k for k, v in obj.location2id.items()}
        obj.id2intent = {v: k for k, v in obj.intent2id.items()}
        return obj


# =========================================================================== #
# 2. Offline ASR hypothesis generation (Whisper decode -> noisy transcript)
# =========================================================================== #
class ASRTranscriber:
    """
    Runs Whisper's own decoder over the dataset once and caches the resulting
    (possibly-wrong) transcript per wav path. This is what makes the setup
    genuinely "semi-cascaded": the text branch sees ASR hallucinations /
    substitutions, not the clean ground-truth label text.
    """

    def __init__(self, model_name: str, device: str, language: str = "en"):
        self.processor = WhisperProcessor.from_pretrained(model_name, local_files_only=True)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name, local_files_only=True).to(device)
        self.model.eval()
        # Some checkpoints ship a stale `forced_decoder_ids` in their saved
        # generation_config, which triggers a deprecation warning and can
        # conflict with passing language/task directly. Clear it so
        # generate(language=..., task=...) is unambiguous.
        self.model.generation_config.forced_decoder_ids = None
        self.device = device
        self.language = language

    @torch.no_grad()
    def transcribe_batch(self, waveforms: List[np.ndarray], sample_rate: int) -> List[str]:
        # return_attention_mask=True avoids the "attention mask not set" warning
        # and gives correct behavior when batches mix full-length/padded clips.
        inputs = self.processor(
            waveforms,
            sampling_rate=sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_features = inputs.input_features.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        # Newer transformers API: pass language/task directly to generate()
        # instead of the deprecated forced_decoder_ids path.
        generated_ids = self.model.generate(
            input_features,
            attention_mask=attention_mask,
            language=self.language,
            task="transcribe",
        )
        texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return [t.strip() for t in texts]

    def build_cache(
        self,
        wav_paths: List[str],
        root_dir: str,
        sample_rate: int,
        cache_path: str,
        batch_size: int = 16,
    ):
        """Decode every wav not already in the cache and write results to disk."""
        cache = {}
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                cache = json.load(f)

        todo = [p for p in wav_paths if p not in cache]
        print(f"[ASRTranscriber] {len(todo)} / {len(wav_paths)} utterances need decoding")

        for i in range(0, len(todo), batch_size):
            batch_paths = todo[i : i + batch_size]
            waveforms = [
                load_and_resample(os.path.join(root_dir, p), sample_rate)
                for p in batch_paths
            ]
            hyps = self.transcribe_batch(waveforms, sample_rate)
            for p, h in zip(batch_paths, hyps):
                cache[p] = h

            if (i // batch_size) % 20 == 0:
                with open(cache_path, "w") as f:
                    json.dump(cache, f)
                print(f"  ...{i + len(batch_paths)}/{len(todo)} decoded")

        with open(cache_path, "w") as f:
            json.dump(cache, f)
        print(f"[ASRTranscriber] cache written to {cache_path}")
        return cache


def load_and_resample(wav_path: str, target_sr: int) -> np.ndarray:
    waveform, sr = torchaudio.load(wav_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # mono
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform.squeeze(0).numpy()


# =========================================================================== #
# 3. PyTorch Dataset
# =========================================================================== #
class FluentSpeechCommandsDataset(Dataset):
    """
    One item = (raw waveform, ASR-hypothesis text, action/object/location/intent ids).

    Padding/feature-extraction is deferred to the collate_fn so that Whisper's
    log-mel extraction and BERT tokenization are batched, not per-sample.
    """

    def __init__(
        self,
        csv_path: str,
        root_dir: str,
        vocab: FSCLabelVocab,
        sample_rate: int = 16000,
        max_audio_seconds: float = 5.0,
        asr_cache: Optional[Dict[str, str]] = None,
        use_ground_truth_transcript: bool = False,
    ):
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.vocab = vocab
        self.sample_rate = sample_rate
        self.max_samples = int(max_audio_seconds * sample_rate)
        self.asr_cache = asr_cache or {}
        self.use_ground_truth_transcript = use_ground_truth_transcript

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        wav_rel_path = row["path"]
        wav_path = os.path.join(self.root_dir, wav_rel_path)

        waveform = load_and_resample(wav_path, self.sample_rate)

        # fixed-length crop/pad so batches build clean tensors before Whisper's
        # own feature extractor does its 30s-window padding internally
        if len(waveform) > self.max_samples:
            start = random.randint(0, len(waveform) - self.max_samples)
            waveform = waveform[start : start + self.max_samples]
        # (no manual zero-pad needed here -- WhisperFeatureExtractor pads/pads
        #  to 30s itself; we only *trim* long outliers above)

        if self.use_ground_truth_transcript:
            text = str(row["transcription"])
        else:
            text = self.asr_cache.get(wav_rel_path, str(row["transcription"]))
            # falls back to ground truth if that utterance wasn't decoded yet,
            # so training can proceed even with a partially built cache

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
# 4. Collate function: batches raw audio -> Whisper features, text -> BERT ids
# =========================================================================== #
class FSCCollator:
    def __init__(
        self,
        feature_extractor: WhisperFeatureExtractor,
        tokenizer,  # AutoTokenizer-compatible (BertTokenizerFast, DistilBertTokenizerFast, etc.)
        sample_rate: int = 16000,
        max_text_len: int = 32,
    ):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.max_text_len = max_text_len

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        waveforms = [b["waveform"] for b in batch]
        texts = [b["text"] for b in batch]

        # Whisper feature extractor pads/truncates every clip to its standard
        # 30s window and returns (B, 80, 3000) log-mel features.
        audio_inputs = self.feature_extractor(
            waveforms, sampling_rate=self.sample_rate, return_tensors="pt"
        )
        input_features = audio_inputs.input_features  # (B, 80, 3000)

        text_inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )

        return {
            "input_features": input_features,
            "input_ids": text_inputs["input_ids"],
            "text_attention_mask": text_inputs["attention_mask"],
            "action_labels": torch.tensor([b["action_id"] for b in batch], dtype=torch.long),
            "object_labels": torch.tensor([b["object_id"] for b in batch], dtype=torch.long),
            "location_labels": torch.tensor([b["location_id"] for b in batch], dtype=torch.long),
            "intent_labels": torch.tensor([b["intent_id"] for b in batch], dtype=torch.long),
            "paths": [b["path"] for b in batch],
        }


# =========================================================================== #
# 5. Model components (acoustic encoder / text encoder / cross-attn fusion)
#    -- same building blocks as the SLURP-style version, slot head removed
# =========================================================================== #
class WhisperAcousticEncoder(nn.Module):
    def __init__(self, model_name: str = "openai/whisper-small", layer: int = 6, freeze: bool = True):
        super().__init__()
        self.whisper = WhisperModel.from_pretrained(model_name, local_files_only=True)
        self.layer = layer
        self.hidden_size = self.whisper.config.d_model
        self._frozen = freeze
        if freeze:
            for p in self.whisper.parameters():
                p.requires_grad = False
            self.whisper.eval()

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        ctx = torch.no_grad() if self._frozen else torch.enable_grad()
        with ctx:
            enc_out = self.whisper.encoder(
                input_features, output_hidden_states=True, return_dict=True
            )
        return enc_out.hidden_states[self.layer]  # (B, T_frames, H)


class TextEncoder(nn.Module):
    """
    Generic text encoder -- works with BERT, DistilBERT, or any other
    AutoModel-compatible encoder pointed at by `model_name`. Using AutoModel
    instead of a hardcoded BertModel means the actual architecture is
    resolved from the local config.json's `model_type` field, so switching
    between e.g. "bert-base-uncased" and a local "distilbert-base-uncased"
    checkpoint requires no code change here -- only `bert_model_name` in
    FSCConfig needs to point at the right local directory.

    Note: DistilBertConfig doesn't expose `.hidden_size` (it uses `.dim`
    instead), so hidden size is looked up generically below rather than
    assuming a BERT-specific config attribute name.
    """

    def __init__(self, model_name: str = "bert-base-uncased", freeze: bool = False):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, local_files_only=True)
        self.hidden_size = self._get_hidden_size(self.bert.config)
        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

    @staticmethod
    def _get_hidden_size(config) -> int:
        # BERT/RoBERTa/etc. use `hidden_size`; DistilBERT uses `dim`.
        # Fall back through both rather than hardcoding one.
        for attr in ("hidden_size", "dim"):
            if hasattr(config, attr):
                return getattr(config, attr)
        raise AttributeError(
            f"Could not determine hidden size from config of type {type(config).__name__}. "
            f"Expected a `hidden_size` or `dim` attribute -- check the model's config.json."
        )

    def forward(self, input_ids, attention_mask):
        # DistilBERT (unlike BERT) does not accept/need token_type_ids, and
        # we never pass any here, so this call works unchanged for either.
        return self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


class CrossAttentionFusion(nn.Module):
    def __init__(self, text_dim, acoustic_dim, fusion_dim=768, num_heads=8, ffn_dim=2048, dropout=0.1):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.acoustic_proj = nn.Linear(acoustic_dim, fusion_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(fusion_dim)
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, fusion_dim)
        )
        self.norm2 = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_states, text_mask, acoustic_states, acoustic_mask=None):
        T = self.text_proj(text_states)
        A = self.acoustic_proj(acoustic_states)

        key_padding_mask = ~acoustic_mask.bool() if acoustic_mask is not None else None

        attn_out, attn_weights = self.cross_attn(
            query=T, key=A, value=A,
            key_padding_mask=key_padding_mask,
            need_weights=True, average_attn_weights=True,
        )
        fused = self.norm1(T + self.dropout(attn_out))
        fused = self.norm2(fused + self.dropout(self.ffn(fused)))
        return fused, attn_weights


# =========================================================================== #
# 6. FSC model head: action / object / location (+ optional joined intent)
# =========================================================================== #
class SemiCascadedSLU_FSC(nn.Module):
    """
    Same acoustic+text cross-attention backbone as the SLURP-style model,
    but with utterance-level action/object/location heads instead of a
    token-level slot head, since FSC has no word-aligned slot annotations.
    """

    def __init__(
        self,
        num_actions: int,
        num_objects: int,
        num_locations: int,
        whisper_model_name: str = "openai/whisper-small",
        bert_model_name: str = "bert-base-uncased",
        whisper_layer: int = 6,
        fusion_dim: int = 768,
        freeze_whisper: bool = True,
        freeze_bert: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.acoustic_encoder = WhisperAcousticEncoder(whisper_model_name, whisper_layer, freeze_whisper)
        self.text_encoder = TextEncoder(bert_model_name, freeze_bert)
        self.fusion = CrossAttentionFusion(
            text_dim=self.text_encoder.hidden_size,
            acoustic_dim=self.acoustic_encoder.hidden_size,
            fusion_dim=fusion_dim,
            dropout=dropout,
        )

        # gated residual mix, same rationale as the SLURP version: fall back
        # to "trust the transcript" when the acoustic branch isn't helpful
        self.gate = nn.Sequential(nn.Linear(fusion_dim * 2, fusion_dim), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)

        self.action_head = nn.Linear(fusion_dim, num_actions)
        self.object_head = nn.Linear(fusion_dim, num_objects)
        self.location_head = nn.Linear(fusion_dim, num_locations)

    def forward(
        self,
        input_features: torch.Tensor,
        input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        acoustic_attention_mask: torch.Tensor = None,
        action_labels: torch.Tensor = None,
        object_labels: torch.Tensor = None,
        location_labels: torch.Tensor = None,
        **kwargs,  # absorbs intent_labels / paths etc. from the collator batch
    ):
        acoustic_states = self.acoustic_encoder(input_features)
        text_states = self.text_encoder(input_ids, text_attention_mask)

        fused, attn_weights = self.fusion(
            text_states, text_attention_mask, acoustic_states, acoustic_attention_mask
        )

        text_proj = self.fusion.text_proj(text_states)
        g = self.gate(torch.cat([fused, text_proj], dim=-1))
        mixed = g * fused + (1 - g) * text_proj
        mixed = self.dropout(mixed)

        mask = text_attention_mask.unsqueeze(-1).float()
        pooled = (mixed * mask).sum(1) / mask.sum(1).clamp(min=1e-6)  # (B, fusion_dim)

        action_logits = self.action_head(pooled)
        object_logits = self.object_head(pooled)
        location_logits = self.location_head(pooled)

        output = {
            "action_logits": action_logits,
            "object_logits": object_logits,
            "location_logits": location_logits,
            "cross_attn_weights": attn_weights,
            "acoustic_gate": g,
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
# 7. End-to-end data prep: build vocab, ASR cache, datasets, loaders
# =========================================================================== #
def build_dataloaders(cfg: FSCConfig) -> Tuple[DataLoader, DataLoader, DataLoader, FSCLabelVocab]:
    train_csv = os.path.join(cfg.root_dir, cfg.train_csv)
    valid_csv = os.path.join(cfg.root_dir, cfg.valid_csv)
    test_csv = os.path.join(cfg.root_dir, cfg.test_csv)

    vocab = FSCLabelVocab([train_csv, valid_csv, test_csv])
    vocab.save("fsc_label_vocab.json")
    print(
        f"[vocab] actions={vocab.num_actions} objects={vocab.num_objects} "
        f"locations={vocab.num_locations} joined_intents={vocab.num_intents}"
    )

    asr_cache = {}
    if not cfg.use_ground_truth_transcript:
        transcriber = ASRTranscriber(cfg.whisper_model_name, cfg.device)
        all_paths = pd.concat(
            [pd.read_csv(p)["path"] for p in [train_csv, valid_csv, test_csv]]
        ).tolist()
        asr_cache = transcriber.build_cache(
            all_paths, cfg.root_dir, cfg.sample_rate, cfg.asr_cache_path
        )
        # free the ASR decoder before training; only the frozen encoder is
        # needed inside the model
        del transcriber
        torch.cuda.empty_cache() if cfg.device == "cuda" else None

    feature_extractor = WhisperFeatureExtractor.from_pretrained(cfg.whisper_model_name, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.bert_model_name, local_files_only=True)
    collator = FSCCollator(feature_extractor, tokenizer, sample_rate=cfg.sample_rate)

    common_kwargs = dict(
        root_dir=cfg.root_dir,
        vocab=vocab,
        sample_rate=cfg.sample_rate,
        max_audio_seconds=cfg.max_audio_seconds,
        asr_cache=asr_cache,
        use_ground_truth_transcript=cfg.use_ground_truth_transcript,
    )
    train_ds = FluentSpeechCommandsDataset(train_csv, **common_kwargs)
    valid_ds = FluentSpeechCommandsDataset(valid_csv, **common_kwargs)
    test_ds = FluentSpeechCommandsDataset(test_csv, **common_kwargs)

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
# 8. Training / evaluation loop
# =========================================================================== #
def _field_metrics(y_true: List[int], y_pred: List[int], prefix: str) -> Dict[str, float]:
    """
    Accuracy alone hides class imbalance (FSC has some rare object/location
    classes) -- a model can look fine on accuracy while just always
    predicting the majority class for the rare ones. Precision/recall/F1
    catch that: macro-avg weights every class equally (rare classes count
    fully), weighted-avg accounts for class frequency.
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


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    """
    Returns:
      - loss
      - exact_match_acc: FSC's headline metric. A sample counts as correct
        ONLY if action, object, AND location are all predicted correctly at
        once (2-out-of-3 right still counts as wrong). Reported alone, this
        can hide *which* field is dragging performance down.
      - per-field accuracy: diagnostic -- tells you whether errors are
        concentrated in action vs. object vs. location.
      - per-field macro/weighted precision, recall, F1: catch rare-class
        collapse that plain accuracy can hide (e.g. model nails the common
        "kitchen"/"none" locations but never gets rare ones right --
        accuracy still looks high, macro-F1 exposes it).
    """
    model.eval()
    total, correct_all = 0, 0
    total_loss = 0.0
    all_action_true, all_action_pred = [], []
    all_object_true, all_object_pred = [], []
    all_location_true, all_location_pred = [], []

    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(
            input_features=batch["input_features"],
            input_ids=batch["input_ids"],
            text_attention_mask=batch["text_attention_mask"],
            action_labels=batch["action_labels"],
            object_labels=batch["object_labels"],
            location_labels=batch["location_labels"],
        )
        total_loss += out["loss"].item() * batch["action_labels"].size(0)

        pred_action = out["action_logits"].argmax(-1)
        pred_object = out["object_logits"].argmax(-1)
        pred_location = out["location_logits"].argmax(-1)

        correct_all += (
            (pred_action == batch["action_labels"])
            & (pred_object == batch["object_labels"])
            & (pred_location == batch["location_labels"])
        ).sum().item()
        total += batch["action_labels"].size(0)

        all_action_true += batch["action_labels"].cpu().tolist()
        all_action_pred += pred_action.cpu().tolist()
        all_object_true += batch["object_labels"].cpu().tolist()
        all_object_pred += pred_object.cpu().tolist()
        all_location_true += batch["location_labels"].cpu().tolist()
        all_location_pred += pred_location.cpu().tolist()

    metrics = {
        "loss": total_loss / total,
        "exact_match_acc": correct_all / total,  # standard FSC headline metric
    }
    metrics.update(_field_metrics(all_action_true, all_action_pred, "action"))
    metrics.update(_field_metrics(all_object_true, all_object_pred, "object"))
    metrics.update(_field_metrics(all_location_true, all_location_pred, "location"))
    return metrics


def print_classification_reports(model: nn.Module, loader: DataLoader, device: str, vocab: "FSCLabelVocab"):
    """
    Per-class precision/recall/F1 breakdown for each field, using the
    original label strings instead of ids. Useful for spotting exactly
    which action/object/location classes the model struggles with.
    """
    from sklearn.metrics import classification_report

    model.eval()
    preds = {"action": [], "object": [], "location": []}
    trues = {"action": [], "object": [], "location": []}

    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(
                input_features=batch["input_features"],
                input_ids=batch["input_ids"],
                text_attention_mask=batch["text_attention_mask"],
            )
            preds["action"] += out["action_logits"].argmax(-1).cpu().tolist()
            preds["object"] += out["object_logits"].argmax(-1).cpu().tolist()
            preds["location"] += out["location_logits"].argmax(-1).cpu().tolist()
            trues["action"] += batch["action_labels"].cpu().tolist()
            trues["object"] += batch["object_labels"].cpu().tolist()
            trues["location"] += batch["location_labels"].cpu().tolist()

    id_maps = {"action": vocab.id2action, "object": vocab.id2object, "location": vocab.id2location}
    for field in ["action", "object", "location"]:
        id2label = id_maps[field]
        target_names = [id2label[i] for i in range(len(id2label))]
        print(f"\n=== {field} classification report ===")
        print(
            classification_report(
                trues[field], preds[field], target_names=target_names, zero_division=0
            )
        )


def train(cfg: FSCConfig):
    train_loader, valid_loader, test_loader, vocab = build_dataloaders(cfg)

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

        valid_metrics = evaluate(model, valid_loader, cfg.device)
        print(f"[epoch {epoch}] train_loss={running_loss / len(train_loader):.4f} valid={valid_metrics}")

        if valid_metrics["exact_match_acc"] > best_valid_acc:
            best_valid_acc = valid_metrics["exact_match_acc"]
            torch.save(model.state_dict(), "fsc_semi_cascaded_slu_best.pt")
            print(f"  -> new best ({best_valid_acc:.4f}), checkpoint saved")

    test_metrics = evaluate(model, test_loader, cfg.device)
    print(f"[final test] {test_metrics}")
    print_classification_reports(model, test_loader, cfg.device, vocab)
    return model, vocab, test_metrics


# =========================================================================== #
# 9. Entry point
# =========================================================================== #
if __name__ == "__main__":
    cfg = FSCConfig(
        root_dir="fluent_speech_commands_dataset",
        use_ground_truth_transcript=False,  # False = realistic ASR-hypothesis text input
        batch_size=8,
        epochs=10,
    )
    train(cfg)
