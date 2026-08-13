"""
Semi-Cascaded SLU on SLURP
===========================

Adapts the frozen-Whisper-acoustic + ASR-transcript-text cross-attention
architecture to the SLURP dataset (Bastianelli et al., EMNLP 2020).

Unlike Fluent Speech Commands, SLURP has:
  * A real intent taxonomy (`scenario_action`, ~60 classes) -- utterance-level,
    same as before.
  * WORD-LEVEL slot filling via inline annotation, e.g.:
        sentence:            "is my reminder alarm set for dance class"
        sentence_annotation: "is my reminder alarm set for [event_name : dance class]"
    i.e. this is back to the ORIGINAL intent+slot architecture (masked-mean
    pooled intent head + per-token slot head), not FSC's flat 3-way
    action/object/location classification.
  * MULTIPLE recordings per sentence (different mic conditions: "-headset"
    suffix = close-talk, no suffix = distant/other mic), each with its own
    `wer`/`status` field indicating how well that specific recording's
    audio actually matches the canonical sentence text.

Directory layout expected (standard SLURP release):

    <root_dir>/
        dataset/slurp/
            train.jsonl
            devel.jsonl
            test.jsonl
            train_synthetic.jsonl      (optional, synthetic TTS augmentation)
        slurp_real/
            audio--<id>-headset.wav
            audio--<id>.wav
            ...
        slurp_synth/
            ... (only referenced by train_synthetic.jsonl)

Each jsonl line has (fields we use):
    slurp_id, sentence, sentence_annotation, intent, scenario, action,
    recordings: [{"file": ..., "wer": 0.0, "status": "correct"}, ...]

=====================================================================
IMPORTANT DESIGN NOTE: slot-tag alignment under ASR errors
=====================================================================
`sentence_annotation` gives BIO tags aligned to the CLEAN sentence's words.
But this model's text branch is fed the (possibly wrong) ASR HYPOTHESIS,
per the same semi-cascaded design used for FSC -- so the ground-truth tags
can't be used as-is; they're defined over a different word sequence than
the one the model actually sees.

This script resolves that with word-level Levenshtein alignment: the
clean-sentence BIO tags are projected onto the ASR hypothesis words
(`_align_and_project_tags`). Matched/substituted hypothesis words inherit
the aligned ground-truth word's tag; inserted hypothesis words (no
ground-truth counterpart) get "O". This is a standard technique in
ASR-robust slot-filling work, not something SLURP provides natively --
document it as such if you use it in a paper.
"""

import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Must run before any `transformers`/`huggingface_hub` import.
from hf_offline_utils import enable_offline_mode
enable_offline_mode()

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
class SLURPConfig:
    root_dir: str = "slurp_dataset"
    train_jsonl: str = "dataset/slurp/train.jsonl"
    valid_jsonl: str = "dataset/slurp/devel.jsonl"
    test_jsonl: str = "dataset/slurp/test.jsonl"
    train_synthetic_jsonl: str = "dataset/slurp/train_synthetic.jsonl"
    include_synthetic: bool = False          # augment train split with TTS-synthesized audio

    audio_real_subdir: str = "slurp_real"
    audio_synth_subdir: str = "slurp_synth"

    # keep only recordings SLURP marked as a verified-correct match to the
    # canonical sentence text (recommended -- avoids training on
    # audio/label mismatches from noisy verification)
    recording_filter: str = "correct_only"   # "correct_only" | "all"
    # optionally restrict to one mic condition, e.g. "headset" (close-talk
    # only) or None to keep every recording type
    recording_type_filter: Optional[str] = None

    sample_rate: int = 16000
    max_audio_seconds: float = 11.0          # SLURP utterances run longer than FSC's

    whisper_model_name: str = "openai/whisper-small"
    bert_model_name: str = "bert-base-uncased"   # or a local DistilBERT path, etc.
    whisper_layer: int = 6

    asr_cache_path: str = "slurp_asr_cache.json"
    use_ground_truth_transcript: bool = False

    max_text_len: int = 64
    batch_size: int = 8
    num_workers: int = 4
    lr: float = 3e-5
    epochs: int = 10
    slot_loss_weight: float = 1.0

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================================== #
# 1. jsonl reading + sentence_annotation parsing
# =========================================================================== #
def _read_jsonl(path: str) -> List[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


_ENTITY_PATTERN = re.compile(r"\[\s*([a-zA-Z_]+)\s*:\s*([^\]]+?)\s*\]")


def parse_sentence_annotation(sentence_annotation: str) -> Tuple[List[str], List[str]]:
    """
    "is my reminder alarm set for [event_name : dance class]"
      -> words = ["is","my","reminder","alarm","set","for","dance","class"]
         tags  = ["O","O","O","O","O","O","B-event_name","I-event_name"]
    """
    words, tags = [], []
    pos = 0
    for m in _ENTITY_PATTERN.finditer(sentence_annotation):
        before = sentence_annotation[pos:m.start()]
        for w in before.split():
            words.append(w)
            tags.append("O")

        ent_type = m.group(1).strip()
        filler_words = m.group(2).strip().split()
        for i, w in enumerate(filler_words):
            tags.append(f"B-{ent_type}" if i == 0 else f"I-{ent_type}")
            words.append(w)

        pos = m.end()

    after = sentence_annotation[pos:]
    for w in after.split():
        words.append(w)
        tags.append("O")

    return words, tags


# =========================================================================== #
# 2. Word-level Levenshtein alignment: project clean-sentence BIO tags onto
#    an ASR hypothesis word sequence
# =========================================================================== #
def _align_and_project_tags(gt_words: List[str], gt_tags: List[str], hyp_words: List[str]) -> List[str]:
    n, m = len(gt_words), len(hyp_words)
    if m == 0:
        return []

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if gt_words[i - 1].lower() == hyp_words[j - 1].lower() else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    i, j = n, m
    hyp_tags = ["O"] * m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (
            0 if gt_words[i - 1].lower() == hyp_words[j - 1].lower() else 1
        ):
            hyp_tags[j - 1] = gt_tags[i - 1]   # match or substitution: inherit gt tag
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            i -= 1                              # deletion: gt word missing from hyp, nothing to assign
        else:
            j -= 1                               # insertion: hyp word has no gt counterpart, stays 'O'
    return hyp_tags


# =========================================================================== #
# 3. Label vocab: intents + BIO slot label set
# =========================================================================== #
class SLURPLabelVocab:
    def __init__(self, jsonl_paths: List[str]):
        intents = set()
        slot_types = set()
        for p in jsonl_paths:
            for entry in _read_jsonl(p):
                intents.add(entry["intent"])
                _, tags = parse_sentence_annotation(entry.get("sentence_annotation", entry["sentence"]))
                for t in tags:
                    if t != "O":
                        slot_types.add(t[2:])

        self.intent2id = {v: i for i, v in enumerate(sorted(intents))}
        self.id2intent = {i: v for v, i in self.intent2id.items()}

        slot_labels = ["O"] + sorted(f"{prefix}-{t}" for t in sorted(slot_types) for prefix in ("B", "I"))
        self.slot2id = {v: i for i, v in enumerate(slot_labels)}
        self.id2slot = {i: v for v, i in self.slot2id.items()}

    @property
    def num_intents(self) -> int:
        return len(self.intent2id)

    @property
    def num_slot_labels(self) -> int:
        return len(self.slot2id)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"intent2id": self.intent2id, "slot2id": self.slot2id}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SLURPLabelVocab":
        with open(path) as f:
            d = json.load(f)
        obj = cls.__new__(cls)
        obj.intent2id = d["intent2id"]
        obj.slot2id = d["slot2id"]
        obj.id2intent = {v: k for k, v in obj.intent2id.items()}
        obj.id2slot = {v: k for k, v in obj.slot2id.items()}
        return obj


# =========================================================================== #
# 4. Audio I/O
# =========================================================================== #
def load_and_resample(wav_path: str, target_sr: int):
    import numpy as np
    waveform, sr = torchaudio.load(wav_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform.squeeze(0).numpy()


# =========================================================================== #
# 5. ASR transcript caching (Whisper decode -> noisy hypothesis text)
# =========================================================================== #
class ASRTranscriber:
    def __init__(self, model_name: str, device: str, language: str = "en"):
        self.processor = WhisperProcessor.from_pretrained(model_name, local_files_only=True)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name, local_files_only=True).to(device)
        self.model.eval()
        self.model.generation_config.forced_decoder_ids = None
        self.device = device
        self.language = language

    @torch.no_grad()
    def transcribe_batch(self, waveforms: List, sample_rate: int) -> List[str]:
        inputs = self.processor(
            waveforms, sampling_rate=sample_rate, return_tensors="pt", return_attention_mask=True
        )
        input_features = inputs.input_features.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        generated_ids = self.model.generate(
            input_features, attention_mask=attention_mask, language=self.language, task="transcribe"
        )
        return [t.strip() for t in self.processor.batch_decode(generated_ids, skip_special_tokens=True)]

    def build_cache(self, wav_paths: List[str], root_dir: str, sample_rate: int, cache_path: str,
                     batch_size: int = 16) -> Dict[str, str]:
        cache = {}
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                cache = json.load(f)

        todo = [p for p in wav_paths if p not in cache]
        print(f"[ASRTranscriber] {len(todo)} / {len(wav_paths)} utterances need decoding")

        for i in range(0, len(todo), batch_size):
            batch_paths = todo[i:i + batch_size]
            waveforms = [load_and_resample(os.path.join(root_dir, p), sample_rate) for p in batch_paths]
            hyps = self.transcribe_batch(waveforms, sample_rate)
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
# 6. Dataset: expands each jsonl entry into one example PER recording
# =========================================================================== #
class SLURPDataset(Dataset):
    def __init__(
        self,
        jsonl_paths: List[Tuple[str, str]],   # [(jsonl_path, audio_subdir), ...]
        root_dir: str,
        vocab: SLURPLabelVocab,
        sample_rate: int = 16000,
        max_audio_seconds: float = 11.0,
        asr_cache: Optional[Dict[str, str]] = None,
        use_ground_truth_transcript: bool = False,
        recording_filter: str = "correct_only",
        recording_type_filter: Optional[str] = None,
    ):
        self.root_dir = root_dir
        self.vocab = vocab
        self.sample_rate = sample_rate
        self.max_samples = int(max_audio_seconds * sample_rate)
        self.asr_cache = asr_cache or {}
        self.use_ground_truth_transcript = use_ground_truth_transcript

        self.examples = []
        for jsonl_path, audio_subdir in jsonl_paths:
            for entry in _read_jsonl(jsonl_path):
                if entry["intent"] not in vocab.intent2id:
                    continue  # unseen intent (can happen with synthetic augmentation) -- skip
                words, tags = parse_sentence_annotation(entry.get("sentence_annotation", entry["sentence"]))

                for rec in entry.get("recordings", []):
                    if recording_filter == "correct_only" and rec.get("status") != "correct":
                        continue
                    fname = rec["file"]
                    if recording_type_filter == "headset" and "-headset" not in fname:
                        continue
                    if recording_type_filter == "non_headset" and "-headset" in fname:
                        continue

                    self.examples.append({
                        "wav_rel_path": os.path.join(audio_subdir, fname),
                        "words": words,
                        "tags": tags,
                        "sentence": entry["sentence"],
                        "intent_id": vocab.intent2id[entry["intent"]],
                    })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        ex = self.examples[idx]
        waveform = load_and_resample(os.path.join(self.root_dir, ex["wav_rel_path"]), self.sample_rate)

        if len(waveform) > self.max_samples:
            start = random.randint(0, len(waveform) - self.max_samples)
            waveform = waveform[start:start + self.max_samples]

        asr_text = ex["sentence"] if self.use_ground_truth_transcript else \
            self.asr_cache.get(ex["wav_rel_path"], ex["sentence"])

        return {
            "wav_rel_path": ex["wav_rel_path"],
            "waveform": waveform,
            "words": ex["words"],       # ground-truth words (clean sentence)
            "tags": ex["tags"],         # ground-truth BIO tags aligned to `words`
            "asr_text": asr_text,       # what the text branch will actually see
            "intent_id": ex["intent_id"],
        }


# =========================================================================== #
# 7. Collator: builds Whisper features + subword-tokenizes text + projects
#    slot tags onto whichever word sequence (gt or ASR) is actually used
# =========================================================================== #
class SLURPCollator:
    def __init__(self, feature_extractor, tokenizer, slot2id: Dict[str, int],
                 sample_rate: int = 16000, max_text_len: int = 64,
                 use_ground_truth_transcript: bool = False):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.slot2id = slot2id
        self.sample_rate = sample_rate
        self.max_text_len = max_text_len
        self.use_ground_truth_transcript = use_ground_truth_transcript

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        waveforms = [b["waveform"] for b in batch]
        audio_inputs = self.feature_extractor(waveforms, sampling_rate=self.sample_rate, return_tensors="pt")
        input_features = audio_inputs.input_features

        all_words, all_tags = [], []
        for b in batch:
            if self.use_ground_truth_transcript:
                all_words.append(b["words"])
                all_tags.append(b["tags"])
            else:
                hyp_words = b["asr_text"].split() or ["<empty>"]
                hyp_tags = _align_and_project_tags(b["words"], b["tags"], hyp_words)
                all_words.append(hyp_words)
                all_tags.append(hyp_tags)

        text_inputs = self.tokenizer(
            all_words, is_split_into_words=True, padding=True, truncation=True,
            max_length=self.max_text_len, return_tensors="pt",
        )

        slot_labels = torch.full(text_inputs["input_ids"].shape, -100, dtype=torch.long)
        for i, tags in enumerate(all_tags):
            word_ids = text_inputs.word_ids(batch_index=i)
            prev_word_idx = None
            for pos, w_idx in enumerate(word_ids):
                if w_idx is None:
                    continue
                if w_idx != prev_word_idx:
                    # only the FIRST subword of each word carries the real
                    # label; subsequent subwords of the same word stay -100
                    # (ignored in the loss) -- standard token-classification
                    # subword alignment convention.
                    tag = tags[w_idx] if w_idx < len(tags) else "O"
                    slot_labels[i, pos] = self.slot2id.get(tag, self.slot2id["O"])
                prev_word_idx = w_idx

        return {
            "input_features": input_features,
            "input_ids": text_inputs["input_ids"],
            "text_attention_mask": text_inputs["attention_mask"],
            "slot_labels": slot_labels,
            "intent_labels": torch.tensor([b["intent_id"] for b in batch], dtype=torch.long),
            "paths": [b["wav_rel_path"] for b in batch],
        }


# =========================================================================== #
# 8. Model components (shared with the FSC version's design)
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
            enc_out = self.whisper.encoder(input_features, output_hidden_states=True, return_dict=True)
        return enc_out.hidden_states[self.layer]


class TextEncoder(nn.Module):
    """AutoModel-based so BERT, DistilBERT, etc. all work without code changes."""

    def __init__(self, model_name: str = "bert-base-uncased", freeze: bool = False):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, local_files_only=True)
        self.hidden_size = self._get_hidden_size(self.bert.config)
        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

    @staticmethod
    def _get_hidden_size(config) -> int:
        for attr in ("hidden_size", "dim"):
            if hasattr(config, attr):
                return getattr(config, attr)
        raise AttributeError(f"Could not determine hidden size from config {type(config).__name__}")

    def forward(self, input_ids, attention_mask):
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
            query=T, key=A, value=A, key_padding_mask=key_padding_mask,
            need_weights=True, average_attn_weights=True,
        )
        fused = self.norm1(T + self.dropout(attn_out))
        fused = self.norm2(fused + self.dropout(self.ffn(fused)))
        return fused, attn_weights


# =========================================================================== #
# 9. Full model: intent head (pooled) + slot head (per-token) -- the
#    original SLURP-style design, now with the gated residual mix
# =========================================================================== #
class SemiCascadedSLU_SLURP(nn.Module):
    def __init__(
        self,
        num_intents: int,
        num_slot_labels: int,
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
        self.gate = nn.Sequential(nn.Linear(fusion_dim * 2, fusion_dim), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)
        self.intent_head = nn.Linear(fusion_dim, num_intents)
        self.slot_head = nn.Linear(fusion_dim, num_slot_labels)

    def forward(
        self,
        input_features: torch.Tensor,
        input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        acoustic_attention_mask: torch.Tensor = None,
        intent_labels: torch.Tensor = None,
        slot_labels: torch.Tensor = None,
        slot_loss_weight: float = 1.0,
        **kwargs,
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
        pooled = (mixed * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        intent_logits = self.intent_head(pooled)
        slot_logits = self.slot_head(mixed)

        output = {
            "intent_logits": intent_logits,
            "slot_logits": slot_logits,
            "cross_attn_weights": attn_weights,
            "acoustic_gate": g,
        }

        if intent_labels is not None and slot_labels is not None:
            intent_loss = F.cross_entropy(intent_logits, intent_labels)
            slot_loss = F.cross_entropy(
                slot_logits.view(-1, slot_logits.size(-1)), slot_labels.view(-1), ignore_index=-100
            )
            output["loss"] = intent_loss + slot_loss_weight * slot_loss
            output["intent_loss"] = intent_loss
            output["slot_loss"] = slot_loss

        return output


# =========================================================================== #
# 10. Metrics: intent (accuracy + macro/weighted P/R/F1) and slot filling
#     (span-level exact-match P/R/F1, entity type + boundaries must match --
#     NOTE: this is a standard proxy metric, not SLURP's official
#     semantically-weighted SLU-F1 scorer; use slurp/scripts/evaluation for
#     paper-comparable numbers)
# =========================================================================== #
def _intent_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "intent_acc": acc,
        "intent_precision_macro": p_macro, "intent_recall_macro": r_macro, "intent_f1_macro": f1_macro,
        "intent_precision_weighted": p_w, "intent_recall_weighted": r_w, "intent_f1_weighted": f1_w,
    }


def _bio_to_spans(tags: List[str]) -> List[Tuple[str, int, int]]:
    spans = []
    start, ent_type = None, None
    for i, tag in enumerate(tags + ["O"]):
        if tag.startswith("B-"):
            if start is not None:
                spans.append((ent_type, start, i))
            start, ent_type = i, tag[2:]
        elif tag.startswith("I-") and ent_type == tag[2:]:
            continue
        else:
            if start is not None:
                spans.append((ent_type, start, i))
                start, ent_type = None, None
            if tag.startswith("I-"):
                start, ent_type = i, tag[2:]
    return spans


def _slot_span_metrics(all_true_tags: List[List[str]], all_pred_tags: List[List[str]]) -> Dict[str, float]:
    tp = fp = fn = 0
    for true_tags, pred_tags in zip(all_true_tags, all_pred_tags):
        true_spans = set(_bio_to_spans(true_tags))
        pred_spans = set(_bio_to_spans(pred_tags))
        tp += len(true_spans & pred_spans)
        fp += len(pred_spans - true_spans)
        fn += len(true_spans - pred_spans)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"slot_precision": precision, "slot_recall": recall, "slot_f1": f1}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str, vocab: SLURPLabelVocab) -> Dict[str, float]:
    model.eval()
    total, correct_intent = 0, 0
    total_loss = 0.0
    all_intent_true, all_intent_pred = [], []
    all_slot_true_tags, all_slot_pred_tags = [], []

    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(
            input_features=batch["input_features"],
            input_ids=batch["input_ids"],
            text_attention_mask=batch["text_attention_mask"],
            intent_labels=batch["intent_labels"],
            slot_labels=batch["slot_labels"],
        )
        bsz = batch["intent_labels"].size(0)
        total_loss += out["loss"].item() * bsz
        total += bsz

        pred_intent = out["intent_logits"].argmax(-1)
        correct_intent += (pred_intent == batch["intent_labels"]).sum().item()
        all_intent_true += batch["intent_labels"].cpu().tolist()
        all_intent_pred += pred_intent.cpu().tolist()

        pred_slot_ids = out["slot_logits"].argmax(-1)
        slot_labels = batch["slot_labels"]
        mask = slot_labels != -100
        for i in range(bsz):
            true_ids = slot_labels[i][mask[i]].cpu().tolist()
            pred_ids = pred_slot_ids[i][mask[i]].cpu().tolist()
            all_slot_true_tags.append([vocab.id2slot[t] for t in true_ids])
            all_slot_pred_tags.append([vocab.id2slot[p] for p in pred_ids])

    metrics = {"loss": total_loss / total, "intent_acc": correct_intent / total}
    metrics.update(_intent_metrics(all_intent_true, all_intent_pred))
    metrics.update(_slot_span_metrics(all_slot_true_tags, all_slot_pred_tags))
    return metrics


# =========================================================================== #
# 11. Data pipeline
# =========================================================================== #
def build_dataloaders(cfg: SLURPConfig) -> Tuple[DataLoader, DataLoader, DataLoader, SLURPLabelVocab]:
    train_jsonl = os.path.join(cfg.root_dir, cfg.train_jsonl)
    valid_jsonl = os.path.join(cfg.root_dir, cfg.valid_jsonl)
    test_jsonl = os.path.join(cfg.root_dir, cfg.test_jsonl)
    synth_jsonl = os.path.join(cfg.root_dir, cfg.train_synthetic_jsonl)

    vocab_source_paths = [train_jsonl, valid_jsonl, test_jsonl]
    if cfg.include_synthetic and os.path.exists(synth_jsonl):
        vocab_source_paths.append(synth_jsonl)
    vocab = SLURPLabelVocab(vocab_source_paths)
    vocab.save("slurp_label_vocab.json")
    print(f"[vocab] intents={vocab.num_intents} slot_labels={vocab.num_slot_labels}")

    asr_cache = {}
    if not cfg.use_ground_truth_transcript:
        transcriber = ASRTranscriber(cfg.whisper_model_name, cfg.device)
        all_paths = []
        for jp, subdir in [(train_jsonl, cfg.audio_real_subdir),
                            (valid_jsonl, cfg.audio_real_subdir),
                            (test_jsonl, cfg.audio_real_subdir)]:
            for entry in _read_jsonl(jp):
                for rec in entry.get("recordings", []):
                    if cfg.recording_filter == "correct_only" and rec.get("status") != "correct":
                        continue
                    all_paths.append(os.path.join(subdir, rec["file"]))
        if cfg.include_synthetic and os.path.exists(synth_jsonl):
            for entry in _read_jsonl(synth_jsonl):
                for rec in entry.get("recordings", []):
                    all_paths.append(os.path.join(cfg.audio_synth_subdir, rec["file"]))

        asr_cache = transcriber.build_cache(all_paths, cfg.root_dir, cfg.sample_rate, cfg.asr_cache_path)
        del transcriber
        if cfg.device == "cuda":
            torch.cuda.empty_cache()

    feature_extractor = WhisperFeatureExtractor.from_pretrained(cfg.whisper_model_name, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.bert_model_name, local_files_only=True)
    collator = SLURPCollator(
        feature_extractor, tokenizer, vocab.slot2id,
        sample_rate=cfg.sample_rate, max_text_len=cfg.max_text_len,
        use_ground_truth_transcript=cfg.use_ground_truth_transcript,
    )

    common_kwargs = dict(
        root_dir=cfg.root_dir, vocab=vocab, sample_rate=cfg.sample_rate,
        max_audio_seconds=cfg.max_audio_seconds, asr_cache=asr_cache,
        use_ground_truth_transcript=cfg.use_ground_truth_transcript,
        recording_filter=cfg.recording_filter, recording_type_filter=cfg.recording_type_filter,
    )

    train_jsonl_paths = [(train_jsonl, cfg.audio_real_subdir)]
    if cfg.include_synthetic and os.path.exists(synth_jsonl):
        train_jsonl_paths.append((synth_jsonl, cfg.audio_synth_subdir))

    train_ds = SLURPDataset(train_jsonl_paths, **common_kwargs)
    valid_ds = SLURPDataset([(valid_jsonl, cfg.audio_real_subdir)], **common_kwargs)
    test_ds = SLURPDataset([(test_jsonl, cfg.audio_real_subdir)], **common_kwargs)
    print(f"[data] train={len(train_ds)} valid={len(valid_ds)} test={len(test_ds)} examples (per-recording)")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                               num_workers=cfg.num_workers, collate_fn=collator, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False,
                               num_workers=cfg.num_workers, collate_fn=collator)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, collate_fn=collator)
    return train_loader, valid_loader, test_loader, vocab


# =========================================================================== #
# 12. Training loop
# =========================================================================== #
def train(cfg: SLURPConfig):
    train_loader, valid_loader, test_loader, vocab = build_dataloaders(cfg)

    model = SemiCascadedSLU_SLURP(
        num_intents=vocab.num_intents,
        num_slot_labels=vocab.num_slot_labels,
        whisper_model_name=cfg.whisper_model_name,
        bert_model_name=cfg.bert_model_name,
        whisper_layer=cfg.whisper_layer,
        freeze_whisper=True,
        freeze_bert=False,
    ).to(cfg.device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr)

    best_valid_f1 = -1.0
    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            batch = {k: (v.to(cfg.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(
                input_features=batch["input_features"],
                input_ids=batch["input_ids"],
                text_attention_mask=batch["text_attention_mask"],
                intent_labels=batch["intent_labels"],
                slot_labels=batch["slot_labels"],
                slot_loss_weight=cfg.slot_loss_weight,
            )
            loss = out["loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f} "
                      f"(intent={out['intent_loss'].item():.4f} slot={out['slot_loss'].item():.4f})")

        valid_metrics = evaluate(model, valid_loader, cfg.device, vocab)
        print(f"[epoch {epoch}] train_loss={running_loss / len(train_loader):.4f} valid={valid_metrics}")

        combined = 0.5 * valid_metrics["intent_acc"] + 0.5 * valid_metrics["slot_f1"]
        if combined > best_valid_f1:
            best_valid_f1 = combined
            torch.save(model.state_dict(), "slurp_semi_cascaded_slu_best.pt")
            print(f"  -> new best (intent_acc+slot_f1 avg={best_valid_f1:.4f}), checkpoint saved")

    test_metrics = evaluate(model, test_loader, cfg.device, vocab)
    print(f"[final test] {test_metrics}")
    return model, vocab, test_metrics


if __name__ == "__main__":
    cfg = SLURPConfig(
        root_dir="slurp_dataset",
        use_ground_truth_transcript=False,
        recording_filter="correct_only",
        batch_size=8,
        epochs=10,
    )
    train(cfg)
