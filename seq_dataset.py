"""
seq_dataset.py
SLURP examples with the teacher-format semantics target and (clean, noisy) audio.
Noise handling is shared with kd_dataset.py (on-the-fly mixing at --snr_db, or a
pre-mixed noisy copy of SLURP).
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from kd_dataset import NoiseBank, load_mono, mix_at_snr, TARGET_SR
from teacher_nemo import nemo_entities, render_semantics


class SemanticsSlurpDataset(Dataset):
    def __init__(self, jsonl_path, audio_root, max_audio_seconds=15.0, noise_dir=None,
                 noisy_audio_root=None, snr_db=5.0, deterministic=False, seed=1234):
        if (noise_dir is None) == (noisy_audio_root is None):
            raise ValueError("Give exactly one of noise_dir or noisy_audio_root")
        self.audio_root = Path(audio_root)
        self.noisy_root = Path(noisy_audio_root) if noisy_audio_root else None
        self.noise = NoiseBank(noise_dir) if noise_dir else None
        self.snr_db, self.deterministic, self.seed = snr_db, deterministic, seed
        self.max_samples = int(max_audio_seconds * TARGET_SR)

        self.examples = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                sc, ac = rec.get("scenario"), rec.get("action")
                if sc is None or ac is None:
                    continue
                ents = nemo_entities(rec.get("sentence_annotation", "") or "")
                for r in rec.get("recordings", []):
                    if r.get("file"):
                        self.examples.append({"audio_file": r["file"], "scenario": sc,
                                              "action": ac, "entities": ents})

    def __len__(self):
        return len(self.examples)

    def attach_targets(self, fmt):
        """Render + tokenise every gold target ONCE in the main process, so DataLoader
        workers never need the (unpicklable) NeMo tokenizer. Required on Python 3.14+,
        where workers start via forkserver and must pickle everything they receive."""
        cache = {}
        for ex in self.examples:
            key = (ex["scenario"], ex["action"], tuple(ex["entities"]))
            if key not in cache:
                text = fmt.render(ex["scenario"], ex["action"], ex["entities"])
                cache[key] = (text, fmt.encode_target(text))
            ex["text"], ex["target"] = cache[key]

    @staticmethod
    def _find(root, fname):
        for sub in ("slurp_real", "slurp_synth", ""):
            p = root / sub / fname
            if p.exists():
                return p
        raise FileNotFoundError(f"{fname} not found under {root}")

    def __getitem__(self, idx):
        ex = self.examples[idx]
        clean = load_mono(self._find(self.audio_root, ex["audio_file"]))[: self.max_samples]
        if self.noisy_root is not None:
            noisy = load_mono(self._find(self.noisy_root, ex["audio_file"]))[: self.max_samples]
            n = min(clean.size, noisy.size)
            clean, noisy = clean[:n], noisy[:n]
        else:
            rng = np.random.default_rng(self.seed + idx) if self.deterministic else np.random.default_rng()
            noisy = mix_at_snr(clean, self.noise.sample(clean.size, rng), self.snr_db)
        return {"noisy": torch.from_numpy(np.ascontiguousarray(noisy)),
                "clean": torch.from_numpy(np.ascontiguousarray(clean)), **ex}


class TargetFormatter:
    """Tokenizer + separator only (no GPU model), safe to use inside DataLoader workers."""
    def __init__(self, teacher):
        self.tok, self.sep, self.bos, self.eos = teacher.tok, teacher.sep, teacher.bos, teacher.tok.eos_id

    def render(self, scenario, action, entities):
        return render_semantics(scenario, action, entities, self.sep)

    def encode_target(self, text):
        return [self.bos] + list(self.tok.text_to_ids(text)) + [self.eos]


def collate_semantics(batch):
    """Top-level (picklable) collate. Builds decoder input/labels the way the teacher
    was trained:  target = [BOS] + ids + [EOS];  dec_inp = target[:-1];  labels = target[1:]"""
    if "target" not in batch[0]:
        raise RuntimeError("Call dataset.attach_targets(fmt) before creating the DataLoader")
    B = len(batch)
    n = max(b["noisy"].shape[0] for b in batch)
    noisy = torch.zeros(B, n)
    clean = torch.zeros(B, n)
    att = torch.zeros(B, n, dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["noisy"].shape[0]
        noisy[i, :L], clean[i, :L], att[i, :L] = b["noisy"], b["clean"], 1

    Lt = max(len(b["target"]) for b in batch) - 1
    dec_inp = torch.zeros(B, Lt, dtype=torch.long)
    labels = torch.zeros(B, Lt, dtype=torch.long)
    dec_mask = torch.zeros(B, Lt)
    for i, b in enumerate(batch):
        t = b["target"]
        k = len(t) - 1
        dec_inp[i, :k] = torch.tensor(t[:-1])
        labels[i, :k] = torch.tensor(t[1:])
        dec_mask[i, :k] = 1.0
    return {"noisy": noisy, "clean": clean, "attention_mask": att,
            "dec_inp": dec_inp, "labels": labels, "dec_mask": dec_mask,
            "gold_texts": [b["text"] for b in batch], "audio_files": [b["audio_file"] for b in batch],
            "gold_intents": [f"{b['scenario']}_{b['action']}" for b in batch],
            "gold_entities": [b["entities"] for b in batch]}
