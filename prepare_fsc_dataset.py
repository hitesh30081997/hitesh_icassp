"""
Prepare Fluent Speech Commands (FSC) dataset for joint intent classification
and slot filling with DistilBERT.

Expects the standard FSC directory structure:
    fluent_speech_commands_dataset/
        data/
            train_data.csv
            valid_data.csv
            test_data.csv
        wavs/...

Each CSV row looks like:
    ,path,speakerId,transcription,action,object,location
    0,wavs/speakers/.../0.wav,2BqVo8kVB2Skwgyb,Turn on the lights,activate,lights,none

Steps performed:
  1. Build INTENT label = f"{action}_{object}_{location}"
  2. Build SLOT (BIO) labels at word level by locating the `object` and
     `location` values as substrings inside the transcription
  3. Tokenize with a DistilBERT tokenizer and align word-level BIO labels to
     subword tokens
  4. Save everything as a HuggingFace `datasets.DatasetDict`, ready to feed a
     joint intent+slot DistilBERT model

Install deps first:
    pip install transformers datasets pandas --break-system-packages
"""

import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict

MODEL_NAME = "distilbert-base-uncased"
DATA_DIR = Path("fluent_speech_commands_dataset/data")  # <-- adjust to your path
OUT_DIR = Path("fsc_processed")
MAX_LENGTH = 32
NONE_VALUE = "none"

# Action -> phrases that realize it in the transcript. Used to locate and
# BIO-tag the "action" span, in addition to the object/location slots.
ACTION_SYNONYMS = {
    "activate": ["turn on", "switch on", "start", "enable", "on", "play", "resume", "put on"],
    "deactivate": ["turn off", "switch off", "off", "stop", "kill", "pause"],
    "increase": ["turn up", "crank up", "up", "raise", "higher", "louder", "max", "need volume", "more", "too quiet"],
    "decrease": ["turn down", "lower", "down", "quieter", "soften", "reduce", "too loud", "less", "mute", "softer"],
    "bring": ["get", "fetch", "hand me"],
    "change language": ["switch the language", "change system language", "language settings",
                         "set the language", "set language", "phone's language", "main language"],
}


def find_phrase_span(transcription_words, phrase):
    """Find a sublist match of `phrase`'s words inside transcription_words.
    Returns (start_idx, end_idx_inclusive) or None if not found."""
    if phrase is None or str(phrase).lower() == NONE_VALUE:
        return None
    phrase_words = str(phrase).lower().split()
    n, m = len(transcription_words), len(phrase_words)
    if m == 0 or m > n:
        return None
    for i in range(n - m + 1):
        window = [w.lower().strip(".,!?") for w in transcription_words[i:i + m]]
        if window == phrase_words:
            return i, i + m - 1
    return None


def find_action_span(transcription_words, action_label):
    """Search ACTION_SYNONYMS[action_label] for a phrase that appears in the
    transcript. Longest phrases are tried first so e.g. 'turn on' is matched
    in full rather than only its substring 'on'."""
    candidates = ACTION_SYNONYMS.get(action_label, [])
    for phrase in sorted(candidates, key=lambda p: -len(p.split())):
        span = find_phrase_span(transcription_words, phrase)
        if span is not None:
            return span
    return None


def build_bio_tags(transcription, action_val, object_val, location_val):
    words = transcription.split()
    tags = ["O"] * len(words)

    # 1. action span, tagged first (from synonym phrase list)
    action_span = find_action_span(words, action_val)
    if action_span is not None:
        start, end = action_span
        tags[start] = "B-action"
        for i in range(start + 1, end + 1):
            tags[i] = "I-action"

    # 2. object / location spans — don't clobber words already tagged as action
    for slot_name, slot_val in [("object", object_val), ("location", location_val)]:
        span = find_phrase_span(words, slot_val)
        if span is None:
            continue
        start, end = span
        if tags[start] != "O":
            continue  # overlaps an action tag, skip rather than corrupt it
        tags[start] = f"B-{slot_name}"
        for i in range(start + 1, end + 1):
            if tags[i] == "O":
                tags[i] = f"I-{slot_name}"

    return words, tags


def load_split(csv_path):
    df = pd.read_csv(csv_path)
    records = []
    unmatched = 0
    for _, row in df.iterrows():
        transcription = str(row["transcription"]).strip()
        action, obj, location = row["action"], row["object"], row["location"]
        intent = f"{action}_{obj}_{location}"
        words, tags = build_bio_tags(transcription, action, obj, location)

        # sanity check: warn if a non-"none" slot value never got matched
        if not any("action" in t for t in tags):
            unmatched += 1
        if str(obj).lower() != NONE_VALUE and not any("object" in t for t in tags):
            unmatched += 1
        if str(location).lower() != NONE_VALUE and not any("location" in t for t in tags):
            unmatched += 1

        records.append({
            "transcription": transcription,
            "tokens": words,
            "slot_tags": tags,
            "intent": intent,
            "action": action,
            "object": obj,
            "location": location,
        })
    if unmatched:
        print(f"[{csv_path.name}] warning: {unmatched} slot values could not be "
              f"matched as substrings and were left untagged (likely synonyms, "
              f"e.g. 'lamp' vs 'lights') — inspect these manually if it matters.")
    return records


def coverage_report(records, split_name):
    """Print how many rows got each slot type successfully tagged, broken
    down by action label, so you can see matching gaps before training."""
    from collections import defaultdict

    stats = defaultdict(lambda: {
        "total": 0,
        "action_expected": 0, "action_matched": 0,
        "object_expected": 0, "object_matched": 0,
        "location_expected": 0, "location_matched": 0,
    })

    for r in records:
        action = r["action"]
        s = stats[action]
        s["total"] += 1

        s["action_expected"] += 1
        if any(t.endswith("action") for t in r["slot_tags"]):
            s["action_matched"] += 1

        if str(r["object"]).lower() != NONE_VALUE:
            s["object_expected"] += 1
            if any(t.endswith("object") for t in r["slot_tags"]):
                s["object_matched"] += 1

        if str(r["location"]).lower() != NONE_VALUE:
            s["location_expected"] += 1
            if any(t.endswith("location") for t in r["slot_tags"]):
                s["location_matched"] += 1

    print(f"\n=== Slot coverage report: {split_name} ({len(records)} rows) ===")
    header = f"{'action':<16}{'rows':>6}{'action%':>10}{'object%':>10}{'location%':>11}"
    print(header)
    print("-" * len(header))

    def pct(matched, expected):
        return "  n/a" if expected == 0 else f"{100 * matched / expected:6.1f}%"

    totals = {"total": 0, "action_expected": 0, "action_matched": 0,
              "object_expected": 0, "object_matched": 0,
              "location_expected": 0, "location_matched": 0}

    for action in sorted(stats):
        s = stats[action]
        for k in totals:
            totals[k] += s[k]
        print(f"{action:<16}{s['total']:>6}"
              f"{pct(s['action_matched'], s['action_expected']):>10}"
              f"{pct(s['object_matched'], s['object_expected']):>10}"
              f"{pct(s['location_matched'], s['location_expected']):>11}")

    print("-" * len(header))
    print(f"{'TOTAL':<16}{totals['total']:>6}"
          f"{pct(totals['action_matched'], totals['action_expected']):>10}"
          f"{pct(totals['object_matched'], totals['object_expected']):>10}"
          f"{pct(totals['location_matched'], totals['location_expected']):>11}")


def build_label_maps(all_records):
    intents = sorted({r["intent"] for r in all_records})
    slot_labels = sorted({t for r in all_records for t in r["slot_tags"]})
    if "O" in slot_labels:
        slot_labels.remove("O")
        slot_labels = ["O"] + slot_labels
    intent2id = {lab: i for i, lab in enumerate(intents)}
    slot2id = {lab: i for i, lab in enumerate(slot_labels)}
    return intent2id, slot2id


def align_labels_with_tokens(tokenizer, tokens, slot_tags, slot2id, max_length):
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    word_ids = encoding.word_ids()
    label_ids = []
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)  # [CLS], [SEP], [PAD]
        elif word_idx != previous_word_idx:
            label_ids.append(slot2id[slot_tags[word_idx]])
        else:
            # subword continuation of a word: convert B- to I- if needed
            tag = slot_tags[word_idx]
            if tag.startswith("B-"):
                tag = "I-" + tag[2:]
            label_ids.append(slot2id.get(tag, slot2id["O"]))
        previous_word_idx = word_idx
    encoding["slot_labels"] = label_ids
    return encoding


def process_split(records, tokenizer, intent2id, slot2id, max_length):
    processed = {
        "input_ids": [], "attention_mask": [], "slot_labels": [],
        "intent_label": [], "transcription": [],
    }
    for r in records:
        enc = align_labels_with_tokens(tokenizer, r["tokens"], r["slot_tags"], slot2id, max_length)
        processed["input_ids"].append(enc["input_ids"])
        processed["attention_mask"].append(enc["attention_mask"])
        processed["slot_labels"].append(enc["slot_labels"])
        processed["intent_label"].append(intent2id[r["intent"]])
        processed["transcription"].append(r["transcription"])
    return Dataset.from_dict(processed)


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_records = load_split(DATA_DIR / "train_data.csv")
    valid_records = load_split(DATA_DIR / "valid_data.csv")
    test_records = load_split(DATA_DIR / "test_data.csv")

    coverage_report(train_records, "train")
    coverage_report(valid_records, "validation")
    coverage_report(test_records, "test")

    # build label maps from the full data so train/val/test share one vocabulary
    intent2id, slot2id = build_label_maps(train_records + valid_records + test_records)

    ds = DatasetDict({
        "train": process_split(train_records, tokenizer, intent2id, slot2id, MAX_LENGTH),
        "validation": process_split(valid_records, tokenizer, intent2id, slot2id, MAX_LENGTH),
        "test": process_split(test_records, tokenizer, intent2id, slot2id, MAX_LENGTH),
    })

    OUT_DIR.mkdir(exist_ok=True, parents=True)
    ds.save_to_disk(str(OUT_DIR / "dataset"))

    with open(OUT_DIR / "intent2id.json", "w") as f:
        json.dump(intent2id, f, indent=2)
    with open(OUT_DIR / "slot2id.json", "w") as f:
        json.dump(slot2id, f, indent=2)

    print(f"\nSaved to {OUT_DIR}/dataset")
    print(f"Intents: {len(intent2id)} | Slot labels: {len(slot2id)}")
    print(f"Train/Val/Test sizes: {len(ds['train'])}/{len(ds['validation'])}/{len(ds['test'])}")


if __name__ == "__main__":
    main()
