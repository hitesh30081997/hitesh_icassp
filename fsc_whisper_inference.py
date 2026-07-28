"""
Zero-shot ASR inference on the Fluent Speech Commands (FSC) dataset
using a locally downloaded Whisper-small model (fully offline).

Produces:
  1. A CSV with columns: audio_path, actual_transcript, predicted_transcript
  2. Overall WER / CER metrics printed to console and saved to a metrics file.

Requirements (install once, while you have internet access):
    pip install transformers torch torchaudio jiwer pandas soundfile tqdm

-----------------------------------------------------------------------------
FOLDER ASSUMPTIONS (standard FSC release layout):

    FSC_ROOT/
        wavs/
            speakers/.../*.wav
        data/
            train_data.csv
            valid_data.csv
            test_data.csv        <-- has columns: index, path, speakerId,
                                      transcription, action, object, location
            (path column is relative to FSC_ROOT, e.g. "wavs/speakers/xxx/yyy.wav")

If your CSV / folder layout differs, just edit CSV_PATH / AUDIO_ROOT / the
column names in the CONFIG section below.
-----------------------------------------------------------------------------
"""

import os
import time
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import jiwer

# =============================================================================
# CONFIG — edit these paths for your machine
# =============================================================================

# Path to the locally downloaded whisper-small model directory
# (the folder containing config.json, pytorch_model.bin / model.safetensors,
#  tokenizer files, preprocessor_config.json, etc.)
MODEL_DIR = "/path/to/local/whisper-small"

# Root folder of the Fluent Speech Commands dataset
FSC_ROOT = "/path/to/fluent_speech_commands_dataset"

# CSV that lists which audio files to run inference on (e.g. test split)
CSV_PATH = os.path.join(FSC_ROOT, "data", "test_data.csv")

# Column names in that CSV
AUDIO_PATH_COL = "path"            # relative path to the wav file
TRANSCRIPT_COL = "transcription"   # ground-truth transcript

# Where results should be written
OUTPUT_CSV = "/path/to/output/fsc_whisper_predictions.csv"
METRICS_FILE = "/path/to/output/fsc_whisper_metrics.txt"

# Whisper expects 16 kHz mono audio
TARGET_SAMPLE_RATE = 16000

# Force English transcription (FSC is English). Set to None to let Whisper
# auto-detect language instead.
LANGUAGE = "english"
TASK = "transcribe"

# Use GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Optional: limit number of samples for a quick smoke test (None = all rows)
LIMIT_SAMPLES = None

# =============================================================================
# LOAD MODEL + PROCESSOR (fully offline)
# =============================================================================

def load_model():
    print(f"Loading Whisper model from local path: {MODEL_DIR}")
    processor = WhisperProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_DIR, local_files_only=True
    )
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded on device: {DEVICE}")
    return processor, model


# =============================================================================
# AUDIO LOADING / PREPROCESSING
# =============================================================================

def load_audio(file_path, target_sr=TARGET_SAMPLE_RATE):
    """Load a wav file and resample/mono-mix to what Whisper expects."""
    waveform, sr = torchaudio.load(file_path)

    # Convert to mono if needed
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    return waveform.squeeze(0).numpy()


# =============================================================================
# INFERENCE
# =============================================================================

def transcribe_one(processor, model, audio_array):
    input_features = processor(
        audio_array, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt"
    ).input_features.to(DEVICE)

    forced_decoder_ids = None
    if LANGUAGE is not None:
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=LANGUAGE, task=TASK
        )

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features, forced_decoder_ids=forced_decoder_ids
        )

    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return text.strip()


def run_inference():
    processor, model = load_model()

    print(f"Reading metadata CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    if LIMIT_SAMPLES:
        df = df.head(LIMIT_SAMPLES)

    results = []
    start_time = time.time()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Transcribing"):
        rel_path = str(row[AUDIO_PATH_COL]).strip()
        actual_transcript = str(row[TRANSCRIPT_COL]).strip()

        # FSC csv paths are sometimes given with a leading "wavs/..." already
        # relative to FSC_ROOT — join defensively.
        audio_path = rel_path if os.path.isabs(rel_path) else os.path.join(FSC_ROOT, rel_path)

        if not os.path.exists(audio_path):
            print(f"WARNING: missing file, skipping: {audio_path}")
            continue

        try:
            audio_array = load_audio(audio_path)
            predicted_text = transcribe_one(processor, model, audio_array)
        except Exception as e:
            print(f"ERROR processing {audio_path}: {e}")
            predicted_text = ""

        results.append(
            {
                "audio_path": audio_path,
                "actual_transcript": actual_transcript,
                "predicted_transcript": predicted_text,
            }
        )

    elapsed = time.time() - start_time
    print(f"Inference done on {len(results)} files in {elapsed:.1f}s "
          f"({elapsed / max(len(results), 1):.2f}s/file)")

    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Predictions saved to: {OUTPUT_CSV}")

    return results_df


# =============================================================================
# METRICS: WER / CER
# =============================================================================

def compute_metrics(results_df):
    # Drop rows with empty predictions/refs to avoid skewing jiwer
    valid = results_df[
        (results_df["actual_transcript"].str.strip() != "")
        & (results_df["predicted_transcript"].str.strip() != "")
    ]

    refs = valid["actual_transcript"].tolist()
    hyps = valid["predicted_transcript"].tolist()

    # Standard text normalization before scoring
    transform = jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.RemovePunctuation(),
            jiwer.ReduceToListOfListOfWords(),
        ]
    )
    cer_transform = jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.RemovePunctuation(),
            jiwer.ReduceToListOfListOfChars(),
        ]
    )

    wer = jiwer.wer(
        refs, hyps,
        truth_transform=transform,
        hypothesis_transform=transform,
    )
    cer = jiwer.cer(
        refs, hyps,
        truth_transform=cer_transform,
        hypothesis_transform=cer_transform,
    )

    # Per-utterance WER for the CSV (nice for error analysis)
    per_utt_wer = []
    for r, h in zip(refs, hyps):
        try:
            per_utt_wer.append(jiwer.wer(r, h, truth_transform=transform, hypothesis_transform=transform))
        except Exception:
            per_utt_wer.append(None)

    valid = valid.copy()
    valid["utterance_wer"] = per_utt_wer

    summary = (
        f"Total samples scored : {len(valid)}\n"
        f"Overall WER           : {wer * 100:.2f}%\n"
        f"Overall CER           : {cer * 100:.2f}%\n"
    )
    print("\n=== METRICS ===")
    print(summary)

    with open(METRICS_FILE, "w") as f:
        f.write(summary)
    print(f"Metrics saved to: {METRICS_FILE}")

    # Also overwrite the predictions CSV with the per-utterance WER column added
    valid.to_csv(OUTPUT_CSV, index=False)

    return wer, cer


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    results_df = run_inference()
    if len(results_df) > 0:
        compute_metrics(results_df)
    else:
        print("No results produced — check your paths in the CONFIG section.")
