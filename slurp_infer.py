"""
Inference with NVIDIA's slu_conformer_transformer_large_slurp on .flac (or .wav) files.

Install (SLU was removed from NeMo main, so pin a release that still has it):
    pip install "nemo_toolkit[asr]==2.7.3"

Usage:
    python slurp_infer.py audio1.flac audio2.flac
"""
import ast
import os
import sys
import tempfile
import urllib.request

import numpy as np
import soundfile as sf
import torch

MODEL_URL = (
    "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/"
    "slu_conformer_transformer_large_slurp/versions/1.13.0/files/"
    "slu_conformer_transformer_large_slurp.nemo"
)
MODEL_PATH = "slu_conformer_transformer_large_slurp.nemo"
TARGET_SR = 16000


def download_model(path=MODEL_PATH):
    if not os.path.exists(path):
        print(f"Downloading model to {path} ...")
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


def prepare_audio(src, out_dir):
    """Load FLAC/WAV, downmix to mono, resample to 16 kHz, save as WAV."""
    audio, sr = sf.read(src, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)  # stereo -> mono
    if sr != TARGET_SR:
        import librosa  # installed with nemo_toolkit[asr]
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    out = os.path.join(out_dir, os.path.splitext(os.path.basename(src))[0] + ".wav")
    sf.write(out, audio, TARGET_SR)
    return out


def parse_prediction(pred):
    text = pred.text if hasattr(pred, "text") else str(pred)
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return {"raw": text}  # model occasionally emits malformed output


def main(files):
    from nemo.collections.asr.models import SLUIntentSlotBPEModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # restore_from is used instead of from_pretrained because
    # list_available_models() in this class doesn't return its results.
    model = SLUIntentSlotBPEModel.restore_from(download_model(), map_location=device)
    model.eval()

    with tempfile.TemporaryDirectory() as tmp:
        wavs = [prepare_audio(f, tmp) for f in files]
        preds = model.transcribe(wavs, batch_size=4)

    if isinstance(preds, tuple):  # some NeMo versions return (best, all)
        preds = preds[0]

    for f, p in zip(files, preds):
        result = parse_prediction(p)
        print(f"\n{f}")
        print(f"  scenario: {result.get('scenario')}")
        print(f"  action:   {result.get('action')}")
        for ent in result.get("entities", []):
            print(f"  entity:   {ent.get('type')} = {ent.get('filler')}")
        if "raw" in result:
            print(f"  raw:      {result['raw']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python slurp_infer.py file1.flac [file2.flac ...]")
    main(sys.argv[1:])
