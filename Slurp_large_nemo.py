"""
Inference with NVIDIA's slu_conformer_transformer_large_slurp on .flac (or .wav) files.

Works with NeMo 2.7.3 (Python 3.10):
    pip install Cython "nemo_toolkit[asr]==2.7.3"

Usage:
    python slurp_infer.py audio1.flac audio2.flac
"""
import ast
import os
import sys
import tarfile
import urllib.request

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


def load_audio(path):
    """Load FLAC/WAV as mono 16 kHz float32 numpy array."""
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)  # stereo -> mono
    if sr != TARGET_SR:
        import librosa  # installed with nemo_toolkit[asr]
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    return audio


def load_model(path, device):
    """
    The .nemo was saved with NeMo 1.13; its classifier is a bare MultiLayerPerceptron,
    but NeMo 2.x expects a TokenClassifier (error: "... has no attribute 'mlp'").
    Fix: swap the classifier class in the config and remap its weights.
    """
    from omegaconf import open_dict
    from nemo.collections.asr.models import SLUIntentSlotBPEModel

    cfg = SLUIntentSlotBPEModel.restore_from(path, return_config=True)
    if "TokenClassifier" in cfg.classifier.get("_target_", ""):
        return SLUIntentSlotBPEModel.restore_from(path, map_location=device)

    with open_dict(cfg):
        cfg.classifier._target_ = (
            "nemo.collections.asr.parts.submodules.token_classifier.TokenClassifier"
        )
    model = SLUIntentSlotBPEModel.restore_from(
        path, override_config_path=cfg, map_location=device, strict=False
    )

    with tarfile.open(path, "r:*") as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith("model_weights.ckpt"))
        state = torch.load(tar.extractfile(member), map_location=device)
    state = {
        (k.replace("classifier.", "classifier.mlp.", 1) if k.startswith("classifier.layer") else k): v
        for k, v in state.items()
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = [k for k in missing + unexpected
           if k.startswith(("classifier", "encoder", "decoder", "embedding"))]
    if bad:
        raise RuntimeError(f"Weight mismatch after remapping: {bad[:10]}")
    return model


@torch.no_grad()
def predict_file(model, path, device):
    """
    Calls model.predict() directly instead of model.transcribe(): in NeMo 2.x,
    transcribe() calls decoder.freeze(), which this old model's plain
    TransformerDecoder doesn't have ("... has no attribute 'freeze'").
    """
    audio = torch.tensor(load_audio(path), device=device).unsqueeze(0)  # [1, T]
    length = torch.tensor([audio.shape[1]], device=device)
    return model.predict(input_signal=audio, input_signal_length=length)[0]


def parse_prediction(text):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return {"raw": text}  # model occasionally emits malformed output


def main(files):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(download_model(), device)
    model.eval()

    for f in files:
        result = parse_prediction(predict_file(model, f, device))
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
