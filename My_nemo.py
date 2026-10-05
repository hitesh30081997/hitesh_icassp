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
SCRIPT_VERSION = "v4 (tuple fix)"


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


def patch_decoder_output(model):
    """NeMo 2.x greedy generator returns a tuple; unwrap it before decoding."""
    sg = model.sequence_generator
    orig = sg.decode_semantics_from_tokens
    sg.decode_semantics_from_tokens = (
        lambda t: orig(t[0] if isinstance(t, (tuple, list)) else t)
    )


@torch.no_grad()
def predict_file(model, path, device):
    """
    Runs the model's prediction steps directly, working around two NeMo 2.x issues:
    - transcribe() calls decoder.freeze(), which this old model's plain
      TransformerDecoder doesn't have ("... has no attribute 'freeze'").
    - predict() assumes the greedy generator returns a tensor, but in 2.x it
      returns a tuple (tokens, ...) ("'tuple' object has no attribute 'detach'").
    """
    from nemo.collections.asr.parts.utils.slu_utils import get_seq_mask

    audio = torch.tensor(load_audio(path), device=device).unsqueeze(0)  # [1, T]
    length = torch.tensor([audio.shape[1]], device=device)

    feats, feats_len = model.preprocessor(input_signal=audio, length=length)
    encoded, encoded_len = model.encoder(audio_signal=feats, length=feats_len)
    encoded = encoded.transpose(1, 2)  # BxDxT -> BxTxD
    mask = get_seq_mask(encoded, encoded_len)

    tokens = model.sequence_generator(encoded, mask)
    if isinstance(tokens, (tuple, list)):
        tokens = tokens[0]  # first element is the generated token tensor
    return model.sequence_generator.decode_semantics_from_tokens(tokens)[0]


def parse_prediction(text):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return {"raw": text}  # model occasionally emits malformed output


def main(files):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"slurp_infer {SCRIPT_VERSION}")
    model = load_model(download_model(), device)
    patch_decoder_output(model)
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
