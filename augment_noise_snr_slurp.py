"""
Augment the SLURP dataset with noise at multiple SNR levels, preserving the
original audio directory structure (slurp_real/, slurp_synth/) for each SNR.

SLURP's official layout (github.com/pswietojanski/slurp):
    audio/
        slurp_real/     <- flat folder of .flac files
        slurp_synth/    <- flat folder of .flac files
    dataset/slurp/
        train.jsonl
        train_synthetic.jsonl
        devel.jsonl
        test.jsonl

This script only touches audio. Point --speech_dir at the "audio" folder
(so it picks up both slurp_real and slurp_synth) or at just one of them.
Since SLURP audio files are flat (no per-speaker subfolders) and every
manifest references files purely by filename, the same .jsonl manifests
will work unmodified against the augmented output -- just point your
data-loading code at output_dir/<SNR>dB/... instead of the original audio
folder. Optionally, use --manifest_dir to copy those manifests alongside
each SNR folder for convenience.

Output layout:
    output_dir/
        0dB/slurp_real/*.flac
        0dB/slurp_synth/*.flac
        5dB/slurp_real/*.flac
        5dB/slurp_synth/*.flac
        ...

Usage:
    python augment_noise_snr_slurp.py \
        --speech_dir /path/to/slurp/audio \
        --noise_dir /path/to/noise_samples \
        --output_dir /path/to/output \
        --snrs -5 0 5 10 15 20 \
        --manifest_dir /path/to/slurp/dataset/slurp

Requires: numpy, soundfile, librosa, tqdm
    pip install numpy soundfile librosa tqdm --break-system-packages
"""

import argparse
import random
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm


# SLURP audio ships as FLAC; keep wav/ogg/mp3 too in case of custom variants
AUDIO_EXTS = {".flac", ".wav", ".mp3", ".ogg"}


def load_audio(path, target_sr=None):
    """Load audio as mono float32, optionally resampled to target_sr."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)  # downmix to mono
    if target_sr is not None and sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return audio, sr


def rms(x, eps=1e-10):
    return np.sqrt(np.mean(x**2) + eps)


def fit_noise_to_length(noise, target_len):
    """Loop or randomly crop noise so it matches target_len samples."""
    n = len(noise)
    if n == 0:
        raise ValueError("Empty noise clip")
    if n < target_len:
        reps = int(np.ceil(target_len / n))
        noise = np.tile(noise, reps)
        n = len(noise)
    if n > target_len:
        start = random.randint(0, n - target_len)
        noise = noise[start : start + target_len]
    else:
        noise = noise[:target_len]
    return noise


def mix_at_snr(speech, noise, snr_db, eps=1e-10):
    """Scale noise so that speech-to-noise ratio equals snr_db, then mix."""
    speech_rms = rms(speech, eps)
    noise_rms = rms(noise, eps)
    target_noise_rms = speech_rms / (10 ** (snr_db / 20.0))
    scale = target_noise_rms / (noise_rms + eps)
    scaled_noise = noise * scale
    mixed = speech + scaled_noise

    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed = mixed / peak

    return mixed.astype(np.float32)


def collect_audio_files(root, exts):
    root = Path(root)
    return [p for p in root.rglob("*") if p.suffix.lower() in exts and p.is_file()]


def copy_manifests(manifest_dir, output_dir, snr_labels):
    """Copy SLURP's .jsonl manifest files into each SNR output folder unchanged
    (filenames inside the manifests match the mirrored audio filenames, so no
    path rewriting is needed as long as your data loader points at the new
    per-SNR audio folder)."""
    manifest_dir = Path(manifest_dir)
    manifest_files = list(manifest_dir.glob("*.jsonl"))
    if not manifest_files:
        print(f"Warning: no .jsonl manifests found in {manifest_dir}")
        return
    for snr_label in snr_labels:
        dest_dir = output_dir / snr_label / "manifests"
        dest_dir.mkdir(parents=True, exist_ok=True)
        for mf in manifest_files:
            shutil.copy2(mf, dest_dir / mf.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--speech_dir", required=True, help="SLURP audio root (contains slurp_real/, slurp_synth/), or one of them")
    parser.add_argument("--noise_dir", required=True, help="Folder containing noise samples")
    parser.add_argument("--output_dir", required=True, help="Where augmented dataset is written")
    parser.add_argument(
        "--snrs",
        type=float,
        nargs="+",
        default=[-5, 0, 5, 10, 15, 20],
        help="List of SNR levels in dB (default: -5 0 5 10 15 20)",
    )
    parser.add_argument("--ext", default=".flac", help="Audio extension to process (default: .flac, SLURP's native format)")
    parser.add_argument("--target_sr", type=int, default=None, help="Resample everything to this sample rate (optional)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--skip_existing", action="store_true", help="Skip files that already exist in output")
    parser.add_argument(
        "--manifest_dir",
        default=None,
        help="Path to SLURP's dataset/slurp folder (with train.jsonl etc.) to copy into each SNR output folder",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    speech_dir = Path(args.speech_dir)
    noise_dir = Path(args.noise_dir)
    output_dir = Path(args.output_dir)

    exts = {args.ext.lower()} if args.ext else AUDIO_EXTS

    speech_files = collect_audio_files(speech_dir, exts)
    noise_files = collect_audio_files(noise_dir, AUDIO_EXTS)

    if not speech_files:
        raise SystemExit(f"No audio files found in {speech_dir} with extensions {exts}")
    if not noise_files:
        raise SystemExit(f"No noise files found in {noise_dir}")

    print(f"Found {len(speech_files)} SLURP speech files and {len(noise_files)} noise files.")
    print(f"SNR levels: {args.snrs}")

    noise_cache = {}

    def get_noise_audio(noise_path, sr):
        key = (str(noise_path), sr)
        if key not in noise_cache:
            noise_cache[key] = load_audio(noise_path, target_sr=sr)[0]
        return noise_cache[key]

    snr_labels = [f"{snr:g}dB" for snr in args.snrs]

    for snr, snr_label in zip(args.snrs, snr_labels):
        print(f"\n=== Processing SNR = {snr_label} ===")

        for speech_path in tqdm(speech_files, desc=snr_label):
            # rel_path preserves e.g. "slurp_real/audio-123.flac" or "slurp_synth/audio-456.flac"
            rel_path = speech_path.relative_to(speech_dir)
            out_path = output_dir / snr_label / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if args.skip_existing and out_path.exists():
                continue

            speech, sr = load_audio(speech_path, target_sr=args.target_sr)

            noise_path = random.choice(noise_files)
            noise = get_noise_audio(noise_path, sr)

            noise_fit = fit_noise_to_length(noise, len(speech))
            mixed = mix_at_snr(speech, noise_fit, snr)

            sf.write(str(out_path), mixed, sr)

    if args.manifest_dir:
        print("\nCopying SLURP manifests into each SNR folder...")
        copy_manifests(args.manifest_dir, output_dir, snr_labels)

    print("\nDone. Augmented SLURP dataset written to:", output_dir)


if __name__ == "__main__":
    main()
