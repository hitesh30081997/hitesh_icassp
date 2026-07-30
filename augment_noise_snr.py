"""
Augment a fluent-speech dataset with noise at multiple SNR levels,
preserving the original directory structure for each SNR.

Output layout:
    output_dir/
        0dB/<same subfolders/filenames as speech_dir>
        5dB/...
        10dB/...
        ...

Usage:
    python augment_noise_snr.py \
        --speech_dir /path/to/fluent_speech \
        --noise_dir /path/to/noise_samples \
        --output_dir /path/to/output \
        --snrs -5 0 5 10 15 20 \
        --ext .wav

Requires: numpy, soundfile, librosa (for resampling)
    pip install numpy soundfile librosa tqdm --break-system-packages
"""

import argparse
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}


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
    # desired noise rms: speech_rms / 10^(snr/20)
    target_noise_rms = speech_rms / (10 ** (snr_db / 20.0))
    scale = target_noise_rms / (noise_rms + eps)
    scaled_noise = noise * scale
    mixed = speech + scaled_noise

    # peak-normalize if clipping would occur
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed = mixed / peak

    return mixed.astype(np.float32)


def collect_audio_files(root, exts):
    root = Path(root)
    return [p for p in root.rglob("*") if p.suffix.lower() in exts and p.is_file()]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--speech_dir", required=True, help="Root of the clean fluent speech dataset")
    parser.add_argument("--noise_dir", required=True, help="Folder containing noise samples")
    parser.add_argument("--output_dir", required=True, help="Where augmented dataset is written")
    parser.add_argument(
        "--snrs",
        type=float,
        nargs="+",
        default=[-5, 0, 5, 10, 15, 20],
        help="List of SNR levels in dB (default: -5 0 5 10 15 20)",
    )
    parser.add_argument("--ext", default=None, help="Only process this extension (e.g. .wav). Default: any of %s" % AUDIO_EXTS)
    parser.add_argument("--target_sr", type=int, default=None, help="Resample everything to this sample rate (optional)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--skip_existing", action="store_true", help="Skip files that already exist in output")
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

    print(f"Found {len(speech_files)} speech files and {len(noise_files)} noise files.")
    print(f"SNR levels: {args.snrs}")

    # Cache loaded noise files in memory to avoid re-reading repeatedly for large noise sets
    noise_cache = {}

    def get_noise_audio(noise_path, sr):
        key = (str(noise_path), sr)
        if key not in noise_cache:
            noise_cache[key] = load_audio(noise_path, target_sr=sr)[0]
        return noise_cache[key]

    for snr in args.snrs:
        snr_label = f"{snr:g}dB"
        print(f"\n=== Processing SNR = {snr_label} ===")

        for speech_path in tqdm(speech_files, desc=snr_label):
            rel_path = speech_path.relative_to(speech_dir)
            out_path = output_dir / snr_label / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if args.skip_existing and out_path.exists():
                continue

            speech, sr = load_audio(speech_path, target_sr=args.target_sr)

            # Randomly pick a noise sample for this speech file at this SNR
            noise_path = random.choice(noise_files)
            noise = get_noise_audio(noise_path, sr)

            noise_fit = fit_noise_to_length(noise, len(speech))
            mixed = mix_at_snr(speech, noise_fit, snr)

            sf.write(str(out_path), mixed, sr)

    print("\nDone. Augmented dataset written to:", output_dir)


if __name__ == "__main__":
    main()
