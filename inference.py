from __future__ import annotations

import argparse
import json
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly


SAMPLE_RATE = 16000
DEMO_SECONDS = 6
DEMO_LENGTH = SAMPLE_RATE * DEMO_SECONDS
NUM_CLASSES = 41


def load_labels(path: Path) -> dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    if not isinstance(labels, dict):
        raise ValueError("labels.json must be a mapping from label name to class index.")
    return {str(k): int(v) for k, v in labels.items()}


def resample_audio(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return wav
    factor = gcd(orig_sr, target_sr)
    up = target_sr // factor
    down = orig_sr // factor
    return resample_poly(wav, up, down).astype(np.float32, copy=False)


def load_wav(path: Path) -> torch.Tensor:
    wav, sr = sf.read(str(path), always_2d=True, dtype="float32")
    wav = wav.mean(axis=1)
    wav = resample_audio(wav, sr, SAMPLE_RATE)

    if wav.shape[0] < DEMO_LENGTH:
        wav = np.pad(wav, (0, DEMO_LENGTH - wav.shape[0]))
    elif wav.shape[0] > DEMO_LENGTH:
        wav = wav[:DEMO_LENGTH]

    return torch.from_numpy(np.ascontiguousarray(wav)).float()


def make_label(label_name: str, labels: dict[str, int]) -> torch.Tensor:
    if label_name not in labels:
        available = ", ".join(labels.keys())
        raise ValueError(f"Unknown label '{label_name}'. Available labels: {available}")
    label_id = labels[label_name]
    if label_id < 0 or label_id >= NUM_CLASSES:
        raise ValueError(f"Invalid class index for '{label_name}': {label_id}")

    label = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    label[label_id] = 1.0
    return label


def save_wav(path: Path, wav: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = wav.detach().cpu().float().view(-1).clamp(-1.0, 1.0).numpy()
    sf.write(str(path), wav, SAMPLE_RATE)


def main() -> None:
    parser = argparse.ArgumentParser(description="LCI-GCF inference-only demo")
    parser.add_argument("--input", required=True, type=Path, help="Path to an input mixture wav.")
    parser.add_argument("--label", required=True, type=str, help="Target class name, e.g., Snare_drum.")
    parser.add_argument("--output", required=True, type=Path, help="Path to save the extracted wav.")
    parser.add_argument("--model", default=Path("checkpoints/lci_gcf_demo.pt"), type=Path)
    parser.add_argument("--labels", default=Path("labels.json"), type=Path)
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available. CPU is used by default.")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    labels = load_labels(args.labels)
    mixture = load_wav(args.input).unsqueeze(0).to(device)
    label = make_label(args.label, labels).unsqueeze(0).to(device)

    model = torch.jit.load(str(args.model), map_location=device)
    model.eval()

    with torch.no_grad():
        estimate = model(mixture, label).squeeze(0)

    save_wav(args.output, estimate)
    print(f"Saved output to: {args.output}")


if __name__ == "__main__":
    main()

